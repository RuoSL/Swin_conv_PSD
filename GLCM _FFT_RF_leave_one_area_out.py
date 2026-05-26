import os
import cv2
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from typing import List, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.feature import graycomatrix, graycoprops
from scipy.fft import fft2, fftshift


AREAS = ["1_patch", "3_patch", "o2_1_1_tile"]


# =========================
# Feature Extraction
# =========================
def extract_basic_stats(gray: np.ndarray) -> List[float]:
    gray_f = gray.astype(np.float32)
    mean_val = float(np.mean(gray_f))
    std_val = float(np.std(gray_f))
    min_val = float(np.min(gray_f))
    max_val = float(np.max(gray_f))
    med_val = float(np.median(gray_f))

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    lap_var = float(np.var(lap))

    return [mean_val, std_val, min_val, max_val, med_val, grad_mean, grad_std, lap_var]


def extract_glcm_features(gray: np.ndarray, glcm_levels: int = 32) -> List[float]:
    """
    先把灰度从 0~255 量化到更少级别，避免 GLCM 过稀疏。
    推荐先用 32 级，后续可尝试 16 / 64 做对比。
    """
    gray_f = gray.astype(np.float32)
    gray_q = np.floor(gray_f / (256.0 / glcm_levels)).astype(np.uint8)
    gray_q = np.clip(gray_q, 0, glcm_levels - 1)

    distances = [1, 2, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        gray_q,
        distances=distances,
        angles=angles,
        levels=glcm_levels,
        symmetric=True,
        normed=True
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for p in props:
        vals = graycoprops(glcm, p)
        feats.extend(vals.flatten().tolist())

    return feats


def radial_profile(data: np.ndarray, num_bins: int = 16) -> List[float]:
    h, w = data.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_max = np.max(r)
    bins = np.linspace(0, r_max, num_bins + 1)

    profile = []
    for i in range(num_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(mask):
            profile.append(float(np.mean(data[mask])))
        else:
            profile.append(0.0)
    return profile


def extract_fft_features(gray: np.ndarray) -> List[float]:
    gray_f = gray.astype(np.float32)

    f = fft2(gray_f)
    fshift = fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)

    h, w = mag_log.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_max = np.max(r)
    low_mask = r < r_max * 0.15
    mid_mask = (r >= r_max * 0.15) & (r < r_max * 0.4)
    high_mask = r >= r_max * 0.4

    low_energy = float(np.mean(mag_log[low_mask]))
    mid_energy = float(np.mean(mag_log[mid_mask]))
    high_energy = float(np.mean(mag_log[high_mask]))
    total_energy = float(np.mean(mag_log))

    radial_feats = radial_profile(mag_log, num_bins=16)

    return [low_energy, mid_energy, high_energy, total_energy] + radial_feats


def extract_features(img_path: str, resize=(224, 224), glcm_levels: int = 32) -> np.ndarray:
    img = Image.open(img_path).convert("L")
    img = img.resize(resize)
    gray = np.array(img)

    feats = []
    feats.extend(extract_basic_stats(gray))
    feats.extend(extract_glcm_features(gray, glcm_levels=glcm_levels))
    feats.extend(extract_fft_features(gray))

    return np.array(feats, dtype=np.float32)


# =========================
# Area helpers
# =========================
def get_area_from_name(fname: str) -> str:
    if fname.startswith("1_patch"):
        return "1_patch"
    elif fname.startswith("3_patch"):
        return "3_patch"
    elif fname.startswith("o2_1_1_tile"):
        return "o2_1_1_tile"
    else:
        raise ValueError(f"Unknown area for file: {fname}")


# =========================
# Target transform
# =========================
def transform_targets_to_monotonic_deltas(y: np.ndarray) -> np.ndarray:
    """
    原始:
        [D20, D40, D60, D80, max]
    转成:
        [D20, D40-D20, D60-D40, D80-D60, max-D80]
    """
    y_new = np.zeros_like(y, dtype=np.float32)
    y_new[:, 0] = y[:, 0]
    y_new[:, 1] = y[:, 1] - y[:, 0]
    y_new[:, 2] = y[:, 2] - y[:, 1]
    y_new[:, 3] = y[:, 3] - y[:, 2]
    y_new[:, 4] = y[:, 4] - y[:, 3]
    return y_new


def inverse_monotonic_deltas(y_delta_pred: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    预测的是:
        [D20, d1, d2, d3, d4]
    其中 d1..d4 应 >= 0
    恢复成:
        [D20, D40, D60, D80, max]
    为保证严格递增，这里把增量截断到 >= eps
    """
    y_delta_pred = y_delta_pred.copy().astype(np.float32)

    d20 = y_delta_pred[:, 0]
    inc1 = np.maximum(y_delta_pred[:, 1], eps)
    inc2 = np.maximum(y_delta_pred[:, 2], eps)
    inc3 = np.maximum(y_delta_pred[:, 3], eps)
    inc4 = np.maximum(y_delta_pred[:, 4], eps)

    d40 = d20 + inc1
    d60 = d40 + inc2
    d80 = d60 + inc3
    dmax = d80 + inc4

    y_pred = np.stack([d20, d40, d60, d80, dmax], axis=1)
    return y_pred


# =========================
# Dataset Loading
# =========================
def load_dataset(data_dir: str, ann_file: str):
    df = pd.read_csv(ann_file)
    print("CSV columns:", df.columns.tolist())

    targets_dict: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        targets_dict[row["image_name"]] = [
            row["D20"], row["D40"], row["D60"], row["D80"], row["max"]
        ]

    image_paths = []
    targets = []
    image_names = []
    image_areas = []

    for fname in os.listdir(data_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")) and fname in targets_dict:
            image_paths.append(os.path.join(data_dir, fname))
            targets.append(targets_dict[fname])
            image_names.append(fname)
            image_areas.append(get_area_from_name(fname))

    return image_paths, np.array(targets, dtype=np.float32), image_names, image_areas


# =========================
# Metrics
# =========================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    mae_each = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_each = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mape_each = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)), axis=0) * 100

    r2_each = []
    for i in range(y_true.shape[1]):
        r2_each.append(r2_score(y_true[:, i], y_pred[:, i]))
    r2_each = np.array(r2_each, dtype=np.float32)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_global = 1.0 - ss_res / (ss_tot + 1e-8)

    return {
        "test_psd_mae": float(mae),
        "test_psd_rmse": float(rmse),
        "test_psd_mape": float(mape),
        "test_psd_r2": float(r2_global),

        "test_psd_D20_mae": float(mae_each[0]),
        "test_psd_D40_mae": float(mae_each[1]),
        "test_psd_D60_mae": float(mae_each[2]),
        "test_psd_D80_mae": float(mae_each[3]),
        "test_psd_max_mae": float(mae_each[4]),

        "test_psd_D20_rmse": float(rmse_each[0]),
        "test_psd_D40_rmse": float(rmse_each[1]),
        "test_psd_D60_rmse": float(rmse_each[2]),
        "test_psd_D80_rmse": float(rmse_each[3]),
        "test_psd_max_rmse": float(rmse_each[4]),

        "test_psd_D20_mape": float(mape_each[0]),
        "test_psd_D40_mape": float(mape_each[1]),
        "test_psd_D60_mape": float(mape_each[2]),
        "test_psd_D80_mape": float(mape_each[3]),
        "test_psd_max_mape": float(mape_each[4]),

        "test_psd_D20_r2": float(r2_each[0]),
        "test_psd_D40_r2": float(r2_each[1]),
        "test_psd_D60_r2": float(r2_each[2]),
        "test_psd_D80_r2": float(r2_each[3]),
        "test_psd_max_r2": float(r2_each[4]),
    }


def rank_violation_rate(y_pred: np.ndarray):
    violations = 0
    for i in range(y_pred.shape[0]):
        pred = y_pred[i]
        if np.any(pred[:-1] >= pred[1:]):
            violations += 1
    return violations / len(y_pred)


# =========================
# Leave-one-area-out
# =========================
def run_leave_one_area_out(
    data_dir,
    ann_file,
    n_estimators=300,
    max_depth=None,
    glcm_levels=32,
    output_dir="./rf_results/glcm_fft_rf_new_leave_one_area_out_monotonic"
):
    os.makedirs(output_dir, exist_ok=True)

    image_paths, targets, image_names, image_areas = load_dataset(data_dir, ann_file)
    print(f"Total dataset size: {len(image_paths)}")

    print("Extracting handcrafted features...")
    X = np.stack([extract_features(p, glcm_levels=glcm_levels) for p in image_paths], axis=0)

    y_raw = targets
    y_train_style = transform_targets_to_monotonic_deltas(y_raw)
    areas = np.array(image_areas)

    print(f"Feature shape: {X.shape}")
    print(f"Target shape (raw): {y_raw.shape}")
    print(f"Target shape (delta): {y_train_style.shape}")
    print(f"GLCM levels: {glcm_levels}")

    all_fold_metrics = []
    all_pred_records = []

    for test_area in AREAS:
        train_areas = [a for a in AREAS if a != test_area]

        train_idx = np.where(areas != test_area)[0]
        test_idx = np.where(areas == test_area)[0]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_delta = y_train_style[train_idx]
        y_test_raw = y_raw[test_idx]
        names_test = [image_names[i] for i in test_idx]

        print(f"\n===== Leave-one-area-out: test={test_area} =====")
        print(f"Train areas: {train_areas}")
        print(f"Test area: {test_area}")
        print(f"Train size: {len(train_idx)}")
        print(f"Test size: {len(test_idx)}")

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train_delta)

        y_pred_delta = model.predict(X_test)
        y_pred_raw = inverse_monotonic_deltas(y_pred_delta, eps=1e-6)

        metrics = compute_metrics(y_test_raw, y_pred_raw)
        metrics["test_rank_violation_rate"] = rank_violation_rate(y_pred_raw)
        metrics["fold"] = AREAS.index(test_area) + 1
        metrics["mode"] = "glcm_fft_rf"
        metrics["test_area"] = test_area
        metrics["train_areas"] = "+".join(train_areas)
        all_fold_metrics.append(metrics)

        print(f"MAE: {metrics['test_psd_mae']:.6f}")
        print(f"RMSE: {metrics['test_psd_rmse']:.6f}")
        print(f"MAPE: {metrics['test_psd_mape']:.6f}%")
        print(f"R2: {metrics['test_psd_r2']:.6f}")
        print(f"Rank violation rate: {metrics['test_rank_violation_rate']:.4f}")

        pred_df = pd.DataFrame({
            "image_name": names_test,
            "test_area": test_area,
            "train_areas": "+".join(train_areas),
            "fold": AREAS.index(test_area) + 1,

            "D20_gt": y_test_raw[:, 0],
            "D40_gt": y_test_raw[:, 1],
            "D60_gt": y_test_raw[:, 2],
            "D80_gt": y_test_raw[:, 3],
            "max_gt": y_test_raw[:, 4],

            "D20_pred": y_pred_raw[:, 0],
            "D40_pred": y_pred_raw[:, 1],
            "D60_pred": y_pred_raw[:, 2],
            "D80_pred": y_pred_raw[:, 3],
            "max_pred": y_pred_raw[:, 4],

            "inc1_pred": np.maximum(y_pred_delta[:, 1], 1e-6),
            "inc2_pred": np.maximum(y_pred_delta[:, 2], 1e-6),
            "inc3_pred": np.maximum(y_pred_delta[:, 3], 1e-6),
            "inc4_pred": np.maximum(y_pred_delta[:, 4], 1e-6),
        })
        pred_path = os.path.join(output_dir, f"test_{test_area}_pred_vs_gt.csv")
        pred_df.to_csv(pred_path, index=False)
        all_pred_records.append(pred_df)

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(output_dir, "per_area_metrics.csv"), index=False)
    metrics_df.to_excel(os.path.join(output_dir, "per_area_metrics.xlsx"), index=False)

    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()

    mean_row = metrics_df[numeric_cols].mean().to_dict()
    std_row = metrics_df[numeric_cols].std(ddof=0).to_dict()

    summary_df = pd.DataFrame([
        {"stat": "mean", **mean_row},
        {"stat": "std", **std_row},
    ])
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    summary_df.to_excel(os.path.join(output_dir, "metrics_summary.xlsx"), index=False)

    all_preds_df = pd.concat(all_pred_records, axis=0, ignore_index=True)
    all_preds_df.to_csv(os.path.join(output_dir, "all_pred_vs_gt.csv"), index=False)

    print("\n===== Leave-one-area-out Average Results =====")
    for col in numeric_cols:
        print(f"{col:25s}: {mean_row[col]:.6f} ± {std_row[col]:.6f}")

    print(f"\nSaved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./PSD_dataset/images")
    parser.add_argument("--ann_file", type=str, default="./PSD_dataset/annotations.csv")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--glcm_levels", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./rf_results/glcm_fft_rf_new_leave_one_area_out_monotonic")
    args = parser.parse_args()

    run_leave_one_area_out(
        data_dir=args.data_dir,
        ann_file=args.ann_file,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        glcm_levels=args.glcm_levels,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
