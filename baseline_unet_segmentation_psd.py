import argparse
import csv
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF


SEED = 42
PSD_LABELS = ["D20", "D40", "D60", "D80", "max"]
AREA_ORDER = ["1_patch", "3_patch", "o2_1_1_tile"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_area_from_name(filename: str) -> str:
    name = os.path.splitext(filename)[0].lower()
    if name.startswith("o2_1_1_tile"):
        return "o2_1_1_tile"
    if name.startswith("1_patch"):
        return "1_patch"
    if name.startswith("3_patch"):
        return "3_patch"
    raise ValueError(f"Cannot infer area from filename: {filename}")


def find_mask_name(mask_dir: Path, image_name: str) -> Optional[str]:
    stem = Path(image_name).stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    return None


class PairedSegTransform:
    def __init__(self, size: int = 512, train: bool = True):
        self.size = (size, size)
        self.train = train
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.3:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            angle = random.uniform(-15.0, 15.0)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, self.mean, self.std)

        mask_t = torch.from_numpy(np.array(mask, dtype=np.uint8)).float()
        mask_t = (mask_t > 127).float().unsqueeze(0)
        return image_t, mask_t


class RockSegDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, ann_file: Path, transform: Optional[PairedSegTransform]):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        df = pd.read_csv(ann_file)
        required = {"image_name", "D20", "D40", "D60", "D80", "max"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in annotation file: {sorted(missing)}")

        self.samples = []
        missing_images = []
        missing_masks = []
        for _, row in df.iterrows():
            image_name = str(row["image_name"])
            if "mask_name" in df.columns:
                mask_name = str(row["mask_name"])
            else:
                mask_name = find_mask_name(mask_dir, image_name)
                if mask_name is None:
                    mask_name = Path(image_name).with_suffix(".png").name

            image_path = image_dir / image_name
            mask_path = mask_dir / mask_name
            image_exists = image_path.exists()
            mask_exists = mask_path.exists()
            if not image_exists:
                missing_images.append(image_name)
            if not mask_exists:
                missing_masks.append(mask_name)
            if not image_exists or not mask_exists:
                continue

            psd = [float(row[label]) for label in PSD_LABELS]
            if any(np.isnan(psd)):
                continue

            self.samples.append(
                {
                    "image_name": image_name,
                    "mask_name": mask_name,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "area": get_area_from_name(image_name),
                    "psd": psd,
                }
            )

        self.areas = [s["area"] for s in self.samples]
        print(f"Loaded samples: {len(self.samples)}")
        if len(self.samples) == 0:
            print(f"Annotation rows: {len(df)}")
            print(f"Image dir: {image_dir}")
            print(f"Mask dir: {mask_dir}")
            print(f"Missing images: {len(missing_images)}")
            print(f"Missing masks: {len(missing_masks)}")
            if missing_images:
                print("First missing images:", missing_images[:5])
            if missing_masks:
                print("First missing masks:", missing_masks[:5])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")

        if self.transform:
            image_t, mask_t = self.transform(image, mask)
        else:
            image_t = TF.to_tensor(image)
            mask_t = torch.from_numpy(np.array(mask, dtype=np.uint8)).float().unsqueeze(0) / 255.0

        return {
            "image": image_t,
            "mask": mask_t,
            "psd": torch.tensor(sample["psd"], dtype=torch.float32),
            "image_name": sample["image_name"],
        }


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)
        self.out = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    return bce + dice_loss(logits, masks)


def compute_segmentation_metrics(logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> Dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).float()
    masks = (masks >= 0.5).float()

    tp = torch.sum((pred == 1) & (masks == 1)).item()
    tn = torch.sum((pred == 0) & (masks == 0)).item()
    fp = torch.sum((pred == 1) & (masks == 0)).item()
    fn = torch.sum((pred == 0) & (masks == 1)).item()

    eps = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    foreground_iou = tp / (tp + fp + fn + eps)
    background_iou = tn / (tn + fp + fn + eps)
    miou = 0.5 * (foreground_iou + background_iou)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(foreground_iou),
        "foreground_iou": float(foreground_iou),
        "background_iou": float(background_iou),
        "miou": float(miou),
        "dice": float(dice),
    }


def weighted_percentile_from_areas(diameters: np.ndarray, areas: np.ndarray, quantiles: List[float]) -> np.ndarray:
    if len(diameters) == 0 or areas.sum() <= 0:
        return np.full(len(quantiles), np.nan, dtype=np.float64)

    order = np.argsort(diameters)
    diameters = diameters[order]
    areas = areas[order]
    cumulative_fraction = np.cumsum(areas) / areas.sum()

    curve_x = np.concatenate(([0.0], cumulative_fraction))
    curve_y = np.concatenate(([diameters[0]], diameters))
    return np.interp(np.array(quantiles, dtype=np.float64), curve_x, curve_y)


def psd_from_binary_mask(mask: np.ndarray, min_area: float = 1.0) -> Optional[Dict[str, float]]:
    """Match build_psd_dataset_from_via.py mask_percentile PSD calculation."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    diameters = []
    areas = []
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        areas.append(area)
        diameters.append(2.0 * math.sqrt(area / math.pi))

    if len(diameters) == 0:
        return None

    diam_np = np.array(diameters, dtype=np.float64)
    areas_np = np.array(areas, dtype=np.float64)
    d20, d40, d60, d80, dmax = np.percentile(diam_np, [20, 40, 60, 80, 100])

    return {
        "D20": float(d20),
        "D40": float(d40),
        "D60": float(d60),
        "D80": float(d80),
        "max": float(dmax),
        "particle_count": int(len(diam_np)),
        "total_area_px2": float(areas_np.sum()),
    }


def split_indices(dataset: RockSegDataset, fold_idx: int) -> Tuple[List[int], List[int], List[int], str, List[str]]:
    all_indices = np.arange(len(dataset))
    all_areas = np.array(dataset.areas)

    test_area = AREA_ORDER[fold_idx]
    train_areas = [a for a in AREA_ORDER if a != test_area]
    train_val_indices = all_indices[np.isin(all_areas, train_areas)]
    test_indices = all_indices[all_areas == test_area]

    rng = np.random.default_rng(SEED)
    shuffled = train_val_indices.copy()
    rng.shuffle(shuffled)
    val_size = int(len(shuffled) * 0.15)

    val_indices = shuffled[:val_size].tolist()
    train_indices = shuffled[val_size:].tolist()
    return train_indices, val_indices, test_indices.tolist(), test_area, train_areas


def make_loader(
    dataset: RockSegDataset,
    indices: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, amp: bool, threshold: float) -> Dict[str, float]:
    model.train()
    losses = []
    metric_values = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = segmentation_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics = compute_segmentation_metrics(logits.detach(), masks, threshold)
        losses.append(loss.detach().item())
        metric_values.append(metrics)

    out = {"loss": float(np.mean(losses))}
    for key in metric_values[0]:
        out[key] = float(np.mean([m[key] for m in metric_values]))
    return out


@torch.no_grad()
def evaluate_segmentation(model, loader, device, amp: bool, threshold: float) -> Dict[str, float]:
    model.eval()
    losses = []
    metric_values = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = segmentation_loss(logits, masks)

        metrics = compute_segmentation_metrics(logits, masks, threshold)
        losses.append(loss.item())
        metric_values.append(metrics)

    out = {"loss": float(np.mean(losses))}
    for key in metric_values[0]:
        out[key] = float(np.mean([m[key] for m in metric_values]))
    return out


@torch.no_grad()
def predict_test_psd(
    model,
    loader,
    device,
    amp: bool,
    threshold: float,
    min_area: float,
    pred_mask_dir: Path,
) -> pd.DataFrame:
    model.eval()
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        gts = batch["psd"].cpu().numpy()
        gt_masks = batch["mask"].cpu().numpy()
        names = batch["image_name"]

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)

        probs = torch.sigmoid(logits).cpu().numpy()
        for i, image_name in enumerate(names):
            pred_mask = (probs[i, 0] >= threshold).astype(np.uint8) * 255
            mask_name = f"{Path(image_name).stem}_unet_pred.png"
            cv2.imwrite(str(pred_mask_dir / mask_name), pred_mask)

            pred_psd = psd_from_binary_mask(pred_mask, min_area=min_area)
            gt_mask = (gt_masks[i, 0] > 0.5).astype(np.uint8) * 255
            gt_mask_psd = psd_from_binary_mask(gt_mask, min_area=min_area)
            row = {
                "image_name": image_name,
                "pred_mask_name": mask_name,
                "D20_gt": float(gts[i, 0]),
                "D40_gt": float(gts[i, 1]),
                "D60_gt": float(gts[i, 2]),
                "D80_gt": float(gts[i, 3]),
                "max_gt": float(gts[i, 4]),
            }
            if gt_mask_psd is None:
                for label in PSD_LABELS:
                    row[f"{label}_mask_psd"] = np.nan
                row["gt_mask_particle_count"] = 0
                row["gt_mask_total_area_px2"] = 0.0
            else:
                for label in PSD_LABELS:
                    row[f"{label}_mask_psd"] = gt_mask_psd[label]
                row["gt_mask_particle_count"] = gt_mask_psd["particle_count"]
                row["gt_mask_total_area_px2"] = gt_mask_psd["total_area_px2"]

            if pred_psd is None:
                for label in PSD_LABELS:
                    row[f"{label}_pred"] = np.nan
                row["pred_particle_count"] = 0
                row["pred_total_area_px2"] = 0.0
            else:
                for label in PSD_LABELS:
                    row[f"{label}_pred"] = pred_psd[label]
                row["pred_particle_count"] = pred_psd["particle_count"]
                row["pred_total_area_px2"] = pred_psd["total_area_px2"]
            rows.append(row)

    return pd.DataFrame(rows)


def regression_metrics(
    pred_df: pd.DataFrame,
    pred_suffix: str = "pred",
    metric_prefix: str = "test",
) -> Dict[str, float]:
    out = {}
    pred_cols = [f"{label}_{pred_suffix}" for label in PSD_LABELS]
    valid = pred_df.dropna(subset=pred_cols).copy()
    out[f"{metric_prefix}_valid_count"] = float(len(valid))
    out[f"{metric_prefix}_missing_count"] = float(len(pred_df) - len(valid))

    if len(valid) == 0:
        for metric in ["mae", "rmse", "mape", "r2"]:
            out[f"{metric_prefix}_psd_{metric}"] = np.nan
        return out

    y_true = valid[[f"{label}_gt" for label in PSD_LABELS]].to_numpy(dtype=np.float64)
    y_pred = valid[pred_cols].to_numpy(dtype=np.float64)

    abs_err = np.abs(y_pred - y_true)
    sq_err = (y_pred - y_true) ** 2
    out[f"{metric_prefix}_psd_mae"] = float(np.mean(abs_err))
    out[f"{metric_prefix}_psd_rmse"] = float(np.sqrt(np.mean(sq_err)))
    out[f"{metric_prefix}_psd_mape"] = float(np.mean(abs_err / (np.abs(y_true) + 1e-8)) * 100.0)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    out[f"{metric_prefix}_psd_r2"] = float(1.0 - ss_res / (ss_tot + 1e-8))

    for idx, label in enumerate(PSD_LABELS):
        out[f"{metric_prefix}_psd_{label}_mae"] = float(np.mean(abs_err[:, idx]))
        out[f"{metric_prefix}_psd_{label}_rmse"] = float(np.sqrt(np.mean(sq_err[:, idx])))
        out[f"{metric_prefix}_psd_{label}_mape"] = float(np.mean(abs_err[:, idx] / (np.abs(y_true[:, idx]) + 1e-8)) * 100.0)
        ss_res_i = np.sum((y_true[:, idx] - y_pred[:, idx]) ** 2)
        ss_tot_i = np.sum((y_true[:, idx] - np.mean(y_true[:, idx])) ** 2)
        out[f"{metric_prefix}_psd_{label}_r2"] = float(1.0 - ss_res_i / (ss_tot_i + 1e-8))

    violations = np.any(y_pred[:, :-1] > y_pred[:, 1:], axis=1)
    out[f"{metric_prefix}_rank_violation_rate"] = float(np.mean(violations))
    return out


def write_summary(per_fold_df: pd.DataFrame, out_dir: Path) -> None:
    num_cols = per_fold_df.select_dtypes(include=[np.number]).columns
    mean_row = per_fold_df[num_cols].mean().to_frame().T
    std_row = per_fold_df[num_cols].std(ddof=0).to_frame().T
    mean_row.insert(0, "stat", "mean")
    std_row.insert(0, "stat", "std")
    summary = pd.concat([mean_row, std_row], ignore_index=True)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_excel(out_dir / "metrics_summary.xlsx", index=False)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp = device.type == "cuda" and args.amp
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")

    out_dir = Path(args.output_dir) / "unet_segmentation_contour_psd" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {out_dir}")

    data_dir = args.image_dir or args.data_dir
    print(f"Image/Data dir: {data_dir}")
    print(f"Mask dir: {args.mask_dir}")
    print(f"Annotation file: {args.ann_file}")

    train_dataset = RockSegDataset(
        Path(data_dir),
        Path(args.mask_dir),
        Path(args.ann_file),
        PairedSegTransform(size=args.image_size, train=True),
    )
    eval_dataset = RockSegDataset(
        Path(data_dir),
        Path(args.mask_dir),
        Path(args.ann_file),
        PairedSegTransform(size=args.image_size, train=False),
    )

    per_fold_records = []
    all_pred_records = []

    for fold_idx in range(args.splits):
        train_idx, val_idx, test_idx, test_area, train_areas = split_indices(eval_dataset, fold_idx)
        print(f"\n===== Fold {fold_idx + 1}/{args.splits} =====")
        print(f"Train areas: {train_areas}")
        print(f"Test area: {test_area}")
        print(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

        train_loader = make_loader(train_dataset, train_idx, args.batch_size, True, args.num_workers)
        val_loader = make_loader(eval_dataset, val_idx, args.batch_size, False, args.num_workers)
        test_loader = make_loader(eval_dataset, test_idx, args.batch_size, False, args.num_workers)

        model = UNet(base_ch=args.base_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        fold_dir = out_dir / f"fold{fold_idx + 1}_{test_area}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = fold_dir / "best_unet.pt"
        history = []
        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, amp, args.threshold)
            val_metrics = evaluate_segmentation(model, val_loader, device, amp, args.threshold)
            scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_miou": train_metrics["miou"],
                "train_iou": train_metrics["iou"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_miou": val_metrics["miou"],
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(record)
            print(
                f"[Fold {fold_idx + 1}][Epoch {epoch:03d}] "
                f"TrainLoss={record['train_loss']:.4f}, ValLoss={record['val_loss']:.4f}, "
                f"ValIoU={record['val_iou']:.4f}, ValDice={record['val_dice']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "args": vars(args),
                    },
                    ckpt_path,
                )

        pd.DataFrame(history).to_csv(fold_dir / "loss_curve.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(history).to_excel(fold_dir / "loss_curve.xlsx", index=False)

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        pred_df = predict_test_psd(
            model,
            test_loader,
            device,
            amp,
            args.threshold,
            args.min_area,
            fold_dir / "pred_masks",
        )
        pred_df.insert(1, "test_area", test_area)
        pred_df.insert(2, "train_areas", "+".join(train_areas))
        pred_df.insert(3, "fold", fold_idx + 1)
        pred_df.to_csv(fold_dir / "pred_vs_gt.csv", index=False, encoding="utf-8-sig")
        pred_df.to_excel(fold_dir / "pred_vs_gt.xlsx", index=False)
        all_pred_records.append(pred_df)
        all_pred_df = pd.concat(all_pred_records, axis=0, ignore_index=True)
        all_pred_df.to_csv(out_dir / "all_pred_vs_gt.csv", index=False, encoding="utf-8-sig")
        all_pred_df.to_excel(out_dir / "all_pred_vs_gt.xlsx", index=False)

        test_seg = evaluate_segmentation(model, test_loader, device, amp, args.threshold)
        fold_record = {
            "fold": fold_idx + 1,
            "mode": "unet_segmentation_contour_psd",
            "test_area": test_area,
            "train_areas": "+".join(train_areas),
            "best_val_loss": best_val_loss,
            "test_seg_loss": test_seg["loss"],
            "test_seg_accuracy": test_seg["accuracy"],
            "test_seg_precision": test_seg["precision"],
            "test_seg_recall": test_seg["recall"],
            "test_seg_miou": test_seg["miou"],
            "test_seg_iou": test_seg["iou"],
            "test_seg_foreground_iou": test_seg["foreground_iou"],
            "test_seg_background_iou": test_seg["background_iou"],
            "test_seg_dice": test_seg["dice"],
        }
        fold_record.update(regression_metrics(pred_df, pred_suffix="pred", metric_prefix="test"))
        fold_record.update(regression_metrics(pred_df, pred_suffix="mask_psd", metric_prefix="oracle_mask"))
        per_fold_records.append(fold_record)

        per_fold_df = pd.DataFrame(per_fold_records)
        per_fold_df.to_csv(out_dir / "per_fold_metrics.csv", index=False, encoding="utf-8-sig")
        per_fold_df.to_excel(out_dir / "per_fold_metrics.xlsx", index=False)
        write_summary(per_fold_df, out_dir)

        print(f"[Fold {fold_idx + 1}] pred_vs_gt saved: {fold_dir / 'pred_vs_gt.csv'}")
        print(f"[Fold {fold_idx + 1}] best checkpoint saved: {ckpt_path}")

    print(f"\nDone. Results saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net segmentation baseline + contour PSD measurement.")
    parser.add_argument("--data_dir", type=str, default="./PSD_dataset/images")
    parser.add_argument("--image_dir", type=str, default=None, help="Deprecated alias for --data_dir.")
    parser.add_argument("--mask_dir", type=str, default="./PSD_dataset/masks")
    parser.add_argument("--ann_file", type=str, default="./PSD_dataset/annotations.csv")
    parser.add_argument("--output_dir", type=str, default="./baseline_results")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--min_area",
        type=float,
        default=1.0,
        help="Minimum connected-component area in pixels for PSD extraction from masks.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
