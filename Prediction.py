import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from Swin_conv import RockModel, get_transforms


def print_fuser_weights(model):

    if not hasattr(model, "fusers"):
        print("The model does not contain fusion (fusers) modules.")
        return

    print("\n Stage modes and weight statistics:")
    for i, fuser in enumerate(model.fusers):
        mode = model.stage_modes[i] if hasattr(model, "stage_modes") else "unknown"
        print(f"Stage {i}: mode = {mode}")
        if fuser is None:
            print("  No fusion module (None)")
        else:
            for name, param in fuser.named_parameters():
                print(f"  {name}, mean: {param.data.mean().item():.6f}")


def predict_images_dual(model_ckpt, img_dir, output_csv, stage_modes=None, device="cuda"):
    # ---- Load checkpoint ----
    print(f"🔹 Loading checkpoint: {model_ckpt}")
    ckpt = torch.load(model_ckpt, map_location=device)
    state_dict = ckpt["state_dict"]

    # ---- Initialize model ----
    model = RockModel(branch_type="conv_swin", stage_modes=stage_modes)
    model = model.to(device)

    # ---- Load model weights ----
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("⚠️ State_dict 不完全匹配")
        if missing:
            print("  Missing keys:", missing)
        if unexpected:
            print("  Unexpected keys:", unexpected)

    model.eval()

    # ---- Print fusion module weights ----
    print_fuser_weights(model)

    # ---- Image preprocessing / transformations ----
    transform = get_transforms()
    results = []
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(("png", "jpg", "jpeg"))]
    print(f"🔸 Found {len(img_list)} images in {img_dir}")

    with torch.no_grad():
        for img_name in tqdm(img_list, desc="Predicting Conv+Swin"):
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert("RGB")

            # ---- Transform 输入 ----
            img_t, _ = transform(img, None)
            img_t = img_t.unsqueeze(0).to(device)

            # ---- 前向推理 ----
            psd_pred, rosin_pred = model(img_t)
            psd_pred = psd_pred.detach().cpu().numpy().reshape(-1)
            rosin_pred = rosin_pred.detach().cpu().numpy().reshape(-1)

            results.append({
                "image_name": img_name,
                "D20": psd_pred[0],
                "D40": psd_pred[1],
                "D60": psd_pred[2],
                "D80": psd_pred[3],
                "max": psd_pred[4],
                "lambda": rosin_pred[0],
                "k": rosin_pred[1]
            })

    # ---- Save results to CSV ----
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n Prediction results have been saved to: {output_csv}")


# ================== Main entry point ==================
if __name__ == "__main__":
    model_ckpt = "./rock-train_conv_swin_skip_concat_gated_gated-fold5-epochepoch=78.ckpt"
    img_dir = "./test/patches_enhanced_images"
    output_csv = "./Predictions_test_dual.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predict_images_dual(
        model_ckpt=model_ckpt,
        img_dir=img_dir,
        output_csv=output_csv,
        stage_modes=["skip", "concat", "gated", "gated"],  # customizable fusion strategies
        device=device
    )
