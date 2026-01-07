import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# === SAM configuration ===
def get_mask_generator(model):
    return SamAutomaticMaskGenerator(
        model=model,
        points_per_side=128,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.80,
        crop_n_layers=0,
        min_mask_region_area=0
    )

# === Generate a single-channel edge map similar to Canny output ===
def generate_sam_canny_like(image_input, sam_model):
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    mask_generator = get_mask_generator(sam_model)
    masks = mask_generator.generate(image_rgb)

    edge_img = np.zeros(image_input.shape[:2], dtype=np.uint8)

    for mask_info in masks:
        mask = mask_info['segmentation'].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edge_img, contours, -1, 255, 1)

    return edge_img

def generate_sam_colormap(image_input, sam_model):
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    mask_generator = get_mask_generator(sam_model)
    masks = mask_generator.generate(image_rgb)

    color_img = np.zeros_like(image_input)

    for mask_info in masks:
        mask = mask_info['segmentation'].astype(np.uint8)
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        color_img[mask == 1] = color

    return color_img

# === Load SAM model ===
sam_checkpoint = "./sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

# === Input and output directories ===
input_folder = "./images"
output_folder = "./images_sam"
os.makedirs(output_folder, exist_ok=True)

# === Batch processing ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        print(f"Processing: {filename}")

        edge_canny_like = generate_sam_canny_like(image, sam)
        color_map = generate_sam_colormap(image, sam)

        name, ext = os.path.splitext(filename)
        out_edge = os.path.join(output_folder, f"{name}_edge{ext}")
        out_color = os.path.join(output_folder, f"{name}_color{ext}")

        cv2.imwrite(out_edge, edge_canny_like)
        cv2.imwrite(out_color, color_map)

print(" SAM mask edges + color maps saved.")
