import cv2
import os

# ==== Image enhancement pipeline ====
def enhance_image_pipeline(image):
    # 1. CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. Unsharp Mask
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # 3. Bilateral Filter
    final = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)

    return final

# ==== Batch processing function ====
def batch_enhance_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    total = 0

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(supported_exts):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        image = cv2.imread(in_path)
        if image is None:
            print(f"Failed to load image: {in_path}")
            continue

        enhanced = enhance_image_pipeline(image)
        cv2.imwrite(out_path, enhanced)
        print(f"Processed and saved: {out_path}")
        total += 1

    print(f"\n Total number of processed images:  {total}")

# ==== Set paths and run ====
if __name__ == "__main__":
    input_folder = "./images"
    output_folder =  "./patches_enhanced_images"

    batch_enhance_images(input_folder, output_folder)
