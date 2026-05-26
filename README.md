# Swin Conv PSD

Code for rock particle size distribution (PSD) estimation from image patches.

This repository contains direct PSD regression models, segmentation-based PSD baselines, and handcrafted-feature regressors. The current experiments use leave-one-area-out validation over three source areas:

- `1_patch`
- `3_patch`
- `o2_1_1_tile`

## Dataset Layout

The scripts expect the dataset under `./PSD_dataset/` by default:

```text
PSD_dataset/
|-- patches_enhanced_images_new/
|   |-- 1_patch_...
|   |-- 3_patch_...
|   `-- o2_1_1_tile_...
|-- masks/
|   `-- *.png
`-- annotations_new.csv
```

`annotations_new.csv` must include:

```text
image_name, mask_name, D20, D40, D60, D80, max
```

Additional metadata columns are allowed.

## Installation

```bash
conda create -n rock_bw python=3.10 -y
conda activate rock_bw
pip install -r requirements.txt
```

If using GPU, install a PyTorch build compatible with the CUDA version on your machine.

## Direct PSD Regression

Main script:

```text
Swin_conv.py
```

Supported branches:

- `conv_only`: ConvNeXt single branch
- `swin_only`: Swin Transformer single branch
- `conv_swin`: ConvNeXt + Swin dual branch

The script trains with leave-one-area-out validation and writes results to:

```text
loss_curves/final_runs/
checkpoints/final_runs/
```

### Conv Single Branch

```bash
CUDA_VISIBLE_DEVICES=0 python Swin_conv.py \
  --branches conv_only \
  --epochs 100 \
  --batch_size 16 \
  --splits 3 \
  --num_workers 4 \
  --target_scale 10 \
  --data_dir ./PSD_dataset/patches_enhanced_images_new \
  --ann_file ./PSD_dataset/annotations_new.csv
```

### Swin Single Branch

```bash
CUDA_VISIBLE_DEVICES=1 python Swin_conv.py \
  --branches swin_only \
  --epochs 100 \
  --batch_size 16 \
  --splits 3 \
  --num_workers 4 \
  --target_scale 10 \
  --data_dir ./PSD_dataset/patches_enhanced_images_new \
  --ann_file ./PSD_dataset/annotations_new.csv
```

### Conv + Swin Dual Branch

```bash
CUDA_VISIBLE_DEVICES=2 python Swin_conv.py \
  --branches conv_swin \
  --f1 skip \
  --f2 add \
  --f3 gated \
  --f4 concat \
  --epochs 100 \
  --batch_size 16 \
  --splits 3 \
  --num_workers 4 \
  --target_scale 10 \
  --data_dir ./PSD_dataset/patches_enhanced_images_new \
  --ann_file ./PSD_dataset/annotations_new.csv
```

`--target_scale 10` divides PSD targets by 10 during training and restores predictions before reporting/saving metrics.

## ResNet Baselines

### ResNet50

```bash
CUDA_VISIBLE_DEVICES=0 python ResNet50.py \
  --epochs 100 \
  --batch_size 64 \
  --splits 3 \
  --num_workers 4 \
  --data_dir ./PSD_dataset/patches_enhanced_images_new \
  --ann_file ./PSD_dataset/annotations_new.csv
```

### ResNet101

```bash
CUDA_VISIBLE_DEVICES=1 python ResNet101.py \
  --epochs 100 \
  --batch_size 32 \
  --splits 3 \
  --num_workers 4 \
  --data_dir ./PSD_dataset/patches_enhanced_images_new \
  --ann_file ./PSD_dataset/annotations_new.csv
```

Outputs are saved under:

```text
loss_curves/resnet50_psd/
loss_curves/resnet101_psd/
checkpoints/resnet50_psd/
checkpoints/resnet101_psd/
```

## Segmentation-Based PSD Baselines

These models train a binary segmentation mask first. PSD values are then calculated from predicted connected components.

Scripts:

- `baseline_unet_segmentation_psd.py`
- `baseline_unetpp_segmentation_psd.py`
- `baseline_deeplabv3plus_segmentation_psd.py`

### U-Net

```bash
CUDA_VISIBLE_DEVICES=0 python baseline_unet_segmentation_psd.py \
  --epochs 100 \
  --batch_size 8 \
  --splits 3 \
  --image_dir ./PSD_dataset/patches_enhanced_images_new \
  --mask_dir ./PSD_dataset/masks \
  --ann_file ./PSD_dataset/annotations_new.csv \
  --num_workers 4
```

### U-Net++

```bash
CUDA_VISIBLE_DEVICES=1 python baseline_unetpp_segmentation_psd.py \
  --epochs 100 \
  --batch_size 8 \
  --splits 3 \
  --image_dir ./PSD_dataset/patches_enhanced_images_new \
  --mask_dir ./PSD_dataset/masks \
  --ann_file ./PSD_dataset/annotations_new.csv \
  --num_workers 4
```

### DeepLabV3+

```bash
CUDA_VISIBLE_DEVICES=2 python baseline_deeplabv3plus_segmentation_psd.py \
  --epochs 100 \
  --batch_size 8 \
  --splits 3 \
  --image_dir ./PSD_dataset/patches_enhanced_images_new \
  --mask_dir ./PSD_dataset/masks \
  --ann_file ./PSD_dataset/annotations_new.csv \
  --num_workers 4
```

Outputs are saved under:

```text
baseline_results/
```

## Handcrafted Feature Baselines

These scripts extract GLCM, FFT, and basic image statistics, then train scikit-learn regressors.

```bash
python "GLCM _FFT_RF_leave_one_area_out.py"
```

```bash
python "GLCM _FFT_ExtraTrees_leave_one_area_out.py"
```

They use CPU, not GPU. Results are saved under:

```text
rf_results/
```

## Metrics

All updated scripts report comparable PSD metrics:

```text
test_psd_mae
test_psd_rmse
test_psd_mape
test_psd_r2
test_psd_D20_mae
test_psd_D40_mae
test_psd_D60_mae
test_psd_D80_mae
test_psd_max_mae
test_rank_violation_rate
```

Each experiment writes:

- `per_fold_metrics.csv`
- `per_fold_metrics.xlsx`
- `metrics_summary.csv`
- `metrics_summary.xlsx`
- `fold*_pred_vs_gt.csv`

## Notes

- Direct regression models are sensitive to PSD target scale. Use `--target_scale 10` for `Swin_conv.py` experiments when training on large PSD values.
- Segmentation-based models are not trained on PSD targets directly; changing PSD scale in the CSV affects only final PSD metric comparison, not mask training.
- Handcrafted tree-based regressors are usually insensitive to linear target scaling after predictions are converted back.
