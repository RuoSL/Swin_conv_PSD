# Swin_conv_PSD

Deep learning pipeline for **rock-avalanche particle size distribution (PSD)** estimation from high-resolution imagery.  
This repository contains model code (Swin/Conv variants), training scripts, and inference utilities.


## Repository structure

Typical layout (may vary slightly depending on your local setup):
```text
Swin_conv_PSD/
├─ data_preprocess/                
│  ├─ images/                          
│  ├─ images_sam/         
│  ├─ patches_enhanced_images/
│  ├─ CLAHE.py 
│  ├─ SAM.py 
│  └─ sam_vit_b_01ec64.pth          
├─ PSD_dataset/                     
│  ├─ patches_enhanced_images/
│  └─ annotations_scale.csv     
├─ checkpoints/                     
├─ predictions/                     
├─ train.py                         
├─ predict.py                    
├─ requirements.txt                 
└─ README.md
```
## Data location

By default, the code assumes the dataset is located at:

- `./PSD_dataset/`

The image patches are expected under (example):

- `./PSD_dataset/patches_enhanced_images/`

If your scripts use arguments, you can point to the dataset explicitly:

```bash
python train.py --data_dir ./PSD_dataset
python predict.py --data_dir ./PSD_dataset
```

## Checkpoints
Training checkpoints are saved under:
- `./checkpoints/`
You may see experiment subfolders such as:
- `checkpoints/conv_only_single/`
- `checkpoints/swin_only_single/`
- `checkpoints/conv_swin_skip_concat_gated_gated/`
  
A `.ckpt file` is a serialized training snapshot and can be used for inference:
```bash
python predict.py --ckpt_path ./checkpoints/<experiment>/<fold>/model.ckpt
```
## Installation
Create an environment (recommended)

Using conda:
```bash
conda create -n swin_conv_psd python=3.10 -y
conda activate swin_conv_psd
```
## Install dependencies
```bash
pip install -r requirements.txt
```
## Run: training
```bash
python train.py \
  --data_dir ./PSD_dataset \
  --output_dir ./checkpoints \
  --model conv_swin \
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-4 \
  --fold 1
```

## Run: prediction / inference
Run inference using a trained checkpoint:
```bash
python predict.py \
  --data_dir ./PSD_dataset \
  --ckpt_path ./checkpoints/<experiment>/<fold>/model.ckpt \
  --out_dir ./predictions
```










