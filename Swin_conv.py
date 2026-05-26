import os
import random
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from timm.models import create_model

# =========================================================
# Global config
# =========================================================
FUSION_MODES = ["skip", "add", "concat", "gated"]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
RUN_ID_ROOT = datetime.now().strftime("%Y%m%d-%H%M%S")


def get_run_id(branch_type: str, fusion_mode: str) -> str:
    tag = f"{branch_type}_{fusion_mode}".replace("+", "_")
    rid = os.environ.get(f"RUN_ID_{tag}")
    if not rid:
        rid = f"{RUN_ID_ROOT}-{tag}"
        os.environ[f"RUN_ID_{tag}"] = rid
    return rid


# =========================================================
# Seeds & Logging
# =========================================================
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


# =========================================================
# Transforms
# =========================================================
class PairedAugment:
    def __init__(self, size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = mean
        self.std = std

        self.pil_aug = T.Compose([
            T.Resize(size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ])

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, img, aux_map):
        img = self.pil_aug(img)
        img = self.to_tensor(img)

        aux_t = None
        if aux_map is not None:
            if isinstance(aux_map, np.ndarray):
                aux_t = torch.from_numpy(aux_map).float()
            elif isinstance(aux_map, torch.Tensor):
                aux_t = aux_map.float()
            if aux_t.ndim == 3 and aux_t.shape[-1] <= 4:
                aux_t = aux_t.permute(2, 0, 1)
            aux_t = aux_t.unsqueeze(0)
            aux_t = F.interpolate(aux_t, size=self.size, mode="nearest")
            aux_t = aux_t.squeeze(0)

        return img, aux_t


class PairedEvalTransform:
    def __init__(self, size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = mean
        self.std = std

        self.resize = T.Resize(size)
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, img, aux_map):
        img = self.resize(img)
        img = self.to_tensor(img)

        aux_t = None
        if aux_map is not None:
            if isinstance(aux_map, np.ndarray):
                aux_t = torch.from_numpy(aux_map).float()
            elif isinstance(aux_map, torch.Tensor):
                aux_t = aux_map.float()
            if aux_t.ndim == 3 and aux_t.shape[-1] <= 4:
                aux_t = aux_t.permute(2, 0, 1)
            aux_t = aux_t.unsqueeze(0)
            aux_t = F.interpolate(aux_t, size=self.size, mode="nearest")
            aux_t = aux_t.squeeze(0)

        return img, aux_t


def get_train_transforms():
    return PairedAugment(size=(224, 224))


def get_eval_transforms():
    return PairedEvalTransform(size=(224, 224))


# =========================================================
# Helpers
# =========================================================
def get_area_from_name(filename: str) -> str:
    name = os.path.splitext(filename)[0].lower()

    if name.startswith("o2_1_1_tile"):
        return "o2_1_1_tile"
    if name.startswith("1_patch"):
        return "1_patch"
    if name.startswith("3_patch"):
        return "3_patch"

    raise ValueError(f"Cannot infer area from filename: {filename}")


# =========================================================
# Dataset
# =========================================================
class RockDataset(Dataset):
    def __init__(self, root, transform=None, annotation_file=None, aux_dir=None, target_scale=1.0):
        self.root = root
        self.aux_dir = aux_dir
        self.target_scale = target_scale
        self.transform = transform or (lambda img, aux: (T.ToTensor()(img), aux))

        self.image_paths = sorted(
            [os.path.join(root, f) for f in os.listdir(root)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

        self.annotations = {}
        if annotation_file:
            df = pd.read_csv(annotation_file)
            print("CSV columns:", df.columns.tolist())
            for _, row in df.iterrows():
                self.annotations[row["image_name"]] = {
                    "psd": [row["D20"], row["D40"], row["D60"], row["D80"], row["max"]],
                }

        self.image_paths = [p for p in self.image_paths if os.path.basename(p) in self.annotations]
        self.areas = [get_area_from_name(os.path.basename(p)) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def _load_aux_from_disk(self, basename):
        npy_path = os.path.join(self.aux_dir, f"{basename}_aux.npy")
        if not os.path.exists(npy_path):
            logger.warning(f"Missing AUX file: {npy_path}, returning None")
            return None
        aux = np.load(npy_path).astype(np.float32)
        return aux

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        basename = os.path.splitext(img_name)[0]

        img = Image.open(img_path).convert("RGB")

        aux = None
        if self.aux_dir is not None:
            aux = self._load_aux_from_disk(basename)

        if self.transform:
            img, aux = self.transform(img, aux)

        ann = self.annotations[img_name]
        return {
            "image": img,
            "aux_map": aux,
            "psd": torch.tensor(ann["psd"], dtype=torch.float32) / self.target_scale,
            "img_name": img_name,
        }


# =========================================================
# Collate
# =========================================================
def rock_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise RuntimeError("Empty batch after filtering None samples")

    out = {}
    out["image"] = torch.stack([b["image"] for b in batch], dim=0)
    out["psd"] = torch.stack([b["psd"] for b in batch], dim=0)
    out["img_name"] = [b.get("img_name", "") for b in batch]

    aux_list = [b.get("aux_map", None) for b in batch]
    if all(a is not None for a in aux_list):
        out["aux_map"] = torch.stack(aux_list, dim=0)

    return out


# =========================================================
# DataModule
# =========================================================
class RockDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, ann_file, batch_size=16, n_splits=3, aux_dir=None, num_workers=4, target_scale=1.0):
        super().__init__()
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.aux_dir = aux_dir
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.num_workers = num_workers
        self.target_scale = target_scale

        self.train_transform = get_train_transforms()
        self.eval_transform = get_eval_transforms()

        self.fold_idx = 0
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.area_order = ["1_patch", "3_patch", "o2_1_1_tile"]

    def set_fold(self, fold_idx: int):
        self.fold_idx = fold_idx

    def setup(self, stage=None):
        assert self.n_splits == 3, "For leave-one-area-out, n_splits must be 3"

        full_dataset = RockDataset(
            self.data_dir,
            transform=None,
            annotation_file=self.ann_file,
            aux_dir=self.aux_dir,
            target_scale=self.target_scale,
        )

        total_size = len(full_dataset)
        print(f"Total dataset size: {total_size}")

        all_indices = np.arange(total_size)
        all_areas = np.array(full_dataset.areas)

        test_area = self.area_order[self.fold_idx]
        train_areas = [a for a in self.area_order if a != test_area]

        train_val_indices = all_indices[np.isin(all_areas, train_areas)]
        test_indices = all_indices[all_areas == test_area]

        rng = np.random.default_rng(42)
        shuffled = train_val_indices.copy()
        rng.shuffle(shuffled)

        val_ratio = 0.15
        val_size = int(len(shuffled) * val_ratio)

        val_indices = shuffled[:val_size]
        train_indices = shuffled[val_size:]

        train_base = RockDataset(
            self.data_dir,
            transform=self.train_transform,
            annotation_file=self.ann_file,
            aux_dir=self.aux_dir,
            target_scale=self.target_scale,
        )
        val_base = RockDataset(
            self.data_dir,
            transform=self.eval_transform,
            annotation_file=self.ann_file,
            aux_dir=self.aux_dir,
            target_scale=self.target_scale,
        )
        test_base = RockDataset(
            self.data_dir,
            transform=self.eval_transform,
            annotation_file=self.ann_file,
            aux_dir=self.aux_dir,
            target_scale=self.target_scale,
        )

        self.train_dataset = torch.utils.data.Subset(train_base, train_indices.tolist())
        self.val_dataset = torch.utils.data.Subset(val_base, val_indices.tolist())
        self.test_dataset = torch.utils.data.Subset(test_base, test_indices.tolist())

        print(f"\n===== Fold {self.fold_idx + 1}/{self.n_splits} =====")
        print(f"Train areas: {train_areas}")
        print(f"Test area: {test_area}")
        print(f"Train size: {len(train_indices)}")
        print(f"Val size: {len(val_indices)}")
        print(f"Test size: {len(test_indices)}")

        preview_names = [os.path.basename(full_dataset.image_paths[i]) for i in test_indices[:10]]
        print(f"First 10 test image names for fold {self.fold_idx + 1}:")
        for name in preview_names:
            print("   ", name)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=rock_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            collate_fn=rock_collate_fn,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=rock_collate_fn,
        )


# =========================================================
# Model
# =========================================================
class RockModel(pl.LightningModule):
    def __init__(
        self,
        branch_type="conv_only",
        stage_fusions=None,
        lr=1e-4,
        weight_decay=1e-5,
        freeze_backbone=True,
        unfreeze_epoch=10,
        target_scale=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.branch_type = branch_type.lower()
        self.target_scale = target_scale
        if stage_fusions is None:
            if self.branch_type == "conv_swin":
                stage_fusions = ["skip", "add", "gated", "concat"]
            else:
                stage_fusions = ["concat", "concat", "concat", "concat"]

        self.stage_fusions = [m.lower() for m in stage_fusions]
        assert len(self.stage_fusions) == 4, "stage_fusions must have 4 items"
        for m in self.stage_fusions:
            assert m in FUSION_MODES, f"Each stage fusion must be one of {FUSION_MODES}"

        self.w_psd = 1.0
        self.w_rank = 0.1

        self.train_losses = []
        self.val_psd_preds, self.val_psd_gts = [], []
        self.test_psd_preds, self.test_psd_gts = [], []

        self.target_C = [128, 192, 256, 384]
        self.backbones_unfrozen = False

        if self.branch_type == "conv_only":
            self.backbone = create_model(
                "convnext_tiny", pretrained=True,
                features_only=True, out_indices=(0, 1, 2, 3)
            )
            chs = self.backbone.feature_info.channels()
            self.in_chs = chs
            self.proj = nn.ModuleList([
                nn.Conv2d(chs[i], self.target_C[i], 1, bias=False)
                for i in range(4)
            ])
            self.is_dual = False

        elif self.branch_type == "swin_only":
            self.backbone = create_model(
                "swin_tiny_patch4_window7_224", pretrained=True,
                features_only=True, out_indices=(0, 1, 2, 3)
            )
            chs = self.backbone.feature_info.channels()
            self.in_chs = chs
            self.proj = nn.ModuleList([
                nn.Conv2d(chs[i], self.target_C[i], 1, bias=False)
                for i in range(4)
            ])
            self.is_dual = False

        elif self.branch_type == "conv_swin":
            self.convnext = create_model(
                "convnext_tiny", pretrained=True,
                features_only=True, out_indices=(0, 1, 2, 3)
            )
            self.swin = create_model(
                "swin_tiny_patch4_window7_224", pretrained=True,
                features_only=True, out_indices=(0, 1, 2, 3)
            )

            cnx_chs = self.convnext.feature_info.channels()
            swin_chs = self.swin.feature_info.channels()

            self.cnx_in_chs = cnx_chs
            self.swin_in_chs = swin_chs

            self.cnx_proj = nn.ModuleList([
                nn.Conv2d(cnx_chs[i], self.target_C[i], 1, bias=False)
                for i in range(4)
            ])
            self.swin_proj = nn.ModuleList([
                nn.Conv2d(swin_chs[i], self.target_C[i], 1, bias=False)
                for i in range(4)
            ])

            self.stage_gates = nn.ModuleList()
            for i in range(4):
                if self.stage_fusions[i] == "gated":
                    self.stage_gates.append(
                        nn.Sequential(
                            nn.Linear(self.target_C[i] * 2, self.target_C[i]),
                            nn.Sigmoid()
                        )
                    )
                else:
                    self.stage_gates.append(nn.Identity())

            self.is_dual = True
        else:
            raise ValueError(f"Unknown branch_type: {self.branch_type}")

        if self.branch_type == "conv_swin":
            in_dim = 0
            for i, mode in enumerate(self.stage_fusions):
                if mode == "concat":
                    in_dim += 2 * self.target_C[i]
                else:
                    in_dim += self.target_C[i]
        else:
            in_dim = sum(self.target_C)

        self.fusion_proj = nn.Linear(in_dim, 768)

        # 轻量 fusion head
        self.fusion = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.psd_head = nn.Linear(192, 5)

        if not self.is_dual:
            self.target_layer = self.proj[-1]
        else:
            self.target_layer = self.cnx_proj[-1]

        if self.branch_type == "conv_swin" and freeze_backbone:
            self.freeze_backbones()

    def freeze_backbones(self):
        if self.branch_type != "conv_swin":
            return

        for p in self.convnext.parameters():
            p.requires_grad = False
        for p in self.swin.parameters():
            p.requires_grad = False

        for p in self.cnx_proj.parameters():
            p.requires_grad = True
        for p in self.swin_proj.parameters():
            p.requires_grad = True
        for p in self.fusion_proj.parameters():
            p.requires_grad = True
        for p in self.fusion.parameters():
            p.requires_grad = True
        for p in self.psd_head.parameters():
            p.requires_grad = True

        if hasattr(self, "stage_gates"):
            for g in self.stage_gates:
                for p in g.parameters():
                    p.requires_grad = True

        self.backbones_unfrozen = False

    def unfreeze_backbones(self):
        if self.branch_type != "conv_swin":
            return

        for p in self.convnext.parameters():
            p.requires_grad = True
        for p in self.swin.parameters():
            p.requires_grad = True

        self.backbones_unfrozen = True

    def on_train_epoch_start(self):
        if self.branch_type == "conv_swin":
            if self.current_epoch == 0 and not self.backbones_unfrozen:
                print("Warm-up: dual backbones frozen.")
            if self.current_epoch == self.hparams.unfreeze_epoch and not self.backbones_unfrozen:
                print(f"Unfreezing dual backbones at epoch {self.current_epoch}.")
                self.unfreeze_backbones()

    def _ensure_nchw(self, x, expected_c=None):
        if x is None:
            return None
        if x.ndim == 4:
            c_nchw = x.shape[1]
            c_nhwc = x.shape[-1]
            if expected_c is not None and c_nchw != expected_c and c_nhwc == expected_c:
                x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, image, aux_map=None):
        if not self.is_dual:
            feats = self.backbone(image)
            pyramid = [
                self.proj[i](self._ensure_nchw(feats[i], self.in_chs[i]))
                for i in range(4)
            ]
            vecs = [F.adaptive_avg_pool2d(p, 1).flatten(1) for p in pyramid]
            fused_vec = torch.cat(vecs, dim=1)

        else:
            cnx_feats = self.convnext(image)
            cnx_pyramid = [
                self.cnx_proj[i](self._ensure_nchw(cnx_feats[i], self.cnx_in_chs[i]))
                for i in range(4)
            ]

            swin_feats = self.swin(image)
            swin_pyramid = [
                self.swin_proj[i](self._ensure_nchw(swin_feats[i], self.swin_in_chs[i]))
                for i in range(4)
            ]

            fused_stage_vecs = []

            for i in range(4):
                cnx_vec_i = F.adaptive_avg_pool2d(cnx_pyramid[i], 1).flatten(1)
                swin_vec_i = F.adaptive_avg_pool2d(swin_pyramid[i], 1).flatten(1)

                mode = self.stage_fusions[i]

                if mode == "skip":
                    fused_i = cnx_vec_i
                elif mode == "add":
                    fused_i = (cnx_vec_i + swin_vec_i) / 2
                elif mode == "concat":
                    fused_i = torch.cat([cnx_vec_i, swin_vec_i], dim=1)
                elif mode == "gated":
                    gate_input = torch.cat([cnx_vec_i, swin_vec_i], dim=1)
                    gate = self.stage_gates[i](gate_input)
                    fused_i = gate * cnx_vec_i + (1 - gate) * swin_vec_i
                else:
                    raise ValueError(f"Unsupported stage fusion mode: {mode}")

                fused_stage_vecs.append(fused_i)

            fused_vec = torch.cat(fused_stage_vecs, dim=1)

        fused = self.fusion_proj(fused_vec)
        feat = self.fusion(fused)
        psd = self.psd_head(feat)
        return psd

    def rank_loss(self, psd_pred):
        loss = 0
        for i in range(4):
            loss += F.relu(psd_pred[:, i] - psd_pred[:, i + 1])
        return loss.mean()

    def training_step(self, batch, batch_idx):
        img, psd_gt = batch["image"], batch["psd"]
        aux_map = batch.get("aux_map", None)

        psd_pred = self(img, aux_map=aux_map)
        l_psd = F.smooth_l1_loss(psd_pred, psd_gt)
        l_rank = self.rank_loss(psd_pred)
        loss = self.w_psd * l_psd + self.w_rank * l_rank

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss at Epoch {self.current_epoch}, Batch {batch_idx}")
            loss = torch.tensor(0.0, device=loss.device)

        self.train_losses.append(loss)
        self.log("train_loss_batch", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img, psd_gt = batch["image"], batch["psd"]
        aux_map = batch.get("aux_map", None)

        psd_pred = self(img, aux_map=aux_map)
        l_psd = F.smooth_l1_loss(psd_pred, psd_gt)
        l_rank = self.rank_loss(psd_pred)
        loss = self.w_psd * l_psd + self.w_rank * l_rank

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.val_psd_preds.append(psd_pred.detach().cpu())
        self.val_psd_gts.append(psd_gt.detach().cpu())

    def test_step(self, batch, batch_idx):
        img, psd_gt = batch["image"], batch["psd"]
        aux_map = batch.get("aux_map", None)

        psd_pred = self(img, aux_map=aux_map)
        l_psd = F.smooth_l1_loss(psd_pred, psd_gt)
        l_rank = self.rank_loss(psd_pred)
        loss = self.w_psd * l_psd + self.w_rank * l_rank

        self.log("test_loss", loss, prog_bar=True, logger=True)

        self.test_psd_preds.append(psd_pred.detach().cpu())
        self.test_psd_gts.append(psd_gt.detach().cpu())

    def on_validation_epoch_end(self):
        psd_preds = torch.cat(self.val_psd_preds, dim=0) * self.target_scale
        psd_gts = torch.cat(self.val_psd_gts, dim=0) * self.target_scale

        psd_mae = torch.mean(torch.abs(psd_preds - psd_gts))
        psd_rmse = torch.sqrt(F.mse_loss(psd_preds, psd_gts))
        psd_mape = torch.mean(torch.abs((psd_gts - psd_preds) / (psd_gts + 1e-8))) * 100
        psd_mean = torch.mean(psd_gts, dim=0, keepdim=True)
        ss_tot_psd = torch.sum((psd_gts - psd_mean) ** 2, dim=0)
        ss_res_psd = torch.sum((psd_gts - psd_preds) ** 2, dim=0)
        psd_r2_sep = 1 - ss_res_psd / (ss_tot_psd + 1e-8)
        psd_r2 = torch.mean(psd_r2_sep)

        violations = 0
        for i in range(psd_preds.shape[0]):
            pred = psd_preds[i]
            for j in range(4):
                if pred[j] >= pred[j + 1]:
                    violations += 1
                    break
        rank_violation_rate = violations / psd_preds.shape[0]

        print(f"Validation Epoch {self.current_epoch} Metrics:")
        print(f"PSD MAE: {psd_mae.item():.4f}")
        print(f"PSD RMSE: {psd_rmse.item():.4f}")
        print(f"PSD MAPE: {psd_mape.item():.2f}%")
        print(f"PSD R²: {psd_r2.item():.4f}")
        print(f"Rank Violation Rate: {rank_violation_rate:.4f}")

        psd_labels = ["D20", "D40", "D60", "D80", "max"]
        for i, label in enumerate(psd_labels):
            self.log(f"val_psd_{label}_r2", psd_r2_sep[i], logger=True)

        self.log("val_psd_mae", psd_mae, logger=True, prog_bar=True)
        self.log("val_psd_rmse", psd_rmse, logger=True)
        self.log("val_psd_mape", psd_mape, logger=True)
        self.log("val_psd_r2", psd_r2, logger=True)
        self.log("val_rank_violation", rank_violation_rate, logger=True)

        self.val_psd_preds.clear()
        self.val_psd_gts.clear()

    def on_test_epoch_end(self):
        psd_preds = torch.cat(self.test_psd_preds, dim=0) * self.target_scale
        psd_gts = torch.cat(self.test_psd_gts, dim=0) * self.target_scale

        psd_mae = torch.mean(torch.abs(psd_preds - psd_gts))
        psd_rmse = torch.sqrt(F.mse_loss(psd_preds, psd_gts))
        psd_mape = torch.mean(torch.abs((psd_gts - psd_preds) / (psd_gts + 1e-8))) * 100
        psd_mean = torch.mean(psd_gts, dim=0, keepdim=True)
        ss_tot_psd = torch.sum((psd_gts - psd_mean) ** 2, dim=0)
        ss_res_psd = torch.sum((psd_gts - psd_preds) ** 2, dim=0)
        psd_r2_sep = 1 - ss_res_psd / (ss_tot_psd + 1e-8)
        psd_r2 = 1 - torch.sum(ss_res_psd) / (torch.sum(ss_tot_psd) + 1e-8)

        violations = sum(
            any(psd_preds[i, j] >= psd_preds[i, j + 1] for j in range(4))
            for i in range(psd_preds.shape[0])
        )
        rank_violation_rate = violations / psd_preds.shape[0]

        psd_labels = ["D20", "D40", "D60", "D80", "max"]
        psd_mae_sep = torch.mean(torch.abs(psd_preds - psd_gts), dim=0)
        psd_rmse_sep = torch.sqrt(torch.mean((psd_preds - psd_gts) ** 2, dim=0))
        psd_mape_sep = torch.mean(torch.abs((psd_gts - psd_preds) / (psd_gts + 1e-8)), dim=0) * 100

        self.log("test_psd_mae", psd_mae, logger=True)
        self.log("test_psd_rmse", psd_rmse, logger=True)
        self.log("test_psd_mape", psd_mape, logger=True)
        self.log("test_psd_r2", psd_r2, logger=True)
        self.log("test_rank_violation_rate", rank_violation_rate, logger=True)

        for i, label in enumerate(psd_labels):
            self.log(f"test_psd_{label}_mae", psd_mae_sep[i], logger=True)
            self.log(f"test_psd_{label}_rmse", psd_rmse_sep[i], logger=True)
            self.log(f"test_psd_{label}_mape", psd_mape_sep[i], logger=True)
            self.log(f"test_psd_{label}_r2", psd_r2_sep[i], logger=True)

        self.test_psd_preds.clear()
        self.test_psd_gts.clear()

    def configure_optimizers(self):
        if self.branch_type == "conv_swin":
            backbone_params = list(self.convnext.parameters()) + list(self.swin.parameters())
            head_params = (
                list(self.cnx_proj.parameters()) +
                list(self.swin_proj.parameters()) +
                list(self.fusion_proj.parameters()) +
                list(self.fusion.parameters()) +
                list(self.psd_head.parameters())
            )

            if hasattr(self, "stage_gates"):
                for g in self.stage_gates:
                    head_params += list(g.parameters())

            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": 1e-5},
                    {"params": head_params, "lr": 1e-4},
                ],
                weight_decay=self.hparams.weight_decay
            )
        else:
            backbone_params = list(self.backbone.parameters())
            head_params = (
                list(self.proj.parameters()) +
                list(self.fusion_proj.parameters()) +
                list(self.fusion.parameters()) +
                list(self.psd_head.parameters())
            )

            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": 1e-5},
                    {"params": head_params, "lr": 1e-4},
                ],
                weight_decay=self.hparams.weight_decay
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def visualize_fusion(self, image, aux_map=None, target_idx=0, save_dir="./vis_out", base_name="sample"):
        os.makedirs(save_dir, exist_ok=True)
        heatmap = self.get_gradcam(image, aux_map=aux_map, target_idx=target_idx)
        import matplotlib.pyplot as plt
        plt.imshow(heatmap, cmap="jet", alpha=0.8)
        out_path = os.path.join(save_dir, f"{base_name}_gradcam.png")
        plt.axis("off")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return {"type": "gradcam", "map": heatmap, "path": out_path}

    def activations_hook(self, module, input, output):
        self.activations = output
        output.register_hook(self.gradients_hook)

    def gradients_hook(self, grad):
        self.gradients = grad

    def get_gradcam(self, img, aux_map=None, target_idx=0):
        self.eval()
        device = next(self.parameters()).device
        img = img.unsqueeze(0).to(device)
        aux_map = aux_map.unsqueeze(0).to(device) if aux_map is not None else None
        self.zero_grad()
        self.activations, self.gradients = None, None
        hook_handle = self.target_layer.register_forward_hook(self.activations_hook)
        psd = self(img, aux_map=aux_map)
        if target_idx < 0 or target_idx >= psd.shape[1]:
            raise ValueError(f"target_idx {target_idx} out of range (psd.shape={psd.shape})")
        output = psd[0, target_idx]
        output.backward(retain_graph=True)
        hook_handle.remove()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM: Failed to capture activations or gradients")
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu().detach()
        heatmap = torch.maximum(heatmap, torch.tensor(0.0))
        heatmap /= (torch.max(heatmap) + 1e-8)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(img.shape[2], img.shape[3]),
            mode="bilinear",
            align_corners=False
        ).squeeze()
        return heatmap.numpy()

# =========================================================
# Loss Curve Callback
# =========================================================
class LossCurveCallback(pl.Callback):
    def __init__(self, train_metrics_every=0):
        super().__init__()
        self.history = []
        self._last_train = None
        self.train_metrics = {}
        self.train_metrics_every = train_metrics_every

    @staticmethod
    def compute_metrics(preds, gts):
        mae = torch.mean(torch.abs(preds - gts)).item()
        rmse = torch.sqrt(F.mse_loss(preds, gts)).item()
        mape = torch.mean(torch.abs((gts - preds) / (gts + 1e-8))).item() * 100
        mean = torch.mean(gts, dim=0, keepdim=True)
        ss_tot = torch.sum((gts - mean) ** 2, dim=0)
        ss_res = torch.sum((gts - preds) ** 2, dim=0)
        r2_sep = 1 - ss_res / (ss_tot + 1e-8)
        r2 = torch.mean(r2_sep).item()
        return mae, rmse, mape, r2

    def on_train_epoch_end(self, trainer, pl_module):
        if len(pl_module.train_losses) > 0:
            vals = [(l.detach().float().item() if torch.is_tensor(l) else float(l))
                    for l in pl_module.train_losses]
            train_loss_epoch = float(np.mean(vals))
        else:
            train_loss_epoch = float("nan")
        self._last_train = train_loss_epoch
        pl_module.train_losses.clear()

        if self.train_metrics_every <= 0 or (trainer.current_epoch + 1) % self.train_metrics_every != 0:
            self.train_metrics = {
                "train_loss": train_loss_epoch,
                "train_psd_mae": float("nan"),
                "train_psd_rmse": float("nan"),
                "train_psd_mape": float("nan"),
                "train_psd_r2": float("nan"),
                "train_rank_violation": float("nan"),
            }
            return

        pl_module.eval()
        psd_preds, psd_gts = [], []
        with torch.no_grad():
            for batch in trainer.datamodule.train_dataloader():
                img = batch["image"].to(pl_module.device)
                psd_gt = batch["psd"].to(pl_module.device)
                aux_map = batch.get("aux_map", None)
                if aux_map is not None:
                    aux_map = aux_map.to(pl_module.device)
                psd_pred = pl_module(img, aux_map=aux_map)
                psd_preds.append(psd_pred.cpu())
                psd_gts.append(psd_gt.cpu())

        target_scale = getattr(pl_module, "target_scale", 1.0)
        psd_preds = torch.cat(psd_preds) * target_scale
        psd_gts = torch.cat(psd_gts) * target_scale

        psd_mae, psd_rmse, psd_mape, psd_r2 = self.compute_metrics(psd_preds, psd_gts)

        violations = sum(any(pred[j] >= pred[j + 1] for j in range(4)) for pred in psd_preds)
        rank_violation_rate = violations / psd_preds.shape[0]

        self.train_metrics = {
            "train_loss": train_loss_epoch,
            "train_psd_mae": psd_mae,
            "train_psd_rmse": psd_rmse,
            "train_psd_mape": psd_mape,
            "train_psd_r2": psd_r2,
            "train_rank_violation": rank_violation_rate,
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        val_loss = trainer.callback_metrics.get("val_loss")
        val_loss = val_loss.item() if val_loss is not None else float("nan")

        record = {"epoch": int(trainer.current_epoch)}

        if self.train_metrics:
            record.update(self.train_metrics)
        else:
            record.update({
                "train_loss": float("nan"),
                "train_psd_mae": float("nan"),
                "train_psd_rmse": float("nan"),
                "train_psd_mape": float("nan"),
                "train_psd_r2": float("nan"),
                "train_rank_violation": float("nan"),
            })

        record["val_loss"] = val_loss
        record["val_psd_mae"] = float(trainer.callback_metrics.get("val_psd_mae", torch.tensor(float("nan"))))
        record["val_psd_rmse"] = float(trainer.callback_metrics.get("val_psd_rmse", torch.tensor(float("nan"))))
        record["val_psd_mape"] = float(trainer.callback_metrics.get("val_psd_mape", torch.tensor(float("nan"))))
        record["val_psd_r2"] = float(trainer.callback_metrics.get("val_psd_r2", torch.tensor(float("nan"))))
        record["val_rank_violation"] = float(trainer.callback_metrics.get("val_rank_violation", torch.tensor(float("nan"))))

        self.history.append(record)
        print(
            f"[Epoch {record['epoch']}] TrainLoss={record['train_loss']:.4f}, "
            f"ValLoss={record['val_loss']:.4f} | "
            f"Train PSD→ MAE {record['train_psd_mae']:.4f}, RMSE {record['train_psd_rmse']:.4f}, R² {record['train_psd_r2']:.4f} | "
            f"Val PSD→ MAE {record['val_psd_mae']:.4f}, RMSE {record['val_psd_rmse']:.4f}, R² {record['val_psd_r2']:.4f}"
        )


# =========================================================
# Run CV
# =========================================================
def run_cv(
    branch_type: str,
    stage_fusions: list,
    data_dir: str,
    ann_file: str,
    aux_dir: str,
    batch_size: int,
    n_splits: int,
    epochs: int,
    num_workers: int,
    train_metrics_every: int,
    target_scale: float,
    base_ckpt_dir: str,
    base_log_dir: str,
    base_curve_dir: str,
    run_id: str,
):
    fusion_tag = "".join([m[0] for m in stage_fusions])
    structure_tag = f"{branch_type}_{fusion_tag}"
    mode_name = structure_tag

    print(f"\n===== Running mode: {mode_name} (branch_type={branch_type}, stage_fusions={stage_fusions}) =====\n")
    last_best_ckpt = None

    curve_dir = os.path.join(base_curve_dir, structure_tag, mode_name, run_id)
    os.makedirs(curve_dir, exist_ok=True)

    all_fold_metrics = []
    all_pred_records = []

    for fold_idx in range(n_splits):
        print(f"\nTraining Fold {fold_idx + 1}/{n_splits}...")

        dm = RockDataModule(
            data_dir=data_dir,
            ann_file=ann_file,
            batch_size=batch_size,
            n_splits=n_splits,
            aux_dir=aux_dir,
            num_workers=num_workers,
            target_scale=target_scale,
        )
        dm.set_fold(fold_idx)
        dm.setup(stage="fit")

        model = RockModel(
            branch_type=branch_type,
            stage_fusions=stage_fusions,
            lr=1e-4,
            weight_decay=1e-4,
            freeze_backbone=True,
            unfreeze_epoch=10,
            target_scale=target_scale,
        )

        ckpt_dir = os.path.join(base_ckpt_dir, structure_tag, mode_name, run_id, f"fold{fold_idx + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)

        loss_cb = LossCurveCallback(train_metrics_every=train_metrics_every)
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"f{fold_idx + 1}-{{epoch:02d}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[loss_cb, ckpt_cb],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            check_val_every_n_epoch=1,
            val_check_interval=1.0,
            num_sanity_val_steps=0,
            benchmark=True,
            deterministic=False,
        )

        trainer.fit(model, datamodule=dm)

        loss_xlsx = os.path.join(curve_dir, f"fold{fold_idx + 1}_loss_curve.xlsx")
        pd.DataFrame(loss_cb.history).to_excel(loss_xlsx, index=False)
        print(f"[{mode_name}][fold {fold_idx + 1}] loss curve saved to: {loss_xlsx}")

        best_ckpt = ckpt_cb.best_model_path
        test_area = dm.area_order[fold_idx]
        train_areas = [a for a in dm.area_order if a != test_area]
        fold_record = {
            "fold": fold_idx + 1,
            "mode": mode_name,
            "test_area": test_area,
            "train_areas": "+".join(train_areas),
        }

        if best_ckpt:
            last_best_ckpt = best_ckpt

            dm.set_fold(fold_idx)
            dm.setup(stage="validate")
            val_out = trainer.validate(
                model=None,
                dataloaders=dm.val_dataloader(),
                ckpt_path=best_ckpt,
                verbose=False
            )[0]

            print(f"[{mode_name}][fold {fold_idx + 1}] best val_loss: {val_out.get('val_loss', float('nan')):.4f}")

            dm.set_fold(fold_idx)
            dm.setup(stage="test")
            test_out = {}
            if dm.test_dataloader() is not None:
                test_out = trainer.test(
                    model=None,
                    dataloaders=dm.test_dataloader(),
                    ckpt_path=best_ckpt,
                    verbose=False
                )[0]

            fold_record.update({k: float(v) for k, v in val_out.items()})
            fold_record.update({k: float(v) for k, v in test_out.items()})

            print(f"[{mode_name}][fold {fold_idx + 1}] Saving predictions vs. ground truth...")

            model = RockModel.load_from_checkpoint(best_ckpt)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()

            dm.set_fold(fold_idx)
            dm.setup(stage="test")

            preds, gts, img_names = [], [], []

            for batch in dm.test_dataloader():
                img = batch["image"].to("cuda" if torch.cuda.is_available() else "cpu")
                psd_gt = batch["psd"]
                names = batch["img_name"]

                with torch.no_grad():
                    psd_pred = model(img)

                preds.append(psd_pred.cpu() * target_scale)
                gts.append(psd_gt.cpu() * target_scale)
                img_names.extend(names)

            preds = torch.cat(preds)
            gts = torch.cat(gts)

            pred_array = torch.cat([gts, preds], dim=1).numpy()
            pred_df = pd.DataFrame(
                pred_array,
                columns=[
                    "D20_gt", "D40_gt", "D60_gt", "D80_gt", "max_gt",
                    "D20_pred", "D40_pred", "D60_pred", "D80_pred", "max_pred"
                ]
            )
            pred_df.insert(0, "image_name", img_names)
            pred_df.insert(1, "test_area", test_area)
            pred_df.insert(2, "train_areas", "+".join(train_areas))
            pred_df.insert(3, "fold", fold_idx + 1)

            pred_csv = os.path.join(curve_dir, f"fold{fold_idx + 1}_pred_vs_gt.csv")
            pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
            all_pred_records.append(pred_df)
            all_pred_csv = os.path.join(curve_dir, "all_pred_vs_gt.csv")
            pd.concat(all_pred_records, axis=0, ignore_index=True).to_csv(
                all_pred_csv,
                index=False,
                encoding="utf-8-sig",
            )
            print(f"[{mode_name}][fold {fold_idx + 1}] predictions saved to {pred_csv}")
            print(f"[{mode_name}][fold {fold_idx + 1}] merged predictions saved to {all_pred_csv}")

            print(f"[{mode_name}][fold {fold_idx + 1}] first 10 saved image names:")
            print(pred_df["image_name"].head(10).to_string(index=False))

        else:
            print(f"[{mode_name}][fold {fold_idx + 1}] no checkpoint saved.")

        all_fold_metrics.append(fold_record)

        df = pd.DataFrame(all_fold_metrics)
        per_fold_csv = os.path.join(curve_dir, "per_fold_metrics.csv")
        per_fold_xlsx = os.path.join(curve_dir, "per_fold_metrics.xlsx")
        df.to_csv(per_fold_csv, index=False)
        df.to_excel(per_fold_xlsx, index=False)
        print(f"[{mode_name}] per-fold metrics saved to:\n  {per_fold_csv}\n  {per_fold_xlsx}")

        num_cols = df.select_dtypes(include=[np.number]).columns
        mean_row = df[num_cols].mean().to_frame().T
        std_row = df[num_cols].std(ddof=0).to_frame().T
        mean_row.insert(0, "stat", "mean")
        std_row.insert(0, "stat", "std")

        summary = pd.concat([mean_row, std_row], ignore_index=True)
        summary_csv = os.path.join(curve_dir, "metrics_summary.csv")
        summary_xlsx = os.path.join(curve_dir, "metrics_summary.xlsx")
        summary.to_csv(summary_csv, index=False)
        summary.to_excel(summary_xlsx, index=False)

        print(f"[{mode_name}] metrics summary saved to:\n  {summary_csv}\n  {summary_xlsx}")
        print("\n===== CV Summary So Far =====")
        for col in num_cols:
            mean_v = mean_row[col].values[0]
            std_v = std_row[col].values[0]
            print(f"{col:30s}: {mean_v:.4f} ± {std_v:.4f}")

    return last_best_ckpt


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", type=str, choices=["conv_only", "swin_only", "conv_swin"], default="conv_only")
    parser.add_argument("--f1", type=str, choices=FUSION_MODES, default="skip")
    parser.add_argument("--f2", type=str, choices=FUSION_MODES, default="add")
    parser.add_argument("--f3", type=str, choices=FUSION_MODES, default="gated")
    parser.add_argument("--f4", type=str, choices=FUSION_MODES, default="concat")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--train_metrics_every",
        type=int,
        default=0,
        help="Compute full train MAE/RMSE/R2 every N epochs. 0 disables the extra train-set pass.",
    )
    parser.add_argument(
        "--target_scale",
        type=float,
        default=1.0,
        help="Divide PSD targets by this value for training, then restore predictions in saved pred_vs_gt files.",
    )
    parser.add_argument("--data_dir", type=str, default="./PSD_dataset/images")
    parser.add_argument("--ann_file", type=str, default="./PSD_dataset/annotations.csv")
    parser.add_argument("--aux_dir", type=str, default=None)
    args = parser.parse_args()

    branch_type = args.branches
    stage_fusions = [args.f1, args.f2, args.f3, args.f4]

    data_dir = args.data_dir
    ann_file = args.ann_file
    aux_dir = args.aux_dir
    batch_size = args.batch_size
    epochs = args.epochs
    n_splits = args.splits
    num_workers = args.num_workers
    train_metrics_every = args.train_metrics_every
    target_scale = args.target_scale

    base_ckpt_dir = "./checkpoints/final_runs"
    base_log_dir = "./logs/final_runs"
    base_curve_dir = "./loss_curves/final_runs"

    fusion_tag = "_".join(stage_fusions)
    run_id = get_run_id(branch_type, fusion_tag)

    print(f"Running with {branch_type} | stage_fusions={stage_fusions} | run_id={run_id}")

    run_cv(
        branch_type=branch_type,
        stage_fusions=stage_fusions,
        data_dir=data_dir,
        ann_file=ann_file,
        aux_dir=aux_dir,
        batch_size=batch_size,
        n_splits=n_splits,
        epochs=epochs,
        num_workers=num_workers,
        train_metrics_every=train_metrics_every,
        target_scale=target_scale,
        base_ckpt_dir=base_ckpt_dir,
        base_log_dir=base_log_dir,
        base_curve_dir=base_curve_dir,
        run_id=run_id
    )


if __name__ == "__main__":
    main()
