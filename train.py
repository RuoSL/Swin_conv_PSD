import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
import cv2
import random
from sklearn.model_selection import KFold
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import argparse
from timm.models import create_model
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
RUN_ID_ROOT = datetime.now().strftime("%Y%m%d-%H%M%S")

def get_run_id(branch_type: str, fusion_type: str) -> str:
    tag = f"{branch_type}_{fusion_type}".replace("+", "_")
    rid = os.environ.get(f"RUN_ID_{tag}")
    if not rid:
        rid = f"{RUN_ID_ROOT}-{tag}"
        os.environ[f"RUN_ID_{tag}"] = rid
    return rid

# ---------------------- Seeds & Logging ----------------------
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

# ---------------------- Paired Augment ----------------------

class PairedAugment:
    def __init__(self, size=(224, 224), mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.size = size
        self.mean = mean
        self.std = std

        self.pil_aug = T.Compose([
            T.Resize(size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ])

        self.to_tensor = T.Compose([
             T.ToTensor(),
             T.Normalize(mean=self.mean, std=self.std)
    ])

    def __call__(self, img, aux_map):
        # Apply PIL augmentations
        img = self.pil_aug(img)

        # Convert to tensor
        img = self.to_tensor(img)

        # Handle aux map
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

def get_transforms():
    return PairedAugment(size=(224, 224))

# ---------------------- Dataset ----------------------
class RockDataset(Dataset):
    def __init__(self, root, transform=None, annotation_file=None, aux_dir=None):
        self.root = root
        self.aux_dir = aux_dir
        self.transform = transform or (lambda img, aux: (transforms.ToTensor()(img), aux))
        self.transform = transform
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.annotations = {}
        if annotation_file:
            df = pd.read_csv(annotation_file)
            print("CSV columns:", df.columns.tolist())
            for _, row in df.iterrows():
                self.annotations[row['image_name']] = {
                    'psd': [row['D20'], row['D40'], row['D60'], row['D80'], row['max']],
                    'rosin': [row['lambda'], row['k']],
                }
        self.image_paths = [p for p in self.image_paths if os.path.basename(p) in self.annotations]

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
            'image': img,
            'aux_map': aux,
            'psd': torch.tensor(ann['psd'], dtype=torch.float32),
            'rosin': torch.tensor(ann['rosin'], dtype=torch.float32),
            'img_name': img_name
        }

# ---------------------- Collate ----------------------
def rock_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise RuntimeError("Empty batch after filtering None samples")
    out = {}
    out['image'] = torch.stack([b['image'] for b in batch], dim=0)
    out['psd'] = torch.stack([b['psd'] for b in batch], dim=0)
    out['rosin'] = torch.stack([b['rosin'] for b in batch], dim=0)
    out['img_name'] = [b.get('img_name', '') for b in batch]
    aux_list = [b.get('aux_map', None) for b in batch]
    if all(a is not None for a in aux_list):
        out['aux_map'] = torch.stack(aux_list, dim=0)
    return out

# ---------------------- DataModule ----------------------
class RockDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, ann_file, batch_size=16, n_splits=5, aux_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.aux_dir = aux_dir
        self.batch_size = batch_size
        self.transform = get_transforms()
        self.n_splits = n_splits
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.kfold = None

    def setup(self, stage=None, fold_idx=0):
        full_dataset = RockDataset(
            self.data_dir,
            self.transform,
            self.ann_file,
            aux_dir=self.aux_dir
        )
        total_size = len(full_dataset)
        print(f"Total dataset size: {total_size}")

        train_val_size = int(0.85 * total_size)
        test_size = total_size - train_val_size
        train_val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Train+Val size: {train_val_size}, Test size: {test_size}")

        self.kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        dataset_indices = list(range(len(train_val_dataset)))
        fold_train_idx, fold_val_idx = list(self.kfold.split(dataset_indices))[fold_idx]

        self.train_dataset = torch.utils.data.Subset(train_val_dataset, fold_train_idx)
        self.val_dataset = torch.utils.data.Subset(train_val_dataset, fold_val_idx)

        print(f"Fold {fold_idx + 1}/{self.n_splits}: "
              f"Train size: {len(fold_train_idx)}, Validation size: {len(fold_val_idx)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=8, pin_memory=True, collate_fn=rock_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=8, pin_memory=True, drop_last=False, collate_fn=rock_collate_fn
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=8, pin_memory=True, collate_fn=rock_collate_fn
        )

# ---------------------- Modules ----------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_conv, x_swin):
        attn_out, _ = self.attn(x_conv, x_swin, x_swin)
        return self.proj(attn_out)


class RockModel(pl.LightningModule):
    def __init__(self, branch_type="conv_only", stage_modes=None, lr=1e-4, weight_decay=1e-5, enable_shapley=False):
        super().__init__()
        self.save_hyperparameters()
        self.branch_type = branch_type.lower()
        self.enable_shapley = enable_shapley

        # --- Loss  ---
        self.w_psd = 1.0
        self.w_rosin = 1.0
        self.w_rank = 0.1

        # --- Metrics  ---
        self.train_losses = []
        self.val_psd_preds, self.val_psd_gts = [], []
        self.val_rosin_preds, self.val_rosin_gts = [], []
        self.test_psd_preds, self.test_psd_gts = [], []
        self.test_rosin_preds, self.test_rosin_gts = [], []

        self.fold_metrics = {key: [] for key in [
            'test_loss', 'test_rank_violation_rate',
            'test_psd_mae', 'test_psd_rmse', 'test_psd_mape', 'test_psd_r2',
            'test_rosin_mae', 'test_rosin_rmse', 'test_rosin_mape', 'test_rosin_r2',
            'test_psd_D20_mae', 'test_psd_D20_rmse', 'test_psd_D20_mape', 'test_psd_D20_r2',
            'test_psd_D40_mae', 'test_psd_D40_rmse', 'test_psd_D40_mape', 'test_psd_D40_r2',
            'test_psd_D60_mae', 'test_psd_D60_rmse', 'test_psd_D60_mape', 'test_psd_D60_r2',
            'test_psd_D80_mae', 'test_psd_D80_rmse', 'test_psd_D80_mape', 'test_psd_D80_r2',
            'test_psd_max_mae', 'test_psd_max_rmse', 'test_psd_max_mape', 'test_psd_max_r2',
            'test_rosin_lambda_mae', 'test_rosin_lambda_rmse', 'test_rosin_lambda_mape', 'test_rosin_lambda_r2',
            'test_rosin_k_mae', 'test_rosin_k_rmse', 'test_rosin_k_mape', 'test_rosin_k_r2',
        ]}

        if self.enable_shapley and self.branch_type == "conv_swin":
            self._phi_conv = []
            self._phi_swin = []
            self.shapley_history = []

        # --- Backbone & Fusion ---
        self.C = [128, 192, 256, 384]
        self.concat_reduce = nn.ModuleList([
            nn.Conv2d(self.C[i] * 2, self.C[i], 1) for i in range(4)
        ])

        if self.branch_type == "conv_only":
            self.backbone = create_model("convnext_tiny", pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
            self.proj = nn.ModuleList([nn.Conv2d(ch, self.C[i], 1, bias=False) for i, ch in enumerate(self.backbone.feature_info.channels())])
            self.is_dual = False

        elif self.branch_type == "swin_only":
            self.backbone = create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
            self.proj = nn.ModuleList([nn.Conv2d(ch, self.C[i], 1, bias=False) for i, ch in enumerate(self.backbone.feature_info.channels())])
            self.is_dual = False

        elif self.branch_type == "conv_swin":
            self.convnext = create_model("convnext_tiny", pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
            self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
            self.swin.patch_embed.img_size = (224, 224)
            self.swin.patch_embed.grid_size = (224 // 4, 224 // 4)
            cnx_chs = self.convnext.feature_info.channels()
            swin_chs = self.swin.feature_info.channels()
            self.cnx_proj = nn.ModuleList([nn.Conv2d(c, self.C[i], 1, bias=False) if c > 0 else nn.Identity() for i, c in enumerate(cnx_chs)])
            self.swin_proj = nn.ModuleList([nn.Conv2d(c, self.C[i], 1, bias=False) if c > 0 else nn.Identity() for i, c in enumerate(swin_chs)])
            self.stage_modes = stage_modes or ['skip', 'add', 'concat', 'gated']
            assert len(self.stage_modes) == 4, "stage_modes must be a list of length 4"
            self.fusers = nn.ModuleList([self._make_fuser(self.C[i], mode) for i, mode in enumerate(self.stage_modes)])
            self.is_dual = True
        else:
            raise ValueError(f"Unknown branch_type: {self.branch_type}")

        in_dim = sum(self.C)
        self.fusion_proj = nn.Linear(in_dim, 768)

        # --- Head ---
        self.fusion = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.psd_head = nn.Linear(256, 5)
        self.rosin_head = nn.Linear(256, 2)

    def _make_fuser(self, dim, mode):
        if mode == "cross_attention":
            return CrossAttentionFusion(dim=dim, num_heads=4)
        elif mode == "gated":
            return nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1),
                nn.Sigmoid()
            )
        else:
            return None

    def _ensure_nchw(self, x, expected_c=None):
        if x is None:
            return None
        if x.ndim == 4:
            c_nchw, c_nhwc = x.shape[1], x.shape[-1]
            if expected_c is not None and c_nchw != expected_c and c_nhwc == expected_c:
                x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, image, aux_map=None):
        if self.branch_type == "conv_only" or self.branch_type == "swin_only":
            feats = self.backbone(image)
            pyramid = []
            for i, feat in enumerate(feats):
                feat = self._ensure_nchw(feat, expected_c=self.proj[i].in_channels if hasattr(self.proj[i], 'in_channels') else None)
                proj_feat = self.proj[i](feat)
                pyramid.append(proj_feat)
        elif self.branch_type == "conv_swin":
            cnx_feats = self.convnext(image)
            swin_feats = self.swin(image)
            pyramid = []
            for i in range(4):
                mode = self.stage_modes[i]
                c_feat = self._ensure_nchw(cnx_feats[i], expected_c=self.cnx_proj[i].in_channels if hasattr(self.cnx_proj[i], 'in_channels') else None)
                s_feat = self._ensure_nchw(swin_feats[i], expected_c=self.swin_proj[i].in_channels if hasattr(self.swin_proj[i], 'in_channels') else None)
                c_feat = self.cnx_proj[i](c_feat)
                s_feat = self.swin_proj[i](s_feat)

                if mode == "skip":
                    fused = c_feat
                elif mode == "add":
                    fused = (c_feat + s_feat) / 2
                elif mode == "concat":
                    fused = torch.cat([c_feat, s_feat], dim=1)
                    fused = self.concat_reduce[i](fused)
                elif mode == "gated":
                    gate = self.fusers[i](torch.cat([c_feat, s_feat], dim=1))
                    fused = gate * c_feat + (1 - gate) * s_feat
                elif mode == "cross_attention":
                    B, C, H, W = c_feat.shape
                    c_flat = c_feat.flatten(2).transpose(1, 2)
                    s_flat = s_feat.flatten(2).transpose(1, 2)
                    fused_flat = self.fusers[i](c_flat, s_flat)
                    fused = fused_flat.transpose(1, 2).reshape(B, C, H, W)
                else:
                    raise ValueError(f"Unknown stage mode {mode}")
                pyramid.append(fused)


        vecs = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in pyramid]
        fused_vec = torch.cat(vecs, dim=1)
        fused = self.fusion_proj(fused_vec)
        feat = self.fusion(fused)
        psd = self.psd_head(feat)
        rosin = self.rosin_head(feat)
        return psd, rosin

    def _total_loss(self, psd_pred, rosin_pred, psd_gt, rosin_gt):
        l_psd = F.smooth_l1_loss(psd_pred, psd_gt, reduction='mean')
        l_ros = F.smooth_l1_loss(rosin_pred, rosin_gt, reduction='mean')
        l_rank = 0.0
        if self.w_rank > 0:
            rl = 0.0
            for i in range(4):
                rl = rl + F.relu(psd_pred[:, i] - psd_pred[:, i + 1]).mean()
            l_rank = self.w_rank * rl
        return self.w_psd * l_psd + self.w_rosin * l_ros + l_rank


    def rank_loss(self, psd_pred):
        loss = 0
        for i in range(4):
            loss += F.relu(psd_pred[:, i] - psd_pred[:, i + 1])
        return loss.mean()

    @torch.no_grad()
    def _forward_masked(self, image, aux_map, use_conv=True, use_swin=True, use_edge=True):
        feats = []
        if use_conv and "convnext" in self.branch_type:
            conv_feat = self.convnext(image)
            conv_proj = self.conv_proj(conv_feat)
            feats.append(conv_proj)
        if use_swin and "swin" in self.branch_type:
            swin_feat = self.swin(image)
            swin_proj = self.swin_proj(swin_feat)
            feats.append(swin_proj)
        if use_edge and "edge" in self.branch_type and aux_map is not None:
            edge_feat = self.edge_branch(aux_map)
            edge_proj = self.edge_proj(edge_feat)
            feats.append(edge_proj)
        if not feats:
            raise ValueError(f"No features extracted in _forward_masked for branch_type: {self.branch_type}")

        if len(feats) == 1:
            fused = feats[0]
        elif self.fusion_type == "concat":
            fused = torch.cat(feats, dim=1)
        elif self.fusion_type == "add":
            fused = sum(feats) / len(feats)
        elif self.fusion_type == "gated":
            fused = self.gated_fusion(feats)
        elif self.fusion_type == "cross_attn":
            fused = feats[0]
            for other in feats[1:]:
                fused = self.cross_fusion(fused, other)
        else:
            raise ValueError(f"Unknown fusion_type {self.fusion_type}")
        fused = self.fusion_proj(fused)
        feat = self.fusion(fused)
        psd = self.psd_head(feat)
        rosin = self.rosin_head(feat)
        return psd, rosin

    @torch.no_grad()
    def compute_shapley_batch(self, batch):
        image = batch['image'].to(self.device)
        psd_gt = batch['psd'].to(self.device)
        rosin_gt = batch['rosin'].to(self.device)
        aux_map = batch.get('aux_map', None)
        if aux_map is not None:
            aux_map = aux_map.to(self.device)
        subset_loss = {}
        for uc in [0, 1]:
            for us in [0, 1]:
                for ue in [0, 1]:
                    psd_p, ros_p = self._forward_masked(image, aux_map, bool(uc), bool(us), bool(ue))
                    subset_loss[(uc, us, ue)] = self._total_loss(psd_p, ros_p, psd_gt, rosin_gt).item()
        from itertools import permutations
        branches = ["conv", "swin", "edge"]
        phi = {b: 0.0 for b in branches}
        perms = list(permutations(branches))
        for order in perms:
            have = {"conv": 0, "swin": 0, "edge": 0}
            prev = subset_loss[(0, 0, 0)]
            for b in order:
                have[b] = 1
                cur = subset_loss[(have["conv"], have["swin"], have["edge"])]
                phi[b] += (prev - cur)
                prev = cur
        for b in branches:
            phi[b] /= len(perms)
        return phi

    def training_step(self, batch, batch_idx):
        img, psd_gt, rosin_gt = batch['image'], batch['psd'], batch['rosin']
        aux_map = batch.get('aux_map', None)
        psd_pred, rosin_pred = self(img, aux_map=aux_map)
        l_psd = F.smooth_l1_loss(psd_pred, psd_gt)
        l_rosin = F.smooth_l1_loss(rosin_pred, rosin_gt)
        l_rank = self.rank_loss(psd_pred)
        loss = (self.w_psd * l_psd + self.w_rosin * l_rosin + self.w_rank * l_rank)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss at Epoch {self.current_epoch}, Batch {batch_idx}")
            loss = torch.tensor(0.0, device=loss.device)
        self.train_losses.append(loss)
        self.log('train_loss_batch', loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img, psd_gt, rosin_gt = batch['image'], batch['psd'], batch['rosin']
        aux_map = batch.get('aux_map', None)
        psd_pred, rosin_pred = self(img, aux_map=aux_map)
        l_psd = F.smooth_l1_loss(psd_pred, psd_gt)
        l_rosin = F.smooth_l1_loss(rosin_pred, rosin_gt)
        l_rank = self.rank_loss(psd_pred)
        loss = (self.w_psd * l_psd + self.w_rosin * l_rosin + self.w_rank * l_rank)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.val_psd_preds.append(psd_pred.detach().cpu())
        self.val_psd_gts.append(psd_gt.detach().cpu())
        self.val_rosin_preds.append(rosin_pred.detach().cpu())
        self.val_rosin_gts.append(rosin_gt.detach().cpu())

        if self.enable_shapley and batch_idx == self.trainer.num_val_batches[0] - 1:
            phi = self.compute_shapley_batch(batch)
            self._phi_conv.append(phi["conv"])
            self._phi_swin.append(phi["swin"])
            self._phi_edge.append(phi["edge"])

    def test_step(self, batch, batch_idx):
        img, psd_gt, rosin_gt = batch['image'], batch['psd'], batch['rosin']
        aux_map = batch.get('aux_map', None)
        psd_pred, rosin_pred = self(img, aux_map=aux_map)
        l_psd = F.smooth_l1_loss(psd_pred, psd_gt)
        l_rosin = F.smooth_l1_loss(rosin_pred, rosin_gt)
        l_rank = self.rank_loss(psd_pred)
        loss = (self.w_psd * l_psd + self.w_rosin * l_rosin + self.w_rank * l_rank)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        psd_mape = torch.abs((psd_gt - psd_pred) / (psd_gt + 1e-8)) * 100
        rosin_mape = torch.abs((rosin_gt - rosin_pred) / (rosin_gt + 1e-8)) * 100
        bs = img.size(0)
        for i in range(psd_mape.shape[0]):
            idx_print = batch_idx * bs + i

        self.test_psd_preds.append(psd_pred.detach().cpu())
        self.test_psd_gts.append(psd_gt.detach().cpu())
        self.test_rosin_preds.append(rosin_pred.detach().cpu())
        self.test_rosin_gts.append(rosin_gt.detach().cpu())

    def on_validation_epoch_end(self):
        psd_preds = torch.cat(self.val_psd_preds, dim=0)
        psd_gts = torch.cat(self.val_psd_gts, dim=0)
        rosin_preds = torch.cat(self.val_rosin_preds, dim=0)
        rosin_gts = torch.cat(self.val_rosin_gts, dim=0)

        psd_mae = torch.mean(torch.abs(psd_preds - psd_gts))
        psd_rmse = torch.sqrt(F.mse_loss(psd_preds, psd_gts))
        psd_mape = torch.mean(torch.abs((psd_gts - psd_preds) / (psd_gts + 1e-8))) * 100
        psd_mean = torch.mean(psd_gts, dim=0, keepdim=True)
        ss_tot_psd = torch.sum((psd_gts - psd_mean) ** 2, dim=0)
        ss_res_psd = torch.sum((psd_gts - psd_preds) ** 2, dim=0)
        psd_r2_sep = 1 - ss_res_psd / (ss_tot_psd + 1e-8)
        psd_r2 = torch.mean(psd_r2_sep)

        rosin_mae = torch.mean(torch.abs(rosin_preds - rosin_gts))
        rosin_rmse = torch.sqrt(F.mse_loss(rosin_preds, rosin_gts))
        rosin_mape = torch.mean(torch.abs((rosin_gts - rosin_preds) / (rosin_gts + 1e-8))) * 100
        rosin_mean = torch.mean(rosin_gts, dim=0, keepdim=True)
        ss_tot_rosin = torch.sum((rosin_gts - rosin_mean) ** 2, dim=0)
        ss_res_rosin = torch.sum((rosin_gts - rosin_preds) ** 2, dim=0)
        rosin_r2_sep = 1 - ss_res_rosin / (ss_tot_rosin + 1e-8)
        rosin_r2 = torch.mean(rosin_r2_sep)

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
        print(f"Rosin MAE: {rosin_mae.item():.4f}")
        print(f"Rosin RMSE: {rosin_rmse.item():.4f}")
        print(f"Rosin MAPE: {rosin_mape.item():.2f}%")
        print(f"Rosin R²: {rosin_r2.item():.4f}")
        print(f"Rank Violation Rate: {rank_violation_rate:.4f}")

        psd_labels = ['D20', 'D40', 'D60', 'D80', 'max']
        print("\nSeparate PSD R² Metrics:")
        for i, label in enumerate(psd_labels):
            print(f"{label} R²: {psd_r2_sep[i].item():.4f}")

        rosin_labels = ['lambda', 'k']
        print("\nSeparate Rosin R² Metrics:")
        for i, label in enumerate(rosin_labels):
            print(f"{label} R²: {rosin_r2_sep[i].item():.4f}")

        for i, label in enumerate(psd_labels):
            self.log(f'val_psd_{label}_r2', psd_r2_sep[i], logger=True)
        for i, label in enumerate(rosin_labels):
            self.log(f'val_rosin_{label}_r2', rosin_r2_sep[i], logger=True)

        self.log('val_psd_mae', psd_mae, logger=True, prog_bar=True)
        self.log('val_psd_rmse', psd_rmse, logger=True)
        self.log('val_psd_mape', psd_mape, logger=True)
        self.log('val_psd_r2', psd_r2, logger=True)
        self.log('val_rosin_mae', rosin_mae, logger=True, prog_bar=True)
        self.log('val_rosin_rmse', rosin_rmse, logger=True)
        self.log('val_rosin_mape', rosin_mape, logger=True)
        self.log('val_rosin_r2', rosin_r2, logger=True)
        self.log('val_rank_violation', rank_violation_rate, logger=True)

        if self.enable_shapley and len(self._phi_conv) > 0:
            phi_mean = {
                "conv": np.mean(self._phi_conv),
                "swin": np.mean(self._phi_swin),
                "edge": np.mean(self._phi_edge),
                "epoch": self.current_epoch,
            }
            self.shapley_history.append(phi_mean)
            self._phi_conv.clear()
            self._phi_swin.clear()
            self._phi_edge.clear()

        self.val_psd_preds.clear()
        self.val_psd_gts.clear()
        self.val_rosin_preds.clear()
        self.val_rosin_gts.clear()

    def on_test_epoch_end(self):
        psd_preds = torch.cat(self.test_psd_preds, dim=0)
        psd_gts = torch.cat(self.test_psd_gts, dim=0)
        rosin_preds = torch.cat(self.test_rosin_preds, dim=0)
        rosin_gts = torch.cat(self.test_rosin_gts, dim=0)

        psd_mae = torch.mean(torch.abs(psd_preds - psd_gts))
        psd_rmse = torch.sqrt(F.mse_loss(psd_preds, psd_gts))
        psd_mape = torch.mean(torch.abs((psd_gts - psd_preds) / (psd_gts + 1e-8))) * 100
        psd_mean = torch.mean(psd_gts, dim=0, keepdim=True)
        ss_tot_psd = torch.sum((psd_gts - psd_mean) ** 2, dim=0)
        ss_res_psd = torch.sum((psd_gts - psd_preds) ** 2, dim=0)
        psd_r2_sep = 1 - ss_res_psd / (ss_tot_psd + 1e-8)
        psd_r2 = 1 - torch.sum(ss_res_psd) / (torch.sum(ss_tot_psd) + 1e-8)

        rosin_mae = torch.mean(torch.abs(rosin_preds - rosin_gts))
        rosin_rmse = torch.sqrt(F.mse_loss(rosin_preds, rosin_gts))
        rosin_mape = torch.mean(torch.abs((rosin_gts - rosin_preds) / (rosin_gts + 1e-8))) * 100
        rosin_mean = torch.mean(rosin_gts, dim=0, keepdim=True)
        ss_tot_rosin = torch.sum((rosin_gts - rosin_mean) ** 2, dim=0)
        ss_res_rosin = torch.sum((rosin_gts - rosin_preds) ** 2, dim=0)
        rosin_r2_sep = 1 - ss_res_rosin / (ss_tot_rosin + 1e-8)
        rosin_r2 = 1 - torch.sum(ss_res_rosin) / (torch.sum(ss_tot_rosin) + 1e-8)

        violations = sum(
            any(psd_preds[i, j] >= psd_preds[i, j + 1] for j in range(4))
            for i in range(psd_preds.shape[0])
        )
        rank_violation_rate = violations / psd_preds.shape[0]

        psd_labels = ['D20', 'D40', 'D60', 'D80', 'max']
        rosin_labels = ['lambda', 'k']
        psd_mae_sep = torch.mean(torch.abs(psd_preds - psd_gts), dim=0)
        psd_rmse_sep = torch.sqrt(torch.mean((psd_preds - psd_gts) ** 2, dim=0))
        psd_mape_sep = torch.mean(torch.abs((psd_gts - psd_preds) / (psd_gts + 1e-8)), dim=0) * 100
        rosin_mae_sep = torch.mean(torch.abs(rosin_preds - rosin_gts), dim=0)
        rosin_rmse_sep = torch.sqrt(torch.mean((rosin_preds - rosin_gts) ** 2, dim=0))
        rosin_mape_sep = torch.mean(torch.abs((rosin_gts - rosin_preds) / (rosin_gts + 1e-8)), dim=0) * 100

        # === Log 所有指标（让 run_cv() 自动收集）===
        self.log('test_psd_mae', psd_mae, logger=True)
        self.log('test_psd_rmse', psd_rmse, logger=True)
        self.log('test_psd_mape', psd_mape, logger=True)
        self.log('test_psd_r2', psd_r2, logger=True)
        self.log('test_rosin_mae', rosin_mae, logger=True)
        self.log('test_rosin_rmse', rosin_rmse, logger=True)
        self.log('test_rosin_mape', rosin_mape, logger=True)
        self.log('test_rosin_r2', rosin_r2, logger=True)
        self.log('test_rank_violation_rate', rank_violation_rate, logger=True)

        # 各通道 log
        for i, label in enumerate(psd_labels):
            self.log(f'test_psd_{label}_mae', psd_mae_sep[i], logger=True)
            self.log(f'test_psd_{label}_rmse', psd_rmse_sep[i], logger=True)
            self.log(f'test_psd_{label}_mape', psd_mape_sep[i], logger=True)
            self.log(f'test_psd_{label}_r2', psd_r2_sep[i], logger=True)
        for i, label in enumerate(rosin_labels):
            self.log(f'test_rosin_{label}_mae', rosin_mae_sep[i], logger=True)
            self.log(f'test_rosin_{label}_rmse', rosin_rmse_sep[i], logger=True)
            self.log(f'test_rosin_{label}_mape', rosin_mape_sep[i], logger=True)
            self.log(f'test_rosin_{label}_r2', rosin_r2_sep[i], logger=True)

        # === 清空缓存 ===
        self.test_psd_preds.clear()
        self.test_psd_gts.clear()
        self.test_rosin_preds.clear()
        self.test_rosin_gts.clear()

    def configure_optimizers(self):
        param_groups = []
        params_main = []
        for name, param in self.named_parameters():
            if not name.startswith("edge_branch"):
                params_main.append(param)
        param_groups.append({"params": params_main, "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay})
        if "edge" in self.branch_type:
            params_edge = list(self.edge_branch.parameters())
            param_groups.append(
                {"params": params_edge, "lr": self.hparams.lr * 2, "weight_decay": self.hparams.weight_decay})
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def visualize_fusion(self, image, aux_map=None, target_idx=0, save_dir="./vis_out", base_name="sample"):
        os.makedirs(save_dir, exist_ok=True)
        if self.fusion_type in ["concat", "add", "gated"]:
            heatmap = self.get_gradcam(image, aux_map=aux_map, target_idx=target_idx)
            import matplotlib.pyplot as plt
            plt.imshow(heatmap, cmap="jet", alpha=0.8)
            out_path = os.path.join(save_dir, f"{base_name}_gradcam.png")
            plt.axis("off")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            return {"type": "gradcam", "map": heatmap, "path": out_path}
        elif self.fusion_type == "cross_attn":
            self.cross_fusion.save_attention = True
            _ = self(image.unsqueeze(0), aux_map.unsqueeze(0) if aux_map is not None else None)
            attn_map = self.cross_fusion.attn_weights
            if attn_map.dim() == 4:
                attn_map = attn_map.mean(1)[0].detach().cpu().numpy()
            elif attn_map.dim() == 3:
                attn_map = attn_map[0].detach().cpu().numpy()
            elif attn_map.dim() == 2:
                attn_map = attn_map.detach().cpu().numpy()
            else:
                raise ValueError(f"Unexpected attn_map shape: {attn_map.shape}")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            plt.imshow(attn_map, cmap="viridis")
            plt.colorbar()
            out_path = os.path.join(save_dir, f"{base_name}_crossattn.png")
            plt.title("Cross-Attention Heatmap")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            return {"type": "cross_attn", "map": attn_map, "path": out_path}
        else:
            raise ValueError(f"Unsupported fusion type {self.fusion_type}")

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
        out = self(img, aux_map=aux_map)
        psd, _ = out
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
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(img.shape[2], img.shape[3]),
                                mode="bilinear", align_corners=False).squeeze()
        return heatmap.numpy()

# ---------------------- Loss Curve Callback ----------------------
class LossCurveCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.history = []
        self._last_train = None
        self.train_metrics = {}

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
            train_loss_epoch = float('nan')
        self._last_train = train_loss_epoch
        pl_module.train_losses.clear()

        pl_module.eval()
        psd_preds, psd_gts, rosin_preds, rosin_gts = [], [], [], []
        with torch.no_grad():
            for batch in trainer.datamodule.train_dataloader():
                img, psd_gt, rosin_gt = batch['image'].to(pl_module.device), batch['psd'].to(pl_module.device), batch['rosin'].to(pl_module.device)
                aux_map = batch.get('aux_map', None)
                if aux_map is not None:
                    aux_map = aux_map.to(pl_module.device)
                psd_pred, rosin_pred = pl_module(img, aux_map=aux_map)
                psd_preds.append(psd_pred.cpu())
                psd_gts.append(psd_gt.cpu())
                rosin_preds.append(rosin_pred.cpu())
                rosin_gts.append(rosin_gt.cpu())

        psd_preds, psd_gts = torch.cat(psd_preds), torch.cat(psd_gts)
        rosin_preds, rosin_gts = torch.cat(rosin_preds), torch.cat(rosin_gts)

        psd_mae, psd_rmse, psd_mape, psd_r2 = self.compute_metrics(psd_preds, psd_gts)
        rosin_mae, rosin_rmse, rosin_mape, rosin_r2 = self.compute_metrics(rosin_preds, rosin_gts)

        violations = sum(any(pred[j] >= pred[j + 1] for j in range(4)) for pred in psd_preds)
        rank_violation_rate = violations / psd_preds.shape[0]

        self.train_metrics = {
            "train_loss": train_loss_epoch,
            "train_psd_mae": psd_mae, "train_psd_rmse": psd_rmse, "train_psd_mape": psd_mape, "train_psd_r2": psd_r2,
            "train_rosin_mae": rosin_mae, "train_rosin_rmse": rosin_rmse, "train_rosin_mape": rosin_mape, "train_rosin_r2": rosin_r2,
            "train_rank_violation": rank_violation_rate,
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val_loss = trainer.callback_metrics.get('val_loss')
        val_loss = val_loss.item() if val_loss is not None else float('nan')
        record = {"epoch": int(trainer.current_epoch)}
        if self.train_metrics:
            record.update(self.train_metrics)
        else:
            record.update({
                "train_loss": float('nan'),
                "train_psd_mae": float('nan'),
                "train_psd_rmse": float('nan'),
                "train_psd_mape": float('nan'),
                "train_psd_r2": float('nan'),
                "train_rosin_mae": float('nan'),
                "train_rosin_rmse": float('nan'),
                "train_rosin_mape": float('nan'),
                "train_rosin_r2": float('nan'),
                "train_rank_violation": float('nan'),
            })
        record["val_loss"] = val_loss
        record["val_psd_mae"] = float(trainer.callback_metrics.get("val_psd_mae", torch.tensor(float('nan'))))
        record["val_psd_rmse"] = float(trainer.callback_metrics.get("val_psd_rmse", torch.tensor(float('nan'))))
        record["val_psd_mape"] = float(trainer.callback_metrics.get("val_psd_mape", torch.tensor(float('nan'))))
        record["val_psd_r2"] = float(trainer.callback_metrics.get("val_psd_r2", torch.tensor(float('nan'))))
        record["val_rosin_mae"] = float(trainer.callback_metrics.get("val_rosin_mae", torch.tensor(float('nan'))))
        record["val_rosin_rmse"] = float(trainer.callback_metrics.get("val_rosin_rmse", torch.tensor(float('nan'))))
        record["val_rosin_mape"] = float(trainer.callback_metrics.get("val_rosin_mape", torch.tensor(float('nan'))))
        record["val_rosin_r2"] = float(trainer.callback_metrics.get("val_rosin_r2", torch.tensor(float('nan'))))
        record["val_rank_violation"] = float(trainer.callback_metrics.get("val_rank_violation", torch.tensor(float('nan'))))
        self.history.append(record)
        print(f"[Epoch {record['epoch']}] TrainLoss={record['train_loss']:.4f}, ValLoss={record['val_loss']:.4f} | "
              f"Train PSD→ MAE {record['train_psd_mae']:.4f}, RMSE {record['train_psd_rmse']:.4f}, R² {record['train_psd_r2']:.4f} | "
              f"Val PSD→ MAE {record['val_psd_mae']:.4f}, RMSE {record['val_psd_rmse']:.4f}, R² {record['val_psd_r2']:.4f}")

def run_cv(
    branch_type: str,
    stage_modes: List[str],
    data_dir: str,
    ann_file: str,
    aux_dir: str,
    batch_size: int,
    n_splits: int,
    epochs: int,
    base_ckpt_dir: str,
    base_log_dir: str,
    base_curve_dir: str,
    run_id: str
):
    structure_tag = f"train_{branch_type.replace('+', '_')}_{'_'.join(stage_modes) if stage_modes else 'single'}"
    mode_name = branch_type

    print(f"\n===== Running mode: {mode_name} (branch_type={branch_type}, stage_modes={stage_modes}) =====\n")
    last_best_ckpt = None

    curve_dir = os.path.join(base_curve_dir, structure_tag, mode_name, run_id)
    os.makedirs(curve_dir, exist_ok=True)

    all_fold_metrics = []

    for fold_idx in range(n_splits):
        print(f"\nTraining Fold {fold_idx + 1}/{n_splits}...")

        dm = RockDataModule(
            data_dir=data_dir,
            ann_file=ann_file,
            batch_size=batch_size,
            n_splits=n_splits,
            aux_dir=None
        )
        dm.setup(stage='fit', fold_idx=fold_idx)

        model = RockModel(branch_type=branch_type, stage_modes=stage_modes, lr=1e-4, weight_decay=1e-3, enable_shapley=False)

        ckpt_dir = os.path.join(base_ckpt_dir, structure_tag, mode_name, run_id, f"fold{fold_idx + 1}")
        log_dir = os.path.join(base_log_dir, structure_tag, mode_name, run_id, f"fold{fold_idx + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)

        loss_cb = LossCurveCallback()
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"rock-{structure_tag}-fold{fold_idx + 1}-epoch{{epoch:02d}}",
            monitor="val_loss", mode="min", save_top_k=1
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[loss_cb, ckpt_cb],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=True,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            check_val_every_n_epoch=1,
            val_check_interval=1.0,
            benchmark=True,
            deterministic=False,
        )

        trainer.fit(model, datamodule=dm)

        loss_xlsx = os.path.join(curve_dir, f"fold{fold_idx + 1}_loss_curve.xlsx")
        pd.DataFrame(loss_cb.history).to_excel(loss_xlsx, index=False)
        print(f"[{mode_name}][fold {fold_idx + 1}] loss curve saved to: {loss_xlsx}")

        if hasattr(model, "shapley_history") and len(model.shapley_history) > 0:
            shp_df = pd.DataFrame(model.shapley_history)
            shp_csv = os.path.join(curve_dir, f"fold{fold_idx + 1}_shapley_curve.csv")
            shp_xlsx = os.path.join(curve_dir, f"fold{fold_idx + 1}_shapley_curve.xlsx")
            shp_df.to_csv(shp_csv, index=False)
            shp_df.to_excel(shp_xlsx, index=False)
            print(f"[{mode_name}][fold {fold_idx + 1}] shapley curve saved to:\n  {shp_csv}\n  {shp_xlsx}")
        else:
            print(f"[{mode_name}][fold {fold_idx + 1}] shapley history is empty.")

        best_ckpt = ckpt_cb.best_model_path
        fold_record = {"fold": fold_idx + 1, "mode": mode_name}

        if best_ckpt:
            last_best_ckpt = best_ckpt
            val_out = trainer.validate(model=None, dataloaders=dm.val_dataloader(), ckpt_path=best_ckpt, verbose=False)[
                0]
            print(f"[{mode_name}][fold {fold_idx + 1}] best val_loss: {val_out.get('val_loss', float('nan')):.4f}")
            dm.setup(stage='test', fold_idx=fold_idx)
            test_out = {}
            if dm.test_dataloader() is not None:
                test_out = \
                trainer.test(model=None, dataloaders=dm.test_dataloader(), ckpt_path=best_ckpt, verbose=False)[0]
            fold_record.update({k: float(v) for k, v in val_out.items()})
            fold_record.update({k: float(v) for k, v in test_out.items()})

            print(f"[{mode_name}][fold {fold_idx + 1}] Saving predictions vs. ground truth...")
            model = RockModel.load_from_checkpoint(best_ckpt)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            preds, gts = [], []
            for batch in dm.test_dataloader():
                img = batch['image']
                psd_gt = batch['psd']
                rosin_gt = batch['rosin']
                img = img.to("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    psd_pred, rosin_pred = model(img)
                preds.append(torch.cat([psd_pred.cpu(), rosin_pred.cpu()], dim=1))
                gts.append(torch.cat([psd_gt.cpu(), rosin_gt.cpu()], dim=1))
            preds = torch.cat(preds)
            gts = torch.cat(gts)
            pred_df = pd.DataFrame(
                torch.cat([gts, preds], dim=1).numpy(),
                columns=["D20_gt", "D40_gt", "D60_gt", "D80_gt", "max_gt", "lambda_gt", "k_gt",
                         "D20_pred", "D40_pred", "D60_pred", "D80_pred", "max_pred", "lambda_pred", "k_pred"]
            )
            pred_csv = os.path.join(curve_dir, f"fold{fold_idx + 1}_pred_vs_gt.csv")
            pred_df.to_csv(pred_csv, index=False)
            print(f"[{mode_name}][fold {fold_idx + 1}] predictions saved to {pred_csv}")

        else:
            print(f"[{mode_name}][fold {fold_idx + 1}] no checkpoint saved.")

        all_fold_metrics.append(fold_record)

        if all_fold_metrics:
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
            print("\n===== 5-Fold Average Results =====")
            for col in num_cols:
                mean_v = mean_row[col].values[0]
                std_v = std_row[col].values[0]
                print(f"{col:30s}: {mean_v:.4f} ± {std_v:.4f}")

    return last_best_ckpt

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", type=str, choices=["conv_only", "swin_only", "conv_swin"], default="conv_only",
                        help="Model type: conv_only, swin_only, or conv_swin (dual branch)")
    parser.add_argument("--stage_modes", type=str, default="skip,concat,gated,gated",
                        help="Stage fusion modes for conv_swin (comma-separated: skip/add/concat/gated/cross_attention)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--splits", type=int, default=5, help="Number of KFold splits")
    parser.add_argument("--data_dir", type=str, default="./PSD_dataset/patches_enhanced_images", help="RGB image directory")
    parser.add_argument("--ann_file", type=str, default="./PSD_dataset/annotations_scale.csv", help="Annotation file")
    args = parser.parse_args()

    branch_type = args.branches
    stage_modes_str = args.stage_modes.split(',') if branch_type == "conv_swin" else None
    stage_modes = stage_modes_str if stage_modes_str else None

    data_dir = args.data_dir
    ann_file = args.ann_file
    batch_size = args.batch_size
    epochs = args.epochs
    n_splits = args.splits

    base_ckpt_dir = "./checkpoints/final_runs"
    base_log_dir = "./logs/final_runs"
    base_curve_dir = "./loss_curves/final_runs"

    run_id = get_run_id(branch_type, '_'.join(stage_modes) if stage_modes else "single")

    print(f"⚡ Running with {branch_type} | stage_modes={stage_modes} | run_id={run_id}")

    best_ckpt = run_cv(
        branch_type=branch_type,
        stage_modes=stage_modes,
        data_dir=data_dir, ann_file=ann_file, aux_dir=None,
        batch_size=batch_size, n_splits=n_splits, epochs=epochs,
        base_ckpt_dir=base_ckpt_dir, base_log_dir=base_log_dir, base_curve_dir=base_curve_dir,
        run_id=run_id
    )

    structure_tag = f"train_{branch_type.replace('+', '_')}_{stage_modes}"
    mode_name = branch_type
    output_dir = os.path.join("./gradcam_heatmaps", structure_tag, mode_name, run_id)
    os.makedirs(output_dir, exist_ok=True)

    if best_ckpt:
        try:
            model = RockModel.load_from_checkpoint(best_ckpt, branch_type=branch_type, fusion_type=fusion_type)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            dataset = RockDataset(data_dir, get_transforms(), ann_file,
                                  aux_dir=aux_dir if "edge" in branch_type else None)
            psd_labels = ['D20', 'D40', 'D60', 'D80', 'max']
            num_images = min(20, len(dataset))

            for idx in range(num_images):
                sample = dataset[idx]
                img = sample['image'].to(model.device)
                img_name = sample['img_name']
                base_name = os.path.splitext(img_name)[0]
                aux_map = sample.get('aux_map')
                if aux_map is not None:
                    aux_map = aux_map.to(model.device)

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
                img_denorm = img * std + mean
                img_np = np.clip(img_denorm.permute(1, 2, 0).cpu().numpy(), 0, 1)
                img_uint8 = (img_np * 255).astype(np.uint8)

                if fusion_type in ["concat", "add", "gated"]:
                    for target_idx, label in enumerate(psd_labels):
                        try:
                            heatmap = model.get_gradcam(img, aux_map=aux_map, target_idx=target_idx)
                            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            overlay = cv2.addWeighted(img_uint8, 0.5, heatmap_colored, 0.5, 0)
                            out_path = os.path.join(output_dir, f"{base_name}_{label}_{fusion_type}.png")
                            cv2.imwrite(out_path, overlay)
                            logging.info(f"Saved Grad-CAM heatmap to {out_path}")
                        except Exception as e:
                            logging.error(f"Grad-CAM failed: {img_name}, PSD {label}, {str(e)}")
                elif fusion_type == "cross_attn":
                    try:
                        vis_out = model.visualize_fusion(img, aux_map=aux_map, save_dir=output_dir, base_name=base_name)
                        logging.info(f"Saved Cross-Attention heatmap to {vis_out['path']}")
                    except Exception as e:
                        logging.error(f"Cross-Attention visualization failed: {img_name}, {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load best checkpoint and generate visualizations: {e}")

if __name__ == "__main__":
    main()
