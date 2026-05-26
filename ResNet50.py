import os
import random
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

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


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================
# Seeds & Logging
# =========================================================
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


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


PSD_LABELS = ["D20", "D40", "D60", "D80", "max"]


def compute_psd_metrics(prefix: str, preds: torch.Tensor, gts: torch.Tensor) -> Dict[str, float]:
    preds = preds.float()
    gts = gts.float()

    abs_err = torch.abs(preds - gts)
    sq_err = (preds - gts) ** 2
    mae_sep = torch.mean(abs_err, dim=0)
    rmse_sep = torch.sqrt(torch.mean(sq_err, dim=0))
    mape_sep = torch.mean(abs_err / (torch.abs(gts) + 1e-8), dim=0) * 100

    mean = torch.mean(gts, dim=0, keepdim=True)
    ss_tot_sep = torch.sum((gts - mean) ** 2, dim=0)
    ss_res_sep = torch.sum(sq_err, dim=0)
    r2_sep = 1 - ss_res_sep / (ss_tot_sep + 1e-8)

    metrics = {
        f"{prefix}_psd_mae": float(torch.mean(abs_err).item()),
        f"{prefix}_psd_rmse": float(torch.sqrt(torch.mean(sq_err)).item()),
        f"{prefix}_psd_mape": float(torch.mean(abs_err / (torch.abs(gts) + 1e-8)).item() * 100),
        f"{prefix}_psd_r2": float((1 - torch.sum(ss_res_sep) / (torch.sum(ss_tot_sep) + 1e-8)).item()),
    }

    for i, label in enumerate(PSD_LABELS):
        metrics[f"{prefix}_psd_{label}_mae"] = float(mae_sep[i].item())
        metrics[f"{prefix}_psd_{label}_rmse"] = float(rmse_sep[i].item())
        metrics[f"{prefix}_psd_{label}_mape"] = float(mape_sep[i].item())
        metrics[f"{prefix}_psd_{label}_r2"] = float(r2_sep[i].item())

    violations = torch.any(preds[:, :-1] >= preds[:, 1:], dim=1).float().mean()
    metrics[f"{prefix}_rank_violation_rate"] = float(violations.item())
    return metrics


# =========================================================
# Transforms
# =========================================================
class TrainTransform:
    def __init__(self, size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = T.Compose([
            T.Resize(size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, aux=None):
        return self.transform(img), None


class EvalTransform:
    def __init__(self, size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, aux=None):
        return self.transform(img), None


def get_train_transforms():
    return TrainTransform(size=(224, 224))


def get_eval_transforms():
    return EvalTransform(size=(224, 224))


# =========================================================
# Dataset
# =========================================================
class RockDataset(Dataset):
    def __init__(self, root, transform=None, annotation_file=None):
        self.root = root
        self.transform = transform or (lambda img, aux: (T.ToTensor()(img), aux))

        self.image_paths = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.annotations = {}
        if annotation_file:
            df = pd.read_csv(annotation_file)
            print("CSV columns:", df.columns.tolist())
            for _, row in df.iterrows():
                self.annotations[row["image_name"]] = {
                    "psd": [row["D20"], row["D40"], row["D60"], row["D80"], row["max"]]
                }

        self.image_paths = [
            p for p in self.image_paths
            if os.path.basename(p) in self.annotations
        ]

        self.areas = [get_area_from_name(os.path.basename(p)) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)

        img = Image.open(img_path).convert("RGB")
        img, _ = self.transform(img, None)

        ann = self.annotations[img_name]

        return {
            "image": img,
            "psd": torch.tensor(ann["psd"], dtype=torch.float32),
            "img_name": img_name
        }


# =========================================================
# Collate
# =========================================================
def rock_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise RuntimeError("Empty batch")

    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "psd": torch.stack([b["psd"] for b in batch], dim=0),
        "img_name": [b.get("img_name", "") for b in batch]
    }


# =========================================================
# DataModule
# =========================================================
class RockDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, ann_file, batch_size=16, n_splits=3, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.num_workers = num_workers

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
            annotation_file=self.ann_file
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
            annotation_file=self.ann_file
        )
        val_base = RockDataset(
            self.data_dir,
            transform=self.eval_transform,
            annotation_file=self.ann_file
        )
        test_base = RockDataset(
            self.data_dir,
            transform=self.eval_transform,
            annotation_file=self.ann_file
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
            collate_fn=rock_collate_fn
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
            collate_fn=rock_collate_fn
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
            collate_fn=rock_collate_fn
        )


# =========================================================
# Model
# =========================================================
class ResNet50PSD(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-3, pretrained=True):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = create_model("resnet50", pretrained=pretrained, num_classes=0)
        in_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

        self.w_psd = 1.0
        self.w_rank = 0.1

        self.train_losses = []
        self.val_psd_preds, self.val_psd_gts = [], []
        self.test_psd_preds, self.test_psd_gts = [], []

    def forward(self, x):
        feat = self.backbone(x)
        psd = self.head(feat)
        return psd

    def rank_loss(self, psd_pred):
        loss = 0
        for i in range(4):
            loss += F.relu(psd_pred[:, i] - psd_pred[:, i + 1])
        return loss.mean()

    def compute_loss(self, psd_pred, psd_gt):
        l_psd = F.smooth_l1_loss(psd_pred, psd_gt)
        l_rank = self.rank_loss(psd_pred)
        return self.w_psd * l_psd + self.w_rank * l_rank

    def training_step(self, batch, batch_idx):
        img, psd_gt = batch["image"], batch["psd"]
        psd_pred = self(img)

        loss = self.compute_loss(psd_pred, psd_gt)

        self.train_losses.append(loss.detach())
        self.log("train_loss_batch", loss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img, psd_gt = batch["image"], batch["psd"]
        psd_pred = self(img)

        loss = self.compute_loss(psd_pred, psd_gt)
        self.log("val_loss", loss, prog_bar=True, logger=False, on_step=False, on_epoch=True)

        self.val_psd_preds.append(psd_pred.detach().cpu())
        self.val_psd_gts.append(psd_gt.detach().cpu())

    def test_step(self, batch, batch_idx):
        img, psd_gt = batch["image"], batch["psd"]
        psd_pred = self(img)

        loss = self.compute_loss(psd_pred, psd_gt)
        self.log("test_loss", loss, prog_bar=True, logger=False)

        self.test_psd_preds.append(psd_pred.detach().cpu())
        self.test_psd_gts.append(psd_gt.detach().cpu())

    def _compute_metrics(self, preds, gts):
        mae = torch.mean(torch.abs(preds - gts))
        rmse = torch.sqrt(F.mse_loss(preds, gts))
        mape = torch.mean(torch.abs((gts - preds) / (gts + 1e-8))) * 100

        psd_mean = torch.mean(gts, dim=0, keepdim=True)
        ss_tot = torch.sum((gts - psd_mean) ** 2, dim=0)
        ss_res = torch.sum((gts - preds) ** 2, dim=0)
        r2_sep = 1 - ss_res / (ss_tot + 1e-8)
        r2 = 1 - torch.sum(ss_res) / (torch.sum(ss_tot) + 1e-8)

        mae_sep = torch.mean(torch.abs(preds - gts), dim=0)
        rmse_sep = torch.sqrt(torch.mean((preds - gts) ** 2, dim=0))
        mape_sep = torch.mean(torch.abs((gts - preds) / (gts + 1e-8)), dim=0) * 100

        violations = sum(
            any(preds[i, j] >= preds[i, j + 1] for j in range(4))
            for i in range(preds.shape[0])
        )
        rank_violation_rate = violations / preds.shape[0]

        return mae, rmse, mape, r2, mae_sep, rmse_sep, mape_sep, r2_sep, rank_violation_rate

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_psd_preds, dim=0)
        gts = torch.cat(self.val_psd_gts, dim=0)

        mae, rmse, mape, r2, mae_sep, rmse_sep, mape_sep, r2_sep, rank_v = self._compute_metrics(preds, gts)

        labels = ["D20", "D40", "D60", "D80", "max"]

        self.log("val_psd_mae", mae, logger=False, prog_bar=True)
        self.log("val_psd_rmse", rmse, logger=False)
        self.log("val_psd_mape", mape, logger=False)
        self.log("val_psd_r2", r2, logger=False)
        self.log("val_rank_violation", rank_v, logger=False)

        for i, label in enumerate(labels):
            self.log(f"val_psd_{label}_r2", r2_sep[i], logger=False)

        self.val_psd_preds.clear()
        self.val_psd_gts.clear()

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_psd_preds, dim=0)
        gts = torch.cat(self.test_psd_gts, dim=0)

        mae, rmse, mape, r2, mae_sep, rmse_sep, mape_sep, r2_sep, rank_v = self._compute_metrics(preds, gts)

        labels = ["D20", "D40", "D60", "D80", "max"]

        self.log("test_psd_mae", mae, logger=False)
        self.log("test_psd_rmse", rmse, logger=False)
        self.log("test_psd_mape", mape, logger=False)
        self.log("test_psd_r2", r2, logger=False)
        self.log("test_rank_violation_rate", rank_v, logger=False)

        for i, label in enumerate(labels):
            self.log(f"test_psd_{label}_mae", mae_sep[i], logger=False)
            self.log(f"test_psd_{label}_rmse", rmse_sep[i], logger=False)
            self.log(f"test_psd_{label}_mape", mape_sep[i], logger=False)
            self.log(f"test_psd_{label}_r2", r2_sep[i], logger=False)

        self.test_psd_preds.clear()
        self.test_psd_gts.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# =========================================================
# Loss Curve Callback
# =========================================================
class LossCurveCallback(pl.Callback):
    def __init__(self, train_metrics_every=0):
        super().__init__()
        self.history = []
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
            vals = [l.detach().float().item() for l in pl_module.train_losses]
            train_loss_epoch = float(np.mean(vals))
        else:
            train_loss_epoch = float("nan")

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
        preds, gts = [], []

        with torch.no_grad():
            for batch in trainer.datamodule.train_dataloader():
                img = batch["image"].to(pl_module.device)
                psd_gt = batch["psd"].to(pl_module.device)
                psd_pred = pl_module(img)

                preds.append(psd_pred.cpu())
                gts.append(psd_gt.cpu())

        preds = torch.cat(preds)
        gts = torch.cat(gts)

        psd_mae, psd_rmse, psd_mape, psd_r2 = self.compute_metrics(preds, gts)

        violations = sum(any(pred[j] >= pred[j + 1] for j in range(4)) for pred in preds)
        rank_violation_rate = violations / preds.shape[0]

        self.train_metrics = {
            "train_loss": train_loss_epoch,
            "train_psd_mae": psd_mae,
            "train_psd_rmse": psd_rmse,
            "train_psd_mape": psd_mape,
            "train_psd_r2": psd_r2,
            "train_rank_violation": rank_violation_rate,
        }

    def on_validation_end(self, trainer, pl_module):
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
            f"[Epoch {record['epoch']}] "
            f"TrainLoss={record['train_loss']:.4f}, ValLoss={record['val_loss']:.4f} | "
            f"Train PSD MAE={record['train_psd_mae']:.4f}, RMSE={record['train_psd_rmse']:.4f}, R2={record['train_psd_r2']:.4f} | "
            f"Val PSD MAE={record['val_psd_mae']:.4f}, RMSE={record['val_psd_rmse']:.4f}, R2={record['val_psd_r2']:.4f}"
        )


# =========================================================
# Run CV
# =========================================================
def run_cv(
    data_dir,
    ann_file,
    batch_size,
    n_splits,
    epochs,
    num_workers,
    train_metrics_every,
    pretrained,
    progress_bar,
    base_ckpt_dir,
    base_curve_dir,
    run_id,
):
    mode_name = "resnet50_psd_leave_one_area_out"
    structure_tag = "resnet50_psd"

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
            num_workers=num_workers
        )
        dm.set_fold(fold_idx)

        model = ResNet50PSD(lr=1e-4, weight_decay=1e-3, pretrained=pretrained)

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
            enable_progress_bar=progress_bar,
            enable_model_summary=False,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            check_val_every_n_epoch=1,
            val_check_interval=1.0,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            benchmark=True,
            deterministic=False,
        )

        trainer.fit(model, datamodule=dm)

        loss_xlsx = os.path.join(curve_dir, f"fold{fold_idx + 1}_loss_curve.xlsx")
        pd.DataFrame(loss_cb.history).to_excel(loss_xlsx, index=False)
        print(f"[fold {fold_idx + 1}] loss curve saved to: {loss_xlsx}")

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
            dm.set_fold(fold_idx)
            dm.setup(stage="validate")
            val_out = trainer.validate(
                model=None,
                dataloaders=dm.val_dataloader(),
                ckpt_path=best_ckpt,
                verbose=False
            )[0]

            dm.set_fold(fold_idx)
            dm.setup(stage="test")
            test_out = trainer.test(
                model=None,
                dataloaders=dm.test_dataloader(),
                ckpt_path=best_ckpt,
                verbose=False
            )[0]

            fold_record.update({k: float(v) for k, v in val_out.items()})
            fold_record.update({k: float(v) for k, v in test_out.items()})

            model = ResNet50PSD.load_from_checkpoint(best_ckpt)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()

            dm.set_fold(fold_idx)
            dm.setup(stage="test")

            preds, gts, img_names = [], [], []

            for batch in dm.test_dataloader():
                img = batch["image"].to(device)
                psd_gt = batch["psd"]
                names = batch["img_name"]

                with torch.no_grad():
                    psd_pred = model(img)

                preds.append(psd_pred.cpu())
                gts.append(psd_gt.cpu())
                img_names.extend(names)

            preds = torch.cat(preds)
            gts = torch.cat(gts)
            fold_record.update(compute_psd_metrics("test", preds, gts))

            pred_df = pd.DataFrame(
                torch.cat([gts, preds], dim=1).numpy(),
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
            print(f"[fold {fold_idx + 1}] predictions saved to {pred_csv}")
            print(f"[fold {fold_idx + 1}] merged predictions saved to {all_pred_csv}")
            print(pred_df["image_name"].head(10).to_string(index=False))

        else:
            print(f"[fold {fold_idx + 1}] no checkpoint saved.")

        all_fold_metrics.append(fold_record)

        df = pd.DataFrame(all_fold_metrics)
        per_fold_csv = os.path.join(curve_dir, "per_fold_metrics.csv")
        per_fold_xlsx = os.path.join(curve_dir, "per_fold_metrics.xlsx")
        df.to_csv(per_fold_csv, index=False)
        df.to_excel(per_fold_xlsx, index=False)

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

        print("\n===== CV Summary So Far =====")
        for col in num_cols:
            mean_v = mean_row[col].values[0]
            std_v = std_row[col].values[0]
            print(f"{col:30s}: {mean_v:.4f} ± {std_v:.4f}")

    return None


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true", help="Disable timm pretrained weight loading.")
    parser.add_argument("--no_progress_bar", action="store_true", help="Disable Lightning progress bar.")
    parser.add_argument(
        "--train_metrics_every",
        type=int,
        default=0,
        help="Compute full train MAE/RMSE/R2 every N epochs. 0 disables the extra train-set pass.",
    )
    parser.add_argument("--data_dir", type=str, default="./PSD_dataset/images")
    parser.add_argument("--ann_file", type=str, default="./PSD_dataset/annotations.csv")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_cv(
        data_dir=args.data_dir,
        ann_file=args.ann_file,
        batch_size=args.batch_size,
        n_splits=args.splits,
        epochs=args.epochs,
        num_workers=args.num_workers,
        train_metrics_every=args.train_metrics_every,
        pretrained=not args.no_pretrained,
        progress_bar=not args.no_progress_bar,
        base_ckpt_dir="./checkpoints/resnet50_psd",
        base_curve_dir="./loss_curves/resnet50_psd",
        run_id=run_id
    )


if __name__ == "__main__":
    main()
