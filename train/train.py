"""
CrowdSight training script.

Usage
-----
# CSRNet + MSE  (fast baseline)
python -m train.train --model csrnet --loss mse --dataset shanghaitech_a --data_root data/ShanghaiTech

# CSRNet + DM-Count  (best CNN-based accuracy)
python -m train.train --model csrnet --loss dmcount --dataset shanghaitech_a --data_root data/ShanghaiTech

# CLIP-EBC + MSE  (VLM backbone, only ~200K params trained)
python -m train.train --model clip_ebc --loss mse --dataset shanghaitech_a --data_root data/ShanghaiTech

# CLIP-EBC with custom block size / bins
python -m train.train --model clip_ebc --block_size 64 --num_bins 21

# Original MAC-CNN baseline (ablation)
python -m train.train --model mac_cnn --loss mse --dataset shanghaitech_a --data_root data/ShanghaiTech

# Resume from checkpoint
python -m train.train --resume checkpoints/epoch_50.pth

# All options
python -m train.train --help
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from model import get_model, get_loss
from train.config import TrainConfig
from train.dataset import get_dataset
from train.metrics import CrowdMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("crowdsight.train")


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Device ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_mae, cfg):
    torch.save({
        "epoch":               epoch,
        "model_name":          cfg.model_name,
        "loss_type":           cfg.loss_type,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_mae":            best_mae,
    }, path)
    log.info(f"Saved checkpoint → {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt        = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_mae    = ckpt.get("best_mae", float("inf"))
    log.info(f"Resumed from {path} (epoch {ckpt['epoch']}, best MAE={best_mae:.2f})")
    return start_epoch, best_mae


# ─── Train one epoch ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, cfg, epoch, is_dmcount):
    model.train()
    total_loss = 0.0

    for step, (imgs, densities) in enumerate(loader, 1):
        imgs      = imgs.to(device, non_blocking=True)
        densities = densities.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(imgs)

        if is_dmcount:
            loss, _ = criterion(preds, densities)
        else:
            loss = criterion(preds, densities)

        loss.backward()

        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        total_loss += loss.item()

        if step % cfg.log_every == 0:
            log.info(
                f"Epoch {epoch:03d} [{step:4d}/{len(loader)}] "
                f"loss={loss.item():.4f}"
            )

    return total_loss / len(loader)


# ─── Evaluate ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model, loader, criterion, device, is_dmcount):
    model.eval()
    metrics    = CrowdMetrics()
    total_loss = 0.0

    for imgs, densities in loader:
        imgs      = imgs.to(device, non_blocking=True)
        densities = densities.to(device, non_blocking=True)
        preds     = model(imgs)
        if is_dmcount:
            loss, _ = criterion(preds, densities)
        else:
            loss = criterion(preds, densities)
        total_loss += loss.item()
        metrics.update(preds, densities)

    return total_loss / len(loader), metrics.compute()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = get_device()
    log.info(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        dilations=cfg.dilations,
        block_size=cfg.block_size,
        num_bins=cfg.num_bins,
        freeze_clip=cfg.freeze_clip,
    ).to(device)

    output_stride = getattr(model, "output_stride", 8)
    total_p   = getattr(model, "total_params", model.num_params)
    log.info(
        f"Model: {cfg.model_name}  "
        f"(output_stride={output_stride}, "
        f"trainable={model.num_params:,}, total={total_p:,})"
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion  = get_loss(cfg.loss_type)
    is_dmcount = cfg.loss_type in ("dmcount", "dm_count")
    log.info(f"Loss: {cfg.loss_type}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    shared_kw = dict(sigma=cfg.gaussian_sigma, output_stride=output_stride)
    train_ds  = get_dataset(
        cfg.dataset, cfg.data_root, split="train",
        crop_size=cfg.crop_size, augment=True,
        min_scale=cfg.min_scale, max_scale=cfg.max_scale,
        **shared_kw,
    )
    test_ds   = get_dataset(
        cfg.dataset, cfg.data_root, split="test",
        crop_size=None, augment=False,
        **shared_kw,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader  = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    log.info(f"Train: {len(train_ds)}  |  Test: {len(test_ds)}")

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_mae    = float("inf")
    if cfg.resume and Path(cfg.resume).exists():
        start_epoch, best_mae = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, device
        )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── MLflow ────────────────────────────────────────────────────────────────
    if HAS_MLFLOW:
        mlflow.set_tracking_uri(cfg.mlflow_uri)
        mlflow.set_experiment(cfg.experiment_name)
        run = mlflow.start_run(run_name=cfg.run_name)
        mlflow.log_params({
            "model":        cfg.model_name,
            "loss":         cfg.loss_type,
            "dataset":      cfg.dataset,
            "epochs":       cfg.epochs,
            "batch_size":   cfg.batch_size,
            "lr":           cfg.lr,
            "block_size":   cfg.block_size,
            "num_bins":     cfg.num_bins,
            "freeze_clip":  cfg.freeze_clip,
        })
        log.info(f"MLflow run: {run.info.run_id}")
    else:
        log.warning("MLflow not installed — training without experiment tracking.")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, cfg, epoch, is_dmcount
        )
        val_loss, val_metrics = evaluate(model, test_loader, criterion, device, is_dmcount)
        scheduler.step()

        mae  = val_metrics["MAE"]
        rmse = val_metrics["RMSE"]
        lr   = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"MAE={mae:.2f}  RMSE={rmse:.2f}  lr={lr:.2e}"
        )

        if HAS_MLFLOW:
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "MAE": mae, "RMSE": rmse, "lr": lr},
                step=epoch,
            )

        if mae < best_mae:
            best_mae = mae
            save_checkpoint(
                os.path.join(cfg.checkpoint_dir, "best.pth"),
                epoch, model, optimizer, scheduler, best_mae, cfg,
            )
            log.info(f"  ★ New best MAE: {best_mae:.2f}")

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:03d}.pth"),
                epoch, model, optimizer, scheduler, best_mae, cfg,
            )

    if HAS_MLFLOW:
        mlflow.log_metric("best_MAE", best_mae)
        mlflow.end_run()

    log.info(f"Training complete. Best MAE: {best_mae:.2f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p   = argparse.ArgumentParser(description="Train CrowdSight")

    p.add_argument("--model",          default=cfg.model_name,
                   choices=["csrnet", "clip_ebc", "mac_cnn"], dest="model_name")
    p.add_argument("--loss",           default=cfg.loss_type,
                   choices=["mse", "dmcount"], dest="loss_type")
    p.add_argument("--no_pretrained",  action="store_false", dest="pretrained")
    p.add_argument("--block_size",     type=int,  default=cfg.block_size)
    p.add_argument("--num_bins",       type=int,  default=cfg.num_bins)
    p.add_argument("--no_freeze_clip", action="store_false", dest="freeze_clip")
    p.add_argument("--dataset",        default=cfg.dataset)
    p.add_argument("--data_root",      default=cfg.data_root)
    p.add_argument("--epochs",         type=int,   default=cfg.epochs)
    p.add_argument("--batch_size",     type=int,   default=cfg.batch_size)
    p.add_argument("--lr",             type=float, default=cfg.lr)
    p.add_argument("--crop_size",      type=int,   default=cfg.crop_size)
    p.add_argument("--sigma",          type=float, default=cfg.gaussian_sigma,
                   dest="gaussian_sigma")
    p.add_argument("--resume",         default=cfg.resume)
    p.add_argument("--checkpoint_dir", default=cfg.checkpoint_dir)
    p.add_argument("--mlflow_uri",     default=cfg.mlflow_uri)
    p.add_argument("--run_name",       default=cfg.run_name)
    p.add_argument("--seed",           type=int,  default=cfg.seed)
    p.add_argument("--num_workers",    type=int,  default=cfg.num_workers)

    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    main(parse_args())
