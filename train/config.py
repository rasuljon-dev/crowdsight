"""
Training configuration for CrowdSight.

Recommended combinations
-------------------------
csrnet   + mse      → MAE ~68.2  (fast)
csrnet   + dmcount  → MAE ~59.7  (best CNN-based)
clip_ebc + mse      → MAE ~60–63 (VLM backbone, fast to fine-tune)
mac_cnn  + mse      → MAE ~75–85 (original thesis baseline)

Edit values here or override via CLI args in train.py.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name:  str  = "csrnet"      # 'csrnet' | 'clip_ebc' | 'mac_cnn'
    pretrained:  bool = True           # ImageNet pretrained VGG-16 (CSRNet only)
    dilations:   tuple = (1, 2, 3)    # MAC-CNN dilation schedule

    # CLIP-EBC specific
    block_size:  int  = 64            # image block size fed to CLIP (px)
    num_bins:    int  = 21            # count classification bins per block (0…20)
    freeze_clip: bool = True          # freeze CLIP backbone (recommended)

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_type: str = "mse"            # 'mse' | 'dmcount'

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset:        str   = "shanghaitech_a"
    data_root:      str   = "data/ShanghaiTech"
    gaussian_sigma: float = 15.0
    crop_size:      int   = 256
    min_scale:      float = 0.7
    max_scale:      float = 1.3

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:       int   = 200
    batch_size:   int   = 8
    num_workers:  int   = 4
    lr:           float = 1e-5        # low LR for pretrained/frozen backbone
    weight_decay: float = 1e-4
    lr_step_size: int   = 50
    lr_gamma:     float = 0.5
    grad_clip:    float = 1.0

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str           = "checkpoints"
    resume:         Optional[str] = None
    save_every:     int           = 10

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_uri:      str           = "http://localhost:5000"
    experiment_name: str           = "crowdsight"
    run_name:        Optional[str] = None

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed:      int = 42
    log_every: int = 20
