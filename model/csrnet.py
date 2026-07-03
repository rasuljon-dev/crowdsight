"""
CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly
Congested Scenes (CVPR 2018).

Paper: https://arxiv.org/abs/1802.10062
Authors: Yuhong Li, Xiaofan Zhang, Deming Chen

Architecture
------------
Frontend : VGG-16 conv1_1 → conv4_3  (features[0:23], 3 max-pools → 1/8 resolution)
Backend  : 6 dilated convolutional layers (dilation=2, padding=2)
Output   : density map at 1/8 input resolution; density.sum() = crowd count

Benchmark (ShanghaiTech Part A)
--------------------------------
Method          MAE     RMSE
MCNN (2016)    110.2   173.2
CSRNet (2018)   68.2   115.0   ← this model
MAC-CNN (ours)  ~80    ~130
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import Optional


class CSRNet(nn.Module):
    """
    CSRNet crowd density estimator.

    Parameters
    ----------
    pretrained : bool
        Initialise VGG-16 frontend with ImageNet weights.
    load_weights : Optional[str]
        Path to a full CrowdSight checkpoint (`model_state_dict` key expected).
    """

    # VGG-16 features slice: conv1_1 … conv4_3 (no pool4)
    # Output channels: 512 | Spatial stride: 8
    _FRONTEND_END = 23

    def __init__(self, pretrained: bool = True, load_weights: Optional[str] = None):
        super().__init__()

        # ── Frontend: pretrained VGG-16 ──────────────────────────────────────
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT if pretrained else None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:self._FRONTEND_END])

        # ── Backend: dilated convolutions ────────────────────────────────────
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
        )

        # ── Output head ──────────────────────────────────────────────────────
        self.output_layer = nn.Conv2d(64, 1, 1)

        # Initialise backend + output with Gaussian weights
        self._init_backend()

        if load_weights:
            self.load_checkpoint(load_weights)

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_backend(self):
        for m in list(self.backend.modules()) + list(self.output_layer.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) normalised RGB tensor

        Returns
        -------
        density : (B, 1, H/8, W/8)  —  density.sum() ≈ crowd count
        """
        x = self.frontend(x)
        x = self.backend(x)
        return torch.relu(self.output_layer(x))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def output_stride(self) -> int:
        """Spatial downsampling factor between input and output."""
        return 8

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_checkpoint(self, path: str):
        import torch
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        self.load_state_dict(state)
