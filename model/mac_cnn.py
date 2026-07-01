"""
Mixed Attention-Based Multi-Column CNN (MAC-CNN) for Crowd Counting.

Based on: "Improving Crowd Counting Efficiency Using Spatial Attention-Based
Multi-Column CNN", R. Khalimjanov, J. Gwak, M. Jeon, KINGPC 2024.

Architecture:
  - Three parallel CNN columns with different receptive fields (dilations 1, 2, 3)
  - Per-column Channel Attention (Squeeze-and-Excitation style)
  - Spatial Attention gate applied before feature fusion
  - Density map decoder: sum(density_map) ≈ crowd count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Channel Attention ────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + mx)


# ─── Spatial Attention ────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """CBAM-style spatial attention gate."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        gate = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


# ─── Single CNN Column ────────────────────────────────────────────────────────

class CNNColumn(nn.Module):
    """
    One column of the multi-column CNN.
    Uses a fixed dilation rate to capture a specific scale of crowd density.
    """

    def __init__(self, dilation: int = 1, out_channels: int = 128):
        super().__init__()
        pad = dilation
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 9, padding=4 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 7, padding=3 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, out_channels, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.channel_attention(feat)


# ─── MAC-CNN ─────────────────────────────────────────────────────────────────

class MACCNN(nn.Module):
    """
    Mixed Attention-Based Multi-Column CNN.

    Three columns with dilations [1, 2, 3] capture fine, medium, and coarse
    crowd structure. Channel attention refines each column independently.
    Spatial attention is applied after concatenation before the density decoder.

    Output shape: (B, 1, H/4, W/4)  — upsampled to input size by inference engine.
    sum(output) ≈ crowd count (trained with Gaussian-blurred GT density maps).
    """

    COLUMN_CHANNELS = 128

    def __init__(self, dilations: tuple = (1, 2, 3)):
        super().__init__()
        self.columns = nn.ModuleList([
            CNNColumn(dilation=d, out_channels=self.COLUMN_CHANNELS)
            for d in dilations
        ])
        fused = self.COLUMN_CHANNELS * len(dilations)
        self.spatial_attention = SpatialAttention()
        self.decoder = nn.Sequential(
            nn.Conv2d(fused, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),           # density map is non-negative
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run all columns in parallel
        col_feats = [col(x) for col in self.columns]
        # Concatenate along channel dim — all columns produce same H/W due to shared pooling
        fused = torch.cat(col_feats, dim=1)
        # Apply spatial attention gate
        fused = self.spatial_attention(fused)
        # Decode to density map
        return self.decoder(fused)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MACCNN()
    print(f"MAC-CNN parameters: {model.num_params:,}")
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Count estimate (random weights): {out.sum().item():.2f}")
