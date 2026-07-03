"""
CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification
(ICME 2025)

Paper : https://arxiv.org/abs/2403.09281
Code  : https://github.com/Yiming-M/CLIP-EBC

Architecture
------------
1. Divide the input image into non-overlapping blocks of `block_size` × `block_size` pixels.
2. Resize each block to 224 × 224 (CLIP ViT-B/16 native resolution).
3. Pass all blocks through a **frozen** CLIP ViT-B/16 visual encoder → 768-dim CLS token.
4. A lightweight classification head maps each CLS token to `num_bins` count bins.
5. Expected count per block = Σ (softmax(logits) · bin_index).
6. Spatial count map: (B, 1, H//block_size, W//block_size).
7. crowd_count = count_map.sum()

Key advantages over CSRNet
--------------------------
- CLIP's visual features generalise far better to unseen scenes (zero-shot transfer).
- Only ~200 K parameters are trainable (the classification head) — the 86 M-param
  CLIP backbone is frozen, so fine-tuning is fast and requires little labelled data.
- Compatible with the existing training loop, dataset loaders, and MSE / DM-Count loss.

Benchmark (ShanghaiTech Part A, MAE ↓)
---------------------------------------
CSRNet + DM-Count   59.7   ← previous best in this repo
CLIP-EBC (paper)    ~56     ← this model (using EBC cross-entropy; MSE gives ~60-63)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP_EBC(nn.Module):
    """
    CLIP-EBC crowd density estimator.

    Parameters
    ----------
    block_size  : side length (px) of each image block fed to CLIP (default 64).
                  Must divide the (padded) image dimensions evenly.
    num_bins    : number of discrete count bins per block (0 … num_bins-1).
                  Set ≥ max expected heads per block. Default 21 covers most scenes.
    freeze_clip : freeze the CLIP backbone (recommended — keeps training fast).
    """

    # CLIP ViT-B/16 native resolution and CLS-token dimension
    _CLIP_SIZE   = 224
    _CLIP_DIM    = 768

    # Normalization constants
    _CLIP_MEAN  = [0.48145466, 0.4578275,  0.40821073]
    _CLIP_STD   = [0.26862954, 0.26130258, 0.27577711]
    _INET_MEAN  = [0.485,      0.456,      0.406     ]
    _INET_STD   = [0.229,      0.224,      0.225     ]

    def __init__(
        self,
        block_size: int  = 64,
        num_bins:   int  = 21,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.block_size  = block_size
        self.num_bins    = num_bins

        # ── CLIP visual backbone ──────────────────────────────────────────────
        try:
            from transformers import CLIPVisionModel
        except ImportError as e:
            raise ImportError(
                "CLIP-EBC requires the `transformers` package. "
                "Install it with: pip install transformers"
            ) from e

        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad_(False)

        # ── Classification head ───────────────────────────────────────────────
        # Maps CLIP CLS token → per-block count distribution
        self.count_head = nn.Sequential(
            nn.LayerNorm(self._CLIP_DIM),
            nn.Linear(self._CLIP_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_bins),
        )
        self._init_head()

        # ── Normalisation buffers ─────────────────────────────────────────────
        self.register_buffer(
            "clip_mean",
            torch.tensor(self._CLIP_MEAN, dtype=torch.float).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(self._CLIP_STD, dtype=torch.float).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "inet_mean",
            torch.tensor(self._INET_MEAN, dtype=torch.float).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "inet_std",
            torch.tensor(self._INET_STD, dtype=torch.float).view(1, 3, 1, 1),
        )

        # Bin indices for expected-value computation: [0, 1, 2, …, num_bins-1]
        self.register_buffer("bin_idx", torch.arange(num_bins, dtype=torch.float))

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_head(self):
        for m in self.count_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def output_stride(self) -> int:
        """Spatial downsampling factor: output is (H/block_size, W/block_size)."""
        return self.block_size

    @property
    def num_params(self) -> int:
        """Number of *trainable* parameters (excludes frozen CLIP backbone)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        """Total parameters including frozen CLIP backbone."""
        return sum(p.numel() for p in self.parameters())

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _to_clip_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert an ImageNet-normalised tensor to CLIP normalisation.

        x: (B, 3, H, W) — ImageNet normalised
        Returns: (B, 3, H, W) — CLIP normalised
        """
        x_raw = x * self.inet_std + self.inet_mean   # [0, 1]
        return (x_raw - self.clip_mean) / self.clip_std

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  ImageNet-normalised input image(s)

        Returns
        -------
        count_map : (B, 1, H//block_size, W//block_size)
            Per-block expected head count.  count_map.sum() ≈ crowd count.
        """
        B, C, H, W = x.shape
        s = self.block_size

        # ── 1. Pad so H and W are divisible by block_size ────────────────────
        pad_h = (s - H % s) % s
        pad_w = (s - W % s) % s
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H_p, W_p = x.shape
        n_h = H_p // s   # number of blocks vertically
        n_w = W_p // s   # number of blocks horizontally
        N   = n_h * n_w  # blocks per image

        # ── 2. Re-normalise for CLIP ──────────────────────────────────────────
        x_clip = self._to_clip_norm(x)

        # ── 3. Extract non-overlapping blocks ─────────────────────────────────
        # unfold(dim, size, step) → adds a trailing dimension of size `size`
        # After two unfolds: (B, C, n_h, n_w, s, s)
        blocks = x_clip.unfold(2, s, s).unfold(3, s, s)
        blocks = blocks.permute(0, 2, 3, 1, 4, 5).contiguous()   # (B, n_h, n_w, C, s, s)
        blocks = blocks.reshape(B * N, C, s, s)                   # (B*N, C, s, s)

        # ── 4. Resize blocks to CLIP's 224 × 224 ─────────────────────────────
        if s != self._CLIP_SIZE:
            blocks = F.interpolate(
                blocks,
                size=(self._CLIP_SIZE, self._CLIP_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # ── 5. CLIP visual encoder ────────────────────────────────────────────
        # pooler_output = post-LayerNorm CLS token, shape (B*N, 768)
        clip_out  = self.clip(pixel_values=blocks)
        cls_feats = clip_out.pooler_output   # (B*N, 768)

        # ── 6. Classification head ────────────────────────────────────────────
        logits = self.count_head(cls_feats)          # (B*N, num_bins)
        probs  = torch.softmax(logits, dim=-1)       # (B*N, num_bins)
        counts = (probs * self.bin_idx).sum(dim=-1)  # (B*N,)  expected count

        # ── 7. Reshape to spatial count map ───────────────────────────────────
        count_map = counts.reshape(B, 1, n_h, n_w)  # (B, 1, n_h, n_w)

        return count_map
