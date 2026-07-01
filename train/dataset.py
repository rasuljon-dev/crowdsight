"""
CrowdSight dataset loaders.

Supports:
  - ShanghaiTech Part A and Part B
  - NWPU-Crowd (h5 format)

GT density maps are generated on-the-fly from head annotations using a
fixed-sigma Gaussian kernel (adaptive sigma coming in v2).

Directory structure expected:
    ShanghaiTech/
        part_A/
            train_data/  images/  ground-truth/
            test_data/   images/  ground-truth/
        part_B/
            train_data/  images/  ground-truth/
            test_data/   images/  ground-truth/
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ─── Density map generation ───────────────────────────────────────────────────

def points_to_density(
    points: np.ndarray,
    height: int,
    width: int,
    sigma: float = 15.0,
) -> np.ndarray:
    """
    Convert head annotation points to a Gaussian density map.

    Parameters
    ----------
    points : (N, 2) array of (x, y) head coordinates
    height, width : output map size (same as image)
    sigma : Gaussian sigma in pixels

    Returns
    -------
    density : (H, W) float32 array where density.sum() ≈ N (head count)
    """
    density = np.zeros((height, width), dtype=np.float32)
    if len(points) == 0:
        return density

    for x, y in points:
        ix, iy = int(min(round(x), width - 1)), int(min(round(y), height - 1))
        if 0 <= ix < width and 0 <= iy < height:
            density[iy, ix] += 1.0

    density = gaussian_filter(density, sigma=sigma)
    return density


def _downsample_density(
    density: np.ndarray, output_stride: int
) -> torch.Tensor:
    """
    Downsample a density map to model output resolution while preserving count.

    density.sum() == output.sum()  (count-preserving pooling)
    """
    t = torch.from_numpy(density).unsqueeze(0)  # (1, H, W)
    s = output_stride
    # avg_pool divides by s², multiply back to preserve sum
    t = torch.nn.functional.avg_pool2d(t, kernel_size=s, stride=s) * (s * s)
    return t  # (1, H/s, W/s)


# ─── ShanghaiTech ─────────────────────────────────────────────────────────────

def _load_shanghaitech_annotation(mat_path: str) -> np.ndarray:
    """Return (N, 2) array of (x, y) head positions from a .mat file."""
    mat = sio.loadmat(mat_path)
    # ShanghaiTech stores annotations under 'image_info' > 'location'
    points = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
    return points   # shape (N, 2)


class ShanghaiTechDataset(Dataset):
    """
    ShanghaiTech crowd counting dataset (Part A or Part B).

    Args
    ----
    root          : path to part_A or part_B directory
    split         : 'train' or 'test'
    sigma         : Gaussian sigma for density map generation
    crop_size     : square crop size for training augmentation (None = full image)
    augment       : apply random flip / scale / crop (training only)
    output_stride : spatial downsampling of density map (4 for MAC-CNN, 8 for CSRNet)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        sigma: float = 15.0,
        crop_size: Optional[int] = 256,
        augment: bool = True,
        min_scale: float = 0.7,
        max_scale: float = 1.3,
        output_stride: int = 8,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.sigma = sigma
        self.crop_size = crop_size
        self.augment = augment and (split == "train")
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.output_stride = output_stride

        data_dir = self.root / f"{split}_data"
        img_dir = data_dir / "images"
        gt_dir  = data_dir / "ground-truth"

        self.img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        self.gt_paths  = [
            gt_dir / f"GT_{p.stem}.mat" for p in self.img_paths
        ]

        assert len(self.img_paths) > 0, f"No images found in {img_dir}"

        # ImageNet normalisation
        self._mean = torch.tensor([0.485, 0.456, 0.406])
        self._std  = torch.tensor([0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        points = _load_shanghaitech_annotation(str(self.gt_paths[idx]))

        w, h = img.size
        density = points_to_density(points, h, w, sigma=self.sigma)

        if self.augment:
            img, density = self._augment(img, density)

        # Convert image to tensor and normalise
        img_t = TF.to_tensor(img)                         # (3, H, W) in [0,1]
        img_t = TF.normalize(img_t, self._mean, self._std)

        # Density map: downsample to model output resolution, preserve count
        density_t = _downsample_density(density, self.output_stride)

        return img_t, density_t

    def _augment(
        self, img: Image.Image, density: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        w, h = img.size

        # Random scale
        scale = random.uniform(self.min_scale, self.max_scale)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        density = np.array(
            Image.fromarray(density).resize((new_w, new_h), Image.BILINEAR)
        ) * (scale ** 2)

        # Random crop
        if self.crop_size and new_w >= self.crop_size and new_h >= self.crop_size:
            x0 = random.randint(0, new_w - self.crop_size)
            y0 = random.randint(0, new_h - self.crop_size)
            img = img.crop((x0, y0, x0 + self.crop_size, y0 + self.crop_size))
            density = density[y0:y0 + self.crop_size, x0:x0 + self.crop_size]

        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            density = density[:, ::-1].copy()

        return img, density


# ─── NWPU-Crowd (h5 format) ───────────────────────────────────────────────────

class NWPUDataset(Dataset):
    """
    NWPU-Crowd dataset loader (h5 pre-processed density maps).

    Expected structure:
        nwpu/
            train/  *.h5
            val/    *.h5
    Each h5 file contains:
        image   : (H, W, 3) uint8
        density : (H, W)    float32
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        augment: bool = True,
        output_stride: int = 8,
    ):
        super().__init__()
        self.root = Path(root) / split
        self.augment = augment and (split == "train")
        self.output_stride = output_stride
        self.files = sorted(self.root.glob("*.h5"))
        assert len(self.files) > 0, f"No .h5 files found in {self.root}"

        self._mean = torch.tensor([0.485, 0.456, 0.406])
        self._std  = torch.tensor([0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.files[idx], "r") as f:
            img     = Image.fromarray(f["image"][:])
            density = f["density"][:].astype(np.float32)

        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)
            density = density[:, ::-1].copy()

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, self._mean, self._std)

        density_t = _downsample_density(density, self.output_stride)

        return img_t, density_t


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_dataset(name: str, root: str, split: str, **kwargs) -> Dataset:
    """
    Factory function — returns the correct dataset for `name`.

    name: 'shanghaitech_a' | 'shanghaitech_b' | 'nwpu'

    Extra kwargs (sigma, crop_size, augment, min_scale, max_scale, output_stride)
    are forwarded to the dataset constructor.
    """
    if name == "shanghaitech_a":
        return ShanghaiTechDataset(
            root=os.path.join(root, "part_A"), split=split, **kwargs
        )
    elif name == "shanghaitech_b":
        return ShanghaiTechDataset(
            root=os.path.join(root, "part_B"), split=split, **kwargs
        )
    elif name == "nwpu":
        return NWPUDataset(root=root, split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose: shanghaitech_a, shanghaitech_b, nwpu")
