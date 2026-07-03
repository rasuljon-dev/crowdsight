"""
CrowdSight inference engine.

Loads a trained model checkpoint and runs crowd density estimation on images.
Defaults to CSRNet (CVPR 2018) with automatic CUDA → MPS → CPU device detection.

Supported models (auto-detected from checkpoint metadata):
  csrnet  — CSRNet, output stride 8
  mac_cnn — MAC-CNN (original baseline), output stride 4
"""

import base64
import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from model import get_model

# ImageNet normalisation constants
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _density_to_heatmap_b64(density: np.ndarray) -> str:
    """Convert a 2-D density map to a base64-encoded PNG heatmap."""
    mn, mx = density.min(), density.max()
    if mx > mn:
        normed = (density - mn) / (mx - mn)
    else:
        normed = np.zeros_like(density)

    if HAS_MPL:
        rgba = (cm.jet(normed) * 255).astype(np.uint8)
        img = Image.fromarray(rgba, mode="RGBA")
    else:
        gray = (normed * 255).astype(np.uint8)
        img = Image.fromarray(gray, mode="L")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class CrowdInferenceEngine:
    """
    Thread-safe inference engine for CrowdSight crowd density estimation.

    Parameters
    ----------
    weights_path : path to a best.pth checkpoint (optional — random weights if omitted)
    model_name   : 'csrnet' | 'mac_cnn' — overridden by checkpoint metadata if present

    Usage
    -----
    engine = CrowdInferenceEngine(weights_path="checkpoints/best.pth")
    result = engine.analyze(pil_image)
    print(result["count"], result["inference_time_ms"])
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        model_name: str = "csrnet",
    ):
        self.device = _get_device()

        # If checkpoint exists, read model_name from its metadata
        resolved_model = model_name
        state_dict = None
        if weights_path and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location="cpu")
            resolved_model = ckpt.get("model_name", model_name)
            state_dict = ckpt.get("model_state_dict", ckpt)

        self.model = get_model(resolved_model).to(self.device)
        self.model.eval()

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            self._weights_loaded = True
        else:
            self._weights_loaded = False   # random weights — API still runs

        # Output stride controls how much the model downsamples spatially
        self._output_stride = getattr(self.model, "output_stride", 8)

        self._mean = _MEAN.to(self.device)
        self._std  = _STD.to(self.device)

    @property
    def device_name(self) -> str:
        return str(self.device)

    @property
    def weights_loaded(self) -> bool:
        return self._weights_loaded

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image → normalised float tensor, pad to multiple of output_stride."""
        img = image.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)
        t = (t - self._mean) / self._std

        # Pad so H and W are divisible by output_stride
        s = self._output_stride
        _, h, w = t.shape
        ph = (s - h % s) % s
        pw = (s - w % s) % s
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph))

        return t.unsqueeze(0)   # (1, 3, H', W')

    @torch.inference_mode()
    def analyze(self, image: Image.Image) -> dict:
        """
        Run crowd density estimation on a PIL image.

        Returns
        -------
        {
            "count": float,             estimated crowd count
            "density_map": str,         base64-encoded PNG heatmap (jet colormap)
            "inference_time_ms": float
        }
        """
        t0 = time.perf_counter()

        tensor  = self._preprocess(image)
        density = self.model(tensor)          # (1, 1, H/stride, W/stride)

        # Upsample density map to original image size for visualisation
        h_orig, w_orig = image.size[1], image.size[0]
        density_up = F.interpolate(
            density, size=(h_orig, w_orig), mode="bilinear", align_corners=False
        )

        count      = max(0.0, float(density.sum().item()))
        density_np = density_up.squeeze().cpu().numpy()
        heatmap_b64 = _density_to_heatmap_b64(density_np)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "count": round(count, 2),
            "density_map": heatmap_b64,
            "inference_time_ms": round(elapsed_ms, 1),
        }
