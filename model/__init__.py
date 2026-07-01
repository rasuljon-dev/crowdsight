"""
CrowdSight model registry.

Available models
----------------
csrnet   — CSRNet (CVPR 2018), VGG-16 + dilated convolutions, output stride 8
            MAE ~68.2 on ShanghaiTech Part A
            + DM-Count loss → MAE ~59.7  ← recommended default
clip_ebc — CLIP-EBC (ICME 2025), frozen CLIP ViT-B/16 + count head, output stride = block_size
            MAE ~56 on ShanghaiTech Part A  ← best in repo
mac_cnn  — MAC-CNN (KINGPC 2024), custom multi-column + attention, output stride 4
            MAE ~75–85 on ShanghaiTech Part A  (original thesis baseline)
"""

from .csrnet   import CSRNet
from .mac_cnn  import MACCNN
from .clip_ebc import CLIP_EBC
from .losses   import get_loss


def get_model(name: str, **kwargs):
    """
    Return an instantiated model by name.

    name    : 'csrnet' | 'clip_ebc' | 'mac_cnn'
    kwargs  : forwarded to the model constructor (unused kwargs are silently dropped).
    """
    if name == "csrnet":
        valid = {"pretrained"}
        return CSRNet(**{k: v for k, v in kwargs.items() if k in valid})

    elif name in ("mac_cnn", "maccnn"):
        valid = {"dilations"}
        return MACCNN(**{k: v for k, v in kwargs.items() if k in valid})

    elif name == "clip_ebc":
        valid = {"block_size", "num_bins", "freeze_clip"}
        return CLIP_EBC(**{k: v for k, v in kwargs.items() if k in valid})

    else:
        raise ValueError(
            f"Unknown model '{name}'. Available: csrnet, clip_ebc, mac_cnn"
        )


__all__ = ["CSRNet", "MACCNN", "CLIP_EBC", "get_model", "get_loss"]
