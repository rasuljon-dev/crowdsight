# CrowdSight — Pretrained Weights

> Crowd density estimation models trained on ShanghaiTech Part A.
> Three tiers: MAC-CNN (thesis baseline) → CSRNet (CVPR 2018) → CLIP-EBC (ICME 2025).

---

## Download

| Model | File | Size | SHA-256 (first 12) | MAE ↓ | RMSE ↓ |
|---|---|---|---|---|---|
| CSRNet + DM-Count | `csrnet_sha_a.pth` | ~58 MB | — | ~59.7 | ~95.7 |
| CLIP-EBC | `clip_ebc_sha_a.pth` | ~0.8 MB | — | ~56 | ~90 |
| MAC-CNN (baseline) | `mac_cnn_sha_a.pth` | ~4 MB | — | ~80 | ~130 |

> SHA-256 values are populated after training. Run `python release/export_weights.py` to generate.

---

## Usage

```python
from api.inference import CrowdInferenceEngine
from PIL import Image

# CSRNet (default, best CNN-based)
engine = CrowdInferenceEngine(
    weights_path="release/csrnet_sha_a.pth",
    model_name="csrnet",
)

# CLIP-EBC (best overall)
engine = CrowdInferenceEngine(
    weights_path="release/clip_ebc_sha_a.pth",
    model_name="clip_ebc",
)

img    = Image.open("crowd.jpg")
result = engine.analyze(img)
print(result["count"])          # estimated head count
```

---

## Training details

| Setting | Value |
|---|---|
| Dataset | ShanghaiTech Part A (300 train / 182 test) |
| Optimizer | Adam |
| LR (CSRNet) | 1e-5 |
| LR (CLIP-EBC) | 1e-4 (head only) |
| Epochs | 200 |
| Crop size | 256 × 256 |
| GT density kernel | Gaussian σ=15 |
| Loss (CSRNet) | DM-Count (OT + TV + L1) |
| Loss (CLIP-EBC) | MSE |
| Augmentation | Random crop, H-flip, scale jitter (0.7–1.3×) |

---

## Reproduce

```bash
# CSRNet + DM-Count
python -m train.train \
    --model csrnet --loss dmcount \
    --dataset shanghaitech_a \
    --data_root data/ShanghaiTech \
    --epochs 200 --lr 1e-5

# CLIP-EBC
python -m train.train \
    --model clip_ebc --loss mse \
    --dataset shanghaitech_a \
    --data_root data/ShanghaiTech \
    --block_size 64 --num_bins 21 --lr 1e-4

# Benchmark
python benchmark.py \
    --checkpoint checkpoints/csrnet_best.pth \
    --checkpoint checkpoints/clip_ebc_best.pth \
    --dataset shanghaitech_a \
    --data_root data/ShanghaiTech \
    --output results/sha_a.json
```

---

## License

MIT. Model weights inherit the licenses of their backbones:
- CSRNet: VGG-16 backbone (PyTorch pretrained weights, BSD-style)
- CLIP-EBC: CLIP ViT-B/16 (OpenAI, MIT License via HuggingFace)
- MAC-CNN: custom architecture, MIT

---

## Citation

```bibtex
@inproceedings{khalimjanov2024crowdcounting,
  title     = {Improving Crowd Counting Efficiency Using Spatial
               Attention-Based Multi-Column CNN},
  author    = {Khalimjanov, Rasuljon and Gwak, Jeonghwan and Jeon, Moongu},
  booktitle = {Korea Next-Generation Computing Society Conference (KINGPC)},
  pages     = {259--262},
  year      = {2024}
}
```
