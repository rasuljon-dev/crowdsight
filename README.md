# CrowdSight 🧠👥

> **Real-time crowd density estimation — CSRNet · CLIP-EBC · DM-Count loss**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/rasuljon-dev/crowdsight/actions/workflows/ci.yml/badge.svg)](https://github.com/rasuljon-dev/crowdsight/actions/workflows/ci.yml)

CrowdSight turns any image into a **crowd density heatmap** and an **estimated head count** — in milliseconds.
Built on peer-reviewed SOTA research (Awesome-Crowd-Counting leaderboard), packaged as a production-ready REST API.

---

## ✨ Features

- **Three model tiers**: MAC-CNN (thesis baseline) → CSRNet (CVPR 2018) → CLIP-EBC (ICME 2025)
- **DM-Count loss** — OT + TV regularisation for ~13% lower MAE (NeurIPS 2020)
- **One-command deployment** via Docker Compose
- **REST API** with interactive Swagger docs (`/docs`)
- **Auto device detection** — CUDA → MPS → CPU fallback
- **Optional MLflow** experiment tracking
- **Supports ShanghaiTech, NWPU-Crowd, UCF-QNRF** datasets out of the box
- **MAC-CNN baseline** still included for ablation comparison

---

## 🏗 Architecture

### CSRNet (default)

```
Input Image  (H × W)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  VGG-16 Frontend  (features[0:23], ImageNet pretrained)  │
│  conv1_1 → conv1_2 → pool1                          │
│  conv2_1 → conv2_2 → pool2                          │
│  conv3_1 → conv3_2 → conv3_3 → pool3                │
│  conv4_1 → conv4_2 → conv4_3                        │
│  Output: (H/8 × W/8 × 512)                          │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Dilated Backend  (dilation=2, no spatial loss)     │
│  512→512→512→256→128→64→1                           │
└─────────────────────────────────────────────────────┘
    │
    ▼
Density Map  (H/8 × W/8)
crowd_count = density_map.sum()
```

**Why CSRNet?** The pretrained VGG-16 frontend extracts rich multi-scale features.
Dilated convolutions in the backend preserve spatial resolution while expanding the
receptive field — critical for handling scale variation in crowd images.

### CLIP-EBC (--model clip_ebc)

```
Input Image  (H × W)
    │
    ▼  divide into block_size × block_size px blocks
    │
    ├── Block₁ ─► resize 224×224 ─► CLIP ViT-B/16 ─► 768-dim CLS ─► head → E[count₁]
    ├── Block₂ ─► resize 224×224 ─► CLIP ViT-B/16 ─► 768-dim CLS ─► head → E[count₂]
    ┆
    └── Blockₙ ─► resize 224×224 ─► CLIP ViT-B/16 ─► 768-dim CLS ─► head → E[countₙ]
                                         ↓
                        Spatial count map  (H//block_size × W//block_size)
                        crowd_count = count_map.sum()
```

- Only ~200 K parameters are trained (the classification head)
- 86 M-param CLIP backbone stays frozen → fast fine-tuning, works with small datasets
- Blockwise classification into integer bins removes label-boundary ambiguity

### DM-Count Loss (optional, --loss dmcount)

```
L = 0.1 × L_OT  +  0.01 × L_TV  +  1.0 × L_count

L_OT    = normalized OT distance between predicted & GT density distributions
L_TV    = total variation regularisation (smooth density maps)
L_count = L1 on integrated count
```

---

## 🚀 Quick Start

### Local (no Docker)

```bash
git clone https://github.com/rasuljon-dev/crowdsight.git
cd crowdsight

pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive API.

### Docker

```bash
docker compose up --build
```

With MLflow tracking:
```bash
docker compose --profile mlflow up
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

---

## 📡 API Reference

### `GET /health`

```json
{
  "status": "ok",
  "device": "cuda",
  "weights_loaded": true,
  "version": "0.2.0"
}
```

### `POST /analyze`

Upload a JPEG or PNG image and receive:

| Field | Type | Description |
|---|---|---|
| `count` | float | Estimated crowd count |
| `density_map` | string | Base64-encoded PNG heatmap (jet colormap) |
| `inference_time_ms` | float | Wall-clock inference time |

**Python client example:**

```python
import requests, base64
from PIL import Image
import io

with open("crowd.jpg", "rb") as f:
    resp = requests.post("http://localhost:8000/analyze", files={"image": f})

data = resp.json()
print(f"Count: {data['count']}")

# Decode and save heatmap
img = Image.open(io.BytesIO(base64.b64decode(data["density_map"])))
img.save("heatmap.png")
```

**cURL:**
```bash
curl -X POST http://localhost:8000/analyze \
     -F "image=@crowd.jpg" | python -m json.tool
```

---

## 📦 Project Structure

```
crowdsight/
├── model/
│   ├── __init__.py         # model registry (get_model, get_loss)
│   ├── csrnet.py           # CSRNet — VGG-16 + dilated convolutions (CVPR 2018)
│   ├── clip_ebc.py         # CLIP-EBC — frozen CLIP ViT-B/16 + count head (ICME 2025)
│   ├── mac_cnn.py          # MAC-CNN — custom multi-column + attention (KINGPC 2024)
│   └── losses.py           # MSELoss + DMCountLoss (NeurIPS 2020)
├── api/
│   ├── __init__.py
│   ├── inference.py        # CrowdInferenceEngine (model-agnostic)
│   └── main.py             # FastAPI app
├── dashboard/
│   └── app.py              # Streamlit dashboard (image upload + live webcam)
├── train/
│   ├── config.py           # TrainConfig dataclass
│   ├── dataset.py          # ShanghaiTech / NWPU loaders
│   ├── metrics.py          # MAE / MSE / RMSE
│   └── train.py            # Training loop with MLflow
├── dashboard/
│   └── app.py              # Streamlit dashboard (image upload + live webcam)
├── release/
│   ├── export_weights.py   # Strip optimizer state → lean release .pth
│   └── MODEL_CARD.md       # Download links, usage, training details
├── results/                # Benchmark JSON output (generated)
├── checkpoints/            # Trained weights (best.pth per model)
├── benchmark.py            # Evaluate checkpoints → Markdown table + JSON
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🗂 Supported Datasets

| Dataset | Scenes | Images | Max Count |
|---|---|---|---|
| ShanghaiTech Part A | Congested | 482 | 3,139 |
| ShanghaiTech Part B | Sparse | 716 | 578 |
| NWPU-Crowd | Mixed | 5,109 | 20,033 |
| UCF-QNRF | Diverse | 1,535 | 12,865 |

---

## 🏋️ Training

### 1. Download ShanghaiTech

```bash
# From: https://github.com/desenzhou/ShanghaiTechDataset
# Place in: data/ShanghaiTech/part_A/ and part_B/
```

### 2. Train

```bash
# CSRNet + MSE (fast baseline — recommended first run)
python -m train.train \
    --model csrnet \
    --loss mse \
    --dataset shanghaitech_a \
    --data_root data/ShanghaiTech \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-5

# CSRNet + DM-Count (best accuracy)
python -m train.train \
    --model csrnet \
    --loss dmcount \
    --dataset shanghaitech_a \
    --data_root data/ShanghaiTech

# CLIP-EBC (VLM backbone — only ~200K params trained, fast to fine-tune)
python -m train.train \
    --model clip_ebc \
    --loss mse \
    --dataset shanghaitech_a \
    --data_root data/ShanghaiTech \
    --block_size 64 \
    --num_bins 21 \
    --lr 1e-4   # higher LR ok — only the head is trained

# Original MAC-CNN baseline (ablation)
python -m train.train \
    --model mac_cnn \
    --loss mse \
    --dataset shanghaitech_a \
    --data_root data/ShanghaiTech

# Resume from checkpoint
python -m train.train --resume checkpoints/epoch_100.pth

# With MLflow tracking
python -m train.train \
    --model csrnet --loss dmcount \
    --mlflow_uri http://localhost:5000 \
    --run_name "csrnet-dmcount-v1"
```

### 3. Benchmark results (ShanghaiTech Part A)

| Method | Venue | MAE ↓ | RMSE ↓ | Backbone | In repo |
|---|---|---|---|---|---|
| MCNN | CVPR 2016 | 110.2 | 173.2 | custom | — |
| MAC-CNN (our thesis baseline) | KINGPC 2024 | ~80 | ~130 | custom | ✓ |
| CSRNet | CVPR 2018 | 68.2 | 115.0 | VGG-16 | ✓ |
| BL (Bayesian Loss) | ICCV 2019 | 62.8 | 101.8 | VGG-19 | — |
| CSRNet + DM-Count loss | NeurIPS 2020 | 59.7 | 95.7 | VGG-16 | ✓ |
| **CLIP-EBC** | **ICME 2025** | **~56** | **~90** | **CLIP ViT-B/16** | **✓** |
| STEERER | ICCV 2023 | 56.1 | 90.3 | HRNet | — |

> CrowdSight now spans three generations: custom CNN → pretrained CNN → vision-language model.
> CLIP-EBC trains only ~200 K parameters (the classification head) while the 86 M-param
> CLIP backbone stays frozen — making it the fastest to fine-tune despite the best accuracy.

---

## 🖥 Streamlit Dashboard

An interactive dashboard for image upload and live webcam crowd counting.

### Install dashboard dependencies

```bash
pip install streamlit>=1.32.0 opencv-python-headless>=4.9.0
```

### Launch

```bash
cd crowdsight
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### Features

- **Image upload tab** — drop any JPEG/PNG, get count + heatmap overlay + download button
- **Live webcam tab** — toggle streaming, choose camera index, tune FPS (1–30), live count chart
- **Sidebar** — switch model (csrnet / clip_ebc / mac_cnn), set checkpoint path, tune overlay opacity
- **Status bar** — shows active model, device (CUDA/MPS/CPU), weights status, output stride

### Dashboard layout

```
┌─────────────────────────────────────────────────────┐
│ Sidebar          │ Status bar (model / device / …)  │
│ · Model picker   ├──────────────────────────────────┤
│ · Checkpoint     │ Tabs: [📁 Image Upload] [📷 Live] │
│ · FPS            │                                  │
│ · Opacity        │  Original │ Heatmap │ Overlay    │
│                  │                                  │
│                  │  Count: 247  |  12.3 ms          │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark & Release

### Run benchmarks

```bash
python benchmark.py \
    --checkpoint checkpoints/csrnet_best.pth \
    --checkpoint checkpoints/clip_ebc_best.pth \
    --checkpoint checkpoints/mac_cnn_best.pth \
    --dataset shanghaitech_a \
    --data_root data/ShanghaiTech \
    --output results/sha_a.json
```

Prints a Markdown table of MAE / RMSE / inference speed per model.

### Export weights for release

```bash
# Single checkpoint → lean release file (no optimizer state)
python release/export_weights.py \
    --input  checkpoints/csrnet_best.pth \
    --output release/csrnet_sha_a.pth

# All checkpoints at once
python release/export_weights.py \
    --batch \
    --input_dir  checkpoints/ \
    --output_dir release/weights/

# Attach benchmark metrics and push to HuggingFace Hub
python release/export_weights.py \
    --input   checkpoints/csrnet_best.pth \
    --output  release/csrnet_sha_a.pth \
    --metrics results/sha_a.json \
    --hub     YOUR_HF_USERNAME/crowdsight
```

See [`release/MODEL_CARD.md`](release/MODEL_CARD.md) for download links and usage.

---

## 🧪 Tests & CI

### Run tests locally

```bash
pip install pytest pytest-cov
pytest tests/ -v -k "not clip_ebc"   # fast, CPU-only
pytest tests/ -v                      # full suite (requires CLIP weights cached)
```

### GitHub Actions

`.github/workflows/ci.yml` runs on every push/PR:

1. **Lint** — `ruff` + `ast.parse` syntax check on all Python files
2. **Tests** — pytest on Python 3.10, 3.11, 3.12 (CPU-only PyTorch, no GPU required)
3. **Docker build** — verifies the image builds cleanly
4. **Release** — manual trigger only; exports weights + pushes to HuggingFace Hub

To enable HuggingFace Hub publishing, add `HF_TOKEN` to GitHub repo secrets and set `HF_REPO_ID` as a repo variable.

---

## 🗺 Roadmap

- [x] Weekend 1 — MAC-CNN model + FastAPI + Docker
- [x] Weekend 2 — Training pipeline + ShanghaiTech + MLflow
- [x] Weekend 3 — Model upgrade: CSRNet (CVPR 2018) + DM-Count loss (NeurIPS 2020)
- [x] Weekend 4 — Streamlit dashboard + live webcam demo
- [x] Weekend 5 — Benchmark script + pretrained weights export + model card

---

## 📄 Citation

CrowdSight builds on:

```bibtex
@inproceedings{li2018csrnet,
  title     = {CSRNet: Dilated Convolutional Neural Networks for Understanding
               the Highly Congested Scenes},
  author    = {Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle = {CVPR},
  year      = {2018}
}

@inproceedings{wang2020dmcount,
  title     = {Distribution Matching for Crowd Counting},
  author    = {Wang, Boyu and Liu, Huidong and Samaras, Dimitris and Nguyen, Minh Hoai},
  booktitle = {NeurIPS},
  year      = {2020}
}

@inproceedings{shi2025clipebc,
  title     = {CLIP-EBC: CLIP Can Count Accurately through Enhanced
               Blockwise Classification},
  author    = {Shi, Yiming and Lu, Xueting and Xue, Jing-Hao and Ma, Hui},
  booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2025}
}

@inproceedings{khalimjanov2024crowdcounting,
  title     = {Improving Crowd Counting Efficiency Using Spatial
               Attention-Based Multi-Column CNN},
  author    = {Khalimjanov, Rasuljon and Gwak, Jeonghwan and Jeon, Moongu},
  booktitle = {Korea Next-Generation Computing Society Conference (KINGPC)},
  pages     = {259--262},
  year      = {2024}
}
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
