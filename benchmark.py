"""
CrowdSight benchmark script
============================
Evaluates one or more model checkpoints on ShanghaiTech Part A/B (or NWPU)
and prints a Markdown results table.

Usage
-----
# Evaluate a single checkpoint
python benchmark.py --checkpoint checkpoints/csrnet_best.pth \
                    --dataset shanghaitech_a \
                    --data_root data/ShanghaiTech

# Compare all three models at once
python benchmark.py \
    --checkpoint checkpoints/csrnet_best.pth \
    --checkpoint checkpoints/clip_ebc_best.pth \
    --checkpoint checkpoints/mac_cnn_best.pth \
    --dataset shanghaitech_a \
    --data_root data/ShanghaiTech

# Override model name (if checkpoint has no metadata)
python benchmark.py --checkpoint my.pth --model csrnet --dataset shanghaitech_b

# Save results to JSON
python benchmark.py --checkpoint checkpoints/csrnet_best.pth \
                    --output results/sha_part_a.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import get_model                         # noqa: E402
from train.dataset import get_dataset               # noqa: E402
from train.metrics import CrowdMetrics              # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_checkpoint(
    ckpt_path: str,
    model_name_override: str | None,
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sigma: float = 15.0,
) -> dict:
    """
    Load a checkpoint and evaluate it on the given test split.
    Returns a dict with keys: model_name, ckpt_path, MAE, RMSE, MSE,
    num_params, inference_ms_per_image.
    """
    ckpt_path = str(ckpt_path)
    print(f"\n{'='*60}")
    print(f"Checkpoint : {ckpt_path}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = model_name_override or ckpt.get("model_name", "csrnet")
    print(f"Model      : {model_name}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = get_model(model_name).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    output_stride = getattr(model, "output_stride", 8)
    num_params    = getattr(model, "num_params", sum(p.numel() for p in model.parameters()))
    print(f"Output stride : {output_stride}")
    print(f"Params (train): {num_params:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = get_dataset(
        dataset_name, data_root, split="test",
        crop_size=None, augment=False,
        sigma=sigma, output_stride=output_stride,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Test images: {len(ds)}")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    metrics   = CrowdMetrics()
    total_ms  = 0.0
    img_count = 0

    for imgs, densities in loader:
        imgs      = imgs.to(device, non_blocking=True)
        densities = densities.to(device, non_blocking=True)

        # Pad to multiple of output_stride
        s = output_stride
        _, _, h, w = imgs.shape
        ph = (s - h % s) % s
        pw = (s - w % s) % s
        if ph or pw:
            imgs      = F.pad(imgs, (0, pw, 0, ph))
            densities = F.pad(densities, (0, pw, 0, ph))

        t0    = time.perf_counter()
        preds = model(imgs)
        total_ms  += (time.perf_counter() - t0) * 1000
        img_count += imgs.size(0)

        metrics.update(preds, densities)

    result = metrics.compute()
    ms_per_img = total_ms / max(img_count, 1)

    print(f"MAE  : {result['MAE']:.2f}")
    print(f"RMSE : {result['RMSE']:.2f}")
    print(f"Inf  : {ms_per_img:.1f} ms/image  ({device})")

    return {
        "model_name":          model_name,
        "checkpoint":          ckpt_path,
        "dataset":             dataset_name,
        "MAE":                 round(result["MAE"], 2),
        "RMSE":                round(result["RMSE"], 2),
        "MSE":                 round(result["MSE"], 2),
        "num_trainable_params": num_params,
        "inference_ms_per_image": round(ms_per_img, 1),
        "device":              str(device),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty-print Markdown table
# ─────────────────────────────────────────────────────────────────────────────

def print_markdown_table(results: list[dict]) -> None:
    if not results:
        return

    header = (
        "| Model | Dataset | MAE ↓ | RMSE ↓ | "
        "Trainable params | Inf (ms/img) |"
    )
    sep = "|---|---|---|---|---|---|"
    print("\n" + header)
    print(sep)
    for r in results:
        params = f"{r['num_trainable_params'] / 1e6:.1f} M"
        print(
            f"| {r['model_name']} "
            f"| {r['dataset']} "
            f"| {r['MAE']:.2f} "
            f"| {r['RMSE']:.2f} "
            f"| {params} "
            f"| {r['inference_ms_per_image']:.1f} |"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrowdSight benchmark")
    p.add_argument(
        "--checkpoint", "-c", action="append", dest="checkpoints",
        required=True,
        help="Path to checkpoint (.pth). Repeat for multiple models.",
    )
    p.add_argument(
        "--model", default=None,
        help="Override model name (useful if checkpoint has no metadata).",
    )
    p.add_argument("--dataset",     default="shanghaitech_a",
                   choices=["shanghaitech_a", "shanghaitech_b", "nwpu"])
    p.add_argument("--data_root",   default="data/ShanghaiTech")
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sigma",       type=float, default=15.0)
    p.add_argument("--output",      default=None,
                   help="Save results as JSON (e.g. results/sha_a.json).")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    print(f"Device: {device}")

    results = []
    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            print(f"[WARN] Checkpoint not found: {ckpt} — skipping")
            continue
        r = evaluate_checkpoint(
            ckpt_path=ckpt,
            model_name_override=args.model,
            dataset_name=args.dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            sigma=args.sigma,
        )
        results.append(r)

    print_markdown_table(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
