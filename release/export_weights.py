"""
CrowdSight — Export pretrained weights for release
====================================================
Strips optimizer / scheduler state from training checkpoints and writes
lean inference-only files suitable for GitHub Releases or HuggingFace Hub.

Output format
-------------
{
  "model_name":   "csrnet",
  "model_state_dict": { … },          # weights only
  "crowdsight_version": "1.0.0",
  "sha256": "…",                       # hash of state_dict bytes
  "metrics": {                         # optional, from benchmark JSON
      "SHA_A_MAE": 59.7,
      "SHA_A_RMSE": 95.7
  }
}

Usage
-----
# Export single checkpoint
python release/export_weights.py \
    --input  checkpoints/best.pth \
    --output release/csrnet_sha_a.pth

# Export with benchmark metrics attached
python release/export_weights.py \
    --input   checkpoints/csrnet_best.pth \
    --output  release/csrnet_sha_a.pth \
    --metrics results/sha_a.json \
    --model   csrnet

# Export all at once (batch)
python release/export_weights.py \
    --batch \
    --input_dir  checkpoints/ \
    --output_dir release/weights/
"""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "1.0.0"

# ─────────────────────────────────────────────────────────────────────────────

def _sha256_of_state(state_dict: dict) -> str:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def export_single(
    input_path: Path,
    output_path: Path,
    model_name_override: str | None = None,
    metrics: dict | None = None,
) -> dict:
    print(f"\nLoading   : {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")

    model_name   = model_name_override or ckpt.get("model_name", "unknown")
    state_dict   = ckpt.get("model_state_dict", ckpt)
    sha          = _sha256_of_state(state_dict)
    best_mae     = ckpt.get("best_mae", None)

    payload = {
        "model_name":          model_name,
        "crowdsight_version":  VERSION,
        "sha256":              sha,
        "model_state_dict":    state_dict,
    }
    if best_mae is not None:
        payload["best_mae"] = float(best_mae)
    if metrics:
        payload["metrics"] = metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"Saved     : {output_path}  ({size_mb:.1f} MB)")
    print(f"SHA-256   : {sha[:16]}…")
    if best_mae:
        print(f"Best MAE  : {best_mae:.2f}")

    return {"model": model_name, "path": str(output_path), "sha256": sha, "size_mb": round(size_mb, 1)}


def batch_export(input_dir: Path, output_dir: Path) -> list[dict]:
    pths  = sorted(input_dir.glob("*.pth"))
    if not pths:
        print(f"No .pth files found in {input_dir}")
        return []
    results = []
    for p in pths:
        out = output_dir / p.name.replace("best", "release")
        results.append(export_single(p, out))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Optional: push to HuggingFace Hub
# ─────────────────────────────────────────────────────────────────────────────

def push_to_hub(
    local_path: Path,
    repo_id: str,
    token: str | None = None,
) -> None:
    """
    Upload a weight file to a HuggingFace model repo.

    repo_id  : e.g.  "rasuljon-dev/crowdsight"
    token    : HF token (or set env var HUGGING_FACE_HUB_TOKEN)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded → https://huggingface.co/{repo_id}/resolve/main/{local_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CrowdSight weights for release")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input",  help="Single checkpoint to export")
    mode.add_argument("--batch",  action="store_true",
                      help="Export all .pth files in --input_dir")

    p.add_argument("--output",     help="Output path (single mode)")
    p.add_argument("--input_dir",  default="checkpoints")
    p.add_argument("--output_dir", default="release/weights")
    p.add_argument("--model",      default=None, help="Override model_name in checkpoint")
    p.add_argument("--metrics",    default=None,
                   help="Path to benchmark JSON (attach metrics to release file)")
    p.add_argument("--hub",        default=None,
                   help="HuggingFace repo_id to push to (e.g. username/crowdsight)")
    p.add_argument("--hf_token",   default=None, help="HuggingFace token")
    return p.parse_args()


def main():
    args = parse_args()

    metrics = None
    if args.metrics:
        with open(args.metrics) as f:
            data = json.load(f)
        # Flatten list of results to simple dict
        if isinstance(data, list):
            metrics = {}
            for r in data:
                pfx = r["dataset"].upper()
                metrics[f"{pfx}_MAE"]  = r["MAE"]
                metrics[f"{pfx}_RMSE"] = r["RMSE"]
        else:
            metrics = data

    if args.batch:
        results = batch_export(Path(args.input_dir), Path(args.output_dir))
        print(f"\nExported {len(results)} checkpoints.")
    else:
        output = Path(args.output) if args.output else \
                 Path("release") / Path(args.input).name
        result = export_single(
            Path(args.input), output,
            model_name_override=args.model,
            metrics=metrics,
        )
        results = [result]

        if args.hub:
            push_to_hub(output, args.hub, token=args.hf_token)

    # Print summary
    print("\n=== Release summary ===")
    for r in results:
        print(f"  {r['model']:12s}  {r['path']}  ({r['size_mb']} MB)  {r['sha256'][:12]}…")


if __name__ == "__main__":
    main()
