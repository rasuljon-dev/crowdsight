"""
Crowd counting evaluation metrics.

Standard metrics for crowd counting benchmarks:
  - MAE  (Mean Absolute Error)  — average absolute count error per image
  - MSE  (Mean Squared Error)   — penalises large errors more heavily
  - RMSE (Root MSE)             — same units as count
"""

import numpy as np
import torch


class CrowdMetrics:
    """Accumulates per-image errors and computes MAE / MSE / RMSE at epoch end."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._errors: list[float] = []

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Parameters
        ----------
        pred : (B, 1, H, W) predicted density maps
        gt   : (B, 1, H, W) ground-truth density maps
        """
        pred_counts = pred.sum(dim=(1, 2, 3)).detach().cpu().numpy()
        gt_counts   = gt.sum(dim=(1, 2, 3)).detach().cpu().numpy()
        self._errors.extend((pred_counts - gt_counts).tolist())

    def compute(self) -> dict:
        if not self._errors:
            return {"MAE": 0.0, "MSE": 0.0, "RMSE": 0.0}
        errs = np.array(self._errors)
        mae  = float(np.mean(np.abs(errs)))
        mse  = float(np.mean(errs ** 2))
        return {
            "MAE":  round(mae, 2),
            "MSE":  round(mse, 2),
            "RMSE": round(float(np.sqrt(mse)), 2),
        }
