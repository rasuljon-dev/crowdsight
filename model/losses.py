"""
Loss functions for crowd counting.

MSELoss        — standard pixel-wise MSE on density maps (CSRNet default)
DMCountLoss    — Distribution Matching loss from DM-Count (NeurIPS 2020)
                 Paper: https://arxiv.org/abs/2009.13077
                 = count-normalised OT loss + total-variation regularisation
                 Achieves ~59.7 MAE on ShanghaiTech Part A vs 68.2 for plain MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Standard MSE ─────────────────────────────────────────────────────────────

class MSELoss(nn.Module):
    """Pixel-wise MSE between predicted and ground-truth density maps."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, gt)


# ─── DM-Count loss ────────────────────────────────────────────────────────────

class TVLoss(nn.Module):
    """
    Total Variation regularisation — encourages smooth density maps and
    suppresses noise around background regions.

    TV(D) = mean(|D[i,j] - D[i+1,j]| + |D[i,j] - D[i,j+1]|)
    """

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        diff_h = (density[:, :, 1:, :] - density[:, :, :-1, :]).abs()
        diff_w = (density[:, :, :, 1:] - density[:, :, :, :-1]).abs()
        return diff_h.mean() + diff_w.mean()


class OTCountLoss(nn.Module):
    """
    Count-normalised distribution matching loss.

    Instead of full OT (which requires a linear-program solver), we use the
    efficient approximation from DM-Count:

        L_ot = || p_pred / N_pred  -  p_gt / N_gt ||_1

    where p is the spatial density distribution and N is the total count.
    This is the Earth Mover approximation used in the paper's implementation
    (see https://github.com/cvlab-stonybrook/DM-Count).

    When N_gt == 0 (empty scene) we fall back to the raw L1 distance.
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        B = pred.size(0)
        total_loss = torch.tensor(0.0, device=pred.device, requires_grad=False)

        for i in range(B):
            p = pred[i].view(-1)
            g = gt[i].view(-1)

            n_pred = p.sum()
            n_gt   = g.sum()

            if n_gt.item() > 1e-6:
                p_norm = p / (n_pred + 1e-6)
                g_norm = g / (n_gt   + 1e-6)
                total_loss = total_loss + (p_norm - g_norm).abs().sum()
            else:
                total_loss = total_loss + p.abs().sum()

        return total_loss / B


class CountLoss(nn.Module):
    """L1 count loss: |predicted_count - gt_count|."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_count = pred.sum(dim=(1, 2, 3))
        gt_count   = gt.sum(dim=(1, 2, 3))
        return (pred_count - gt_count).abs().mean()


class DMCountLoss(nn.Module):
    """
    Full DM-Count loss (NeurIPS 2020).

    L = w_ot * L_ot  +  w_tv * L_tv  +  w_count * L_count

    Default weights from the original paper.
    """

    def __init__(
        self,
        w_ot: float    = 0.1,
        w_tv: float    = 0.01,
        w_count: float = 1.0,
    ):
        super().__init__()
        self.ot_loss    = OTCountLoss()
        self.tv_loss    = TVLoss()
        self.count_loss = CountLoss()
        self.w_ot    = w_ot
        self.w_tv    = w_tv
        self.w_count = w_count

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        loss : scalar tensor
        components : dict with individual loss values (for logging)
        """
        # Clamp predictions — density values are non-negative
        pred_pos = F.relu(pred)

        l_ot    = self.ot_loss(pred_pos, gt)
        l_tv    = self.tv_loss(pred_pos)
        l_count = self.count_loss(pred_pos, gt)

        loss = self.w_ot * l_ot + self.w_tv * l_tv + self.w_count * l_count

        return loss, {
            "ot":       l_ot.item(),
            "tv":       l_tv.item(),
            "count": l_count.item(),
        }


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_loss(name: str) -> nn.Module:
    """
    Return a loss module by name.

    name: 'mse' | 'dmcount'
    """
    if name == "mse":
        return MSELoss()
    elif name in ("dmcount", "dm_count"):
        return DMCountLoss()
    else:
        raise ValueError(f"Unknown loss '{name}'. Choose: mse, dmcount")
