from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.65,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def multi_scale_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target)
    current_logits = logits
    current_target = target
    for _ in range(2):
        current_logits = F.avg_pool2d(current_logits, kernel_size=2, stride=2)
        current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(current_logits, current_target)
    return loss


def anomaly_loss(
    recon: torch.Tensor,
    clean: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
    improved: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    rec = F.l1_loss(recon, clean)
    if improved:
        seg = multi_scale_bce(logits, mask) + focal_loss(logits, mask) + dice_loss(logits, mask)
    else:
        seg = F.binary_cross_entropy_with_logits(logits, mask)
    total = rec + seg
    return total, {"loss": float(total.detach()), "rec_loss": float(rec.detach()), "seg_loss": float(seg.detach())}
