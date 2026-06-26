from __future__ import annotations

import torch


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().float()
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    tpr = torch.cumsum(sorted_labels, dim=0) / positives
    fpr = torch.cumsum(1.0 - sorted_labels, dim=0) / negatives
    tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
    fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])
    return float(torch.trapz(tpr, fpr))


def best_f1(scores: torch.Tensor, labels: torch.Tensor, steps: int = 80) -> float:
    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().float()
    if labels.sum() == 0:
        return float("nan")
    thresholds = torch.linspace(float(scores.min()), float(scores.max()), steps, device=scores.device)
    best = 0.0
    for threshold in thresholds:
        pred = scores >= threshold
        tp = (pred & (labels > 0.5)).sum().float()
        fp = (pred & (labels <= 0.5)).sum().float()
        fn = ((~pred) & (labels > 0.5)).sum().float()
        f1 = (2.0 * tp / (2.0 * tp + fp + fn + 1e-8)).item()
        best = max(best, f1)
    return best
