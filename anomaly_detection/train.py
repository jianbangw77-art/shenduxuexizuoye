from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import apply_synthetic_anomaly, build_datasets
from .losses import anomaly_loss
from .metrics import auroc, best_f1
from .models import DRAEMLite, anomaly_map
from .utils import AverageMeter, ensure_dir, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DRAEM-Lite anomaly detection experiment")
    parser.add_argument("--dataset", choices=["synthetic", "mvtec"], default="synthetic")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--variant", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/run")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def train_one_epoch(
    model: DRAEMLite,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    improved: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()
    meters = {"loss": AverageMeter(), "rec_loss": AverageMeter(), "seg_loss": AverageMeter()}
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)

    for batch in progress:
        clean = batch["image"].to(device)
        corrupted = []
        masks = []
        for index, image in enumerate(clean.cpu()):
            generator = torch.Generator().manual_seed(epoch * 100_003 + index * 9176 + len(corrupted))
            aug, mask = apply_synthetic_anomaly(image, generator, improved=improved)
            corrupted.append(aug)
            masks.append(mask)
        x = torch.stack(corrupted).to(device)
        mask = torch.stack(masks).to(device)

        recon, logits = model(x)
        loss, parts = anomaly_loss(recon, clean, logits, mask, improved=improved)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for key, value in parts.items():
            meters[key].update(value, n=clean.size(0))
        progress.set_postfix(loss=f"{meters['loss'].avg:.4f}")

    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def evaluate(model: DRAEMLite, loader: DataLoader, device: torch.device, improved: bool) -> dict[str, float]:
    model.eval()
    image_scores = []
    image_labels = []
    pixel_scores = []
    pixel_labels = []

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        recon, logits = model(x)
        score_map = anomaly_map(x, recon, logits, improved=improved)
        score = score_map.flatten(1).amax(dim=1)

        image_scores.append(score.cpu())
        image_labels.append(label.cpu())
        pixel_scores.append(score_map.cpu().flatten())
        pixel_labels.append(mask.cpu().flatten())

    image_scores_t = torch.cat(image_scores)
    image_labels_t = torch.cat(image_labels)
    pixel_scores_t = torch.cat(pixel_scores)
    pixel_labels_t = torch.cat(pixel_labels)
    return {
        "image_auroc": auroc(image_scores_t, image_labels_t),
        "pixel_auroc": auroc(pixel_scores_t, pixel_labels_t),
        "pixel_best_f1": best_f1(pixel_scores_t, pixel_labels_t),
    }


def run(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device)
    improved = args.variant == "improved"

    train_set, test_set = build_datasets(args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DRAEMLite(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []

    for epoch in range(1, args.epochs + 1):
        train_log = train_one_epoch(model, train_loader, optimizer, device, improved, epoch)
        eval_log = evaluate(model, test_loader, device, improved)
        row = {"epoch": epoch, **train_log, **eval_log}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    metrics = history[-1]
    torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, output_dir / "model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
