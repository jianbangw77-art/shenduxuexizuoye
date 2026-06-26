from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _mask_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("L").resize((image_size, image_size), Image.NEAREST)
    array = (np.asarray(image, dtype=np.float32) > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class SyntheticSurfaceDataset(Dataset):
    """Deterministic synthetic industrial-surface dataset.

    Train split returns normal samples only. Test split returns a balanced set
    of normal and anomalous samples with pixel masks, so the full anomaly
    detection pipeline can run without downloading an external dataset.
    """

    def __init__(
        self,
        split: str,
        length: int = 512,
        image_size: int = 128,
        anomaly_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        self.length = length
        self.image_size = image_size
        self.anomaly_ratio = anomaly_ratio
        self.seed = seed

        y = torch.linspace(-1.0, 1.0, image_size)
        x = torch.linspace(-1.0, 1.0, image_size)
        self.yy, self.xx = torch.meshgrid(y, x, indexing="ij")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index * 9973)
        image = self._normal_surface(generator)
        mask = torch.zeros(1, self.image_size, self.image_size)
        label = torch.tensor(0.0)

        if self.split == "test":
            is_anomaly = (index % int(1.0 / self.anomaly_ratio)) == 0
            if is_anomaly:
                image, mask = apply_synthetic_anomaly(image, generator, improved=True)
                label = torch.tensor(1.0)

        return {"image": image.clamp(0.0, 1.0), "mask": mask, "label": label}

    def _normal_surface(self, generator: torch.Generator) -> torch.Tensor:
        phase = torch.rand(3, generator=generator) * math.pi
        freq_x = torch.randint(2, 7, (3,), generator=generator).float()
        freq_y = torch.randint(2, 7, (3,), generator=generator).float()

        channels = []
        for channel in range(3):
            waves = (
                0.08 * torch.sin(freq_x[channel] * self.xx + phase[channel])
                + 0.08 * torch.cos(freq_y[channel] * self.yy - phase[channel])
            )
            rings = 0.04 * torch.sin(8.0 * torch.sqrt(self.xx**2 + self.yy**2) + phase[channel])
            gradient = 0.12 * self.xx + 0.08 * self.yy
            noise = 0.04 * torch.randn(self.image_size, self.image_size, generator=generator)
            channels.append(0.52 + gradient + waves + rings + noise)
        image = torch.stack(channels)
        return image.clamp(0.0, 1.0)


class MVTecDataset(Dataset):
    """Minimal MVTec AD loader for class-level folders."""

    def __init__(self, root: str | Path, split: str, image_size: int = 256) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[tuple[Path, Path | None, int]]:
        if self.split == "train":
            files = sorted((self.root / "train" / "good").glob("*"))
            return [(path, None, 0) for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]

        samples: list[tuple[Path, Path | None, int]] = []
        test_root = self.root / "test"
        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            for image_path in sorted(defect_dir.glob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if defect_dir.name == "good":
                    samples.append((image_path, None, 0))
                else:
                    mask_dir = self.root / "ground_truth" / defect_dir.name
                    mask_path = mask_dir / f"{image_path.stem}_mask{image_path.suffix}"
                    samples.append((image_path, mask_path if mask_path.exists() else None, 1))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, mask_path, label = self.samples[index]
        image = _pil_to_tensor(Image.open(image_path), self.image_size)
        if mask_path is None:
            mask = torch.zeros(1, self.image_size, self.image_size)
        else:
            mask = _mask_to_tensor(Image.open(mask_path), self.image_size)
        return {"image": image, "mask": mask, "label": torch.tensor(float(label))}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_datasets(args) -> tuple[Dataset, Dataset]:
    if args.dataset == "synthetic":
        train_set = SyntheticSurfaceDataset(
            split="train",
            length=args.train_size,
            image_size=args.image_size,
            seed=args.seed,
        )
        test_set = SyntheticSurfaceDataset(
            split="test",
            length=args.test_size,
            image_size=args.image_size,
            seed=args.seed + 10_000,
        )
        return train_set, test_set

    if args.dataset == "mvtec":
        if args.data_root is None:
            raise ValueError("--data-root is required for --dataset mvtec")
        return (
            MVTecDataset(args.data_root, split="train", image_size=args.image_size),
            MVTecDataset(args.data_root, split="test", image_size=args.image_size),
        )

    raise ValueError(f"Unknown dataset: {args.dataset}")


def apply_synthetic_anomaly(
    image: torch.Tensor,
    generator: torch.Generator,
    improved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _random_mask(image.shape[-2], image.shape[-1], generator, improved)
    if improved:
        texture = _structured_noise_like(image, generator)
        alpha = 0.55 + 0.35 * torch.rand(1, generator=generator).item()
        shift = int(torch.randint(4, 19, (1,), generator=generator).item())
        shifted = torch.roll(image, shifts=shift, dims=-1)
        corruption = alpha * texture + (1.0 - alpha) * shifted
    else:
        corruption = torch.rand(image.shape, generator=generator)

    brightness = 0.75 + 0.65 * torch.rand(1, generator=generator).item()
    anomalous = image * (1.0 - mask) + (corruption * brightness).clamp(0.0, 1.0) * mask
    return anomalous.clamp(0.0, 1.0), mask


def _random_mask(height: int, width: int, generator: torch.Generator, improved: bool) -> torch.Tensor:
    mask = torch.zeros(1, height, width)
    count = int(torch.randint(1, 4 if improved else 3, (1,), generator=generator).item())
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    for _ in range(count):
        kind = int(torch.randint(0, 4 if improved else 3, (1,), generator=generator).item())
        if kind == 0:
            cy = int(torch.randint(height // 5, height * 4 // 5, (1,), generator=generator).item())
            cx = int(torch.randint(width // 5, width * 4 // 5, (1,), generator=generator).item())
            ry = int(torch.randint(max(4, height // 24), max(8, height // 8), (1,), generator=generator).item())
            rx = int(torch.randint(max(4, width // 24), max(8, width // 8), (1,), generator=generator).item())
            blob = ((yy - cy).float() / ry) ** 2 + ((xx - cx).float() / rx) ** 2 <= 1.0
            mask[0] = torch.maximum(mask[0], blob.float())
        elif kind == 1:
            y0 = int(torch.randint(0, height - height // 8, (1,), generator=generator).item())
            x0 = int(torch.randint(0, width - width // 8, (1,), generator=generator).item())
            h = int(torch.randint(max(4, height // 24), max(6, height // 6), (1,), generator=generator).item())
            w = int(torch.randint(max(4, width // 24), max(6, width // 6), (1,), generator=generator).item())
            mask[:, y0 : y0 + h, x0 : x0 + w] = 1.0
        elif kind == 2:
            y = int(torch.randint(height // 8, height * 7 // 8, (1,), generator=generator).item())
            thickness = int(torch.randint(1, 4 if improved else 3, (1,), generator=generator).item())
            slope = (torch.rand(1, generator=generator).item() - 0.5) * 0.6
            line = (yy.float() - y - slope * (xx.float() - width / 2.0)).abs() <= thickness
            mask[0] = torch.maximum(mask[0], line.float())
        else:
            speckles = torch.rand(1, height, width, generator=generator) > 0.985
            mask = torch.maximum(mask, speckles.float())

    return mask


def _structured_noise_like(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.rand(image.shape, generator=generator)
    for _ in range(3):
        noise = torch.nn.functional.avg_pool2d(noise.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    color = torch.rand(3, 1, 1, generator=generator) * 0.8 + 0.1
    return (0.65 * noise + 0.35 * color).clamp(0.0, 1.0)
