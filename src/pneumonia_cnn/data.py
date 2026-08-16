from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NORMALISATION_MEAN = 0.5
NORMALISATION_STD = 0.25


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    split: str


def discover_images(normal_dir: Path, pneumonia_dir: Path) -> list[tuple[Path, int]]:
    records: list[tuple[Path, int]] = []
    for directory, label in ((normal_dir, 0), (pneumonia_dir, 1)):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
        paths = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise RuntimeError(f"No supported images were found in {directory}")
        records.extend((path, label) for path in paths)
    return records


def make_stratified_split(
    images: Iterable[tuple[Path, int]], seed: int
) -> list[ImageRecord]:
    rng = random.Random(seed)
    by_label: dict[int, list[Path]] = {0: [], 1: []}
    for path, label in images:
        by_label[label].append(path)

    split_records: list[ImageRecord] = []
    for label, paths in by_label.items():
        rng.shuffle(paths)
        n_test = round(0.15 * len(paths))
        n_validation = round(0.15 * len(paths))
        test_paths = paths[:n_test]
        validation_paths = paths[n_test : n_test + n_validation]
        training_paths = paths[n_test + n_validation :]
        split_records.extend(ImageRecord(path, label, "test") for path in test_paths)
        split_records.extend(ImageRecord(path, label, "validation") for path in validation_paths)
        split_records.extend(ImageRecord(path, label, "training") for path in training_paths)

    split_records.sort(key=lambda record: (record.split, record.label, str(record.path)))
    return split_records


def save_split_manifest(records: list[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "class_name", "split"))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": str(record.path.resolve()),
                    "label": record.label,
                    "class_name": "pneumonia" if record.label == 1 else "normal",
                    "split": record.split,
                }
            )


def load_split_manifest(path: Path) -> list[ImageRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            ImageRecord(Path(row["path"]), int(row["label"]), row["split"])
            for row in csv.DictReader(handle)
        ]


def audit_images(records: list[ImageRecord]) -> dict[str, int]:
    counts = {"training": 0, "validation": 0, "test": 0, "normal": 0, "pneumonia": 0}
    identities: set[Path] = set()
    for record in records:
        resolved = record.path.resolve()
        if resolved in identities:
            raise RuntimeError(f"Duplicate image path across the manifest: {resolved}")
        identities.add(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {resolved}")
        try:
            with Image.open(resolved) as image:
                image.verify()
        except Exception as error:
            raise RuntimeError(f"Unreadable image: {resolved}") from error
        counts[record.split] += 1
        counts["pneumonia" if record.label == 1 else "normal"] += 1
    return counts


def _augment(image: Image.Image, noise_std: float) -> torch.Tensor:
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    angle = random.uniform(-7.0, 7.0)
    translation_limit = int(round(0.04 * image.width))
    translation = (
        random.randint(-translation_limit, translation_limit),
        random.randint(-translation_limit, translation_limit),
    )
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, translate=translation, fillcolor=0)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.90, 1.10))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.90, 1.10))
    tensor = image_to_raw_tensor(image)
    if noise_std > 0:
        tensor = tensor + torch.randn_like(tensor) * noise_std
    return tensor.clamp(0.0, 1.0)


def image_to_raw_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def normalise(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - NORMALISATION_MEAN) / NORMALISATION_STD


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * NORMALISATION_STD + NORMALISATION_MEAN


def load_raw_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return image_to_raw_tensor(image)


class ChestXrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        records: list[ImageRecord],
        image_size: int,
        augment: bool,
        noise_std: float,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        raw_tensor = _augment(image, self.noise_std) if self.augment else image_to_raw_tensor(image)
        label = torch.tensor(float(record.label), dtype=torch.float32)
        return normalise(raw_tensor), label, str(record.path)

