"""Dataset scanning and loading utilities."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from preprocess import prepare_tensor

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DIGIT_CLASSES = [str(i) for i in range(1, 10)]
DIGIT_TO_INDEX = {int(digit): idx for idx, digit in enumerate(DIGIT_CLASSES, start=1)}


@dataclass
class Sample:
    """Pair of image path and label."""

    path: Path
    label: int


@dataclass
class DatasetSummary:
    """Information about the dataset distribution."""

    root: Path
    total_images: int
    class_counts: Dict[int, int]

    @property
    def missing_classes(self) -> List[int]:
        return [digit for digit in range(1, 10) if digit not in self.class_counts]


class DatasetScanner:
    """Recursively scan folders to extract labelled image samples."""

    def __init__(self, min_label: int = 1, max_label: int = 9) -> None:
        self.min_label = min_label
        self.max_label = max_label
        self._file_pattern = re.compile(r"([1-9])")

    def scan(self, root: Path) -> Tuple[List[Sample], DatasetSummary]:
        samples: List[Sample] = []
        counters: Counter[int] = Counter()
        for file_path in self._iter_files(root):
            label = self._infer_label(file_path)
            if label is None:
                continue
            samples.append(Sample(path=file_path, label=label))
            counters[label] += 1
        summary = DatasetSummary(root=root, total_images=sum(counters.values()), class_counts=dict(counters))
        return samples, summary

    def _iter_files(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                yield path

    def _infer_label(self, path: Path) -> int | None:
        parent = path.parent
        if parent.name.isdigit():
            value = int(parent.name)
            if self.min_label <= value <= self.max_label:
                return value

        if parent.name.lower() == "imagenes":
            upper = parent.parent
            if upper.name.isdigit():
                value = int(upper.name)
                if self.min_label <= value <= self.max_label:
                    return value
            match = self._file_pattern.search(path.stem)
            if match:
                value = int(match.group(1))
                if self.min_label <= value <= self.max_label:
                    return value
            return None

        match = self._file_pattern.search(parent.name)
        if match:
            value = int(match.group(1))
            if self.min_label <= value <= self.max_label:
                return value

        match = self._file_pattern.search(path.stem)
        if match:
            value = int(match.group(1))
            if self.min_label <= value <= self.max_label:
                return value
        return None


class DigitDataset(Dataset[Tuple[torch.Tensor, int]]):
    """Torch dataset for digit images."""

    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        tensor = prepare_tensor(sample.path)
        label_index = sample.label - 1
        return tensor, label_index


def split_dataset(
    samples: Sequence[Sample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """Split the dataset into train and validation sets preserving class balance."""

    labels = [sample.label for sample in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=validation_ratio,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    return list(train_samples), list(val_samples)


def create_dataloaders(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch dataloaders for both splits."""

    train_dataset = DigitDataset(train_samples)
    val_dataset = DigitDataset(val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
