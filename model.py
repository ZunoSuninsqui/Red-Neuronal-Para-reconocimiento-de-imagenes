"""Definition of the convolutional neural network used for digit recognition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn


class DigitCNN(nn.Module):
    """A lightweight convolutional neural network inspired by LeNet."""

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.classifier(self.features(x))


@dataclass
class ModelMetadata:
    """Metadata stored together with the model weights."""

    input_size: Tuple[int, int, int] = (1, 28, 28)
    classes: Tuple[str, ...] = tuple(str(i) for i in range(1, 10))
    normalisation: str = "[0,1]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "classes": list(self.classes),
            "normalisation": self.normalisation,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelMetadata":
        return ModelMetadata(
            input_size=tuple(data.get("input_size", (1, 28, 28))),
            classes=tuple(data.get("classes", [str(i) for i in range(1, 10)])),
            normalisation=data.get("normalisation", "[0,1]"),
        )


def build_model(device: torch.device | None = None, num_classes: int = 9) -> DigitCNN:
    model = DigitCNN(num_classes=num_classes)
    if device is not None:
        model.to(device)
    return model


def save_model(path: Path, model: nn.Module, metadata: ModelMetadata) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata.to_dict(),
    }
    torch.save(payload, path)


def load_model(path: Path, device: torch.device | None = None) -> Tuple[DigitCNN, ModelMetadata]:
    checkpoint = torch.load(path, map_location=device or "cpu")
    metadata = ModelMetadata.from_dict(checkpoint.get("metadata", {}))
    model = DigitCNN(num_classes=len(metadata.classes))
    model.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    return model, metadata
