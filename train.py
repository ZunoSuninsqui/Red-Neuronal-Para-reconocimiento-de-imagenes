"""Training utilities and orchestration for the digit recogniser."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from data_loader import Sample, create_dataloaders, split_dataset
from model import ModelMetadata, build_model


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    validation_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0
    min_images_per_class: int = 20


@dataclass
class TrainingResult:
    model: nn.Module
    history: Dict[str, List[float]]
    confusion: np.ndarray
    metadata: ModelMetadata


def _format_metrics(epoch: int, metrics: Dict[str, float]) -> str:
    parts = [f"Época {epoch}"]
    for key, value in metrics.items():
        if key.endswith("accuracy"):
            parts.append(f"{key}: {value:.2%}")
        elif key == "tiempo":
            parts.append(f"{key}: {value:.1f}s")
        else:
            parts.append(f"{key}: {value:.4f}")
    return " | ".join(parts)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    loss_value = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return loss_value, accuracy


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    loss_value = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return loss_value, accuracy


def train_model(
    samples: Sequence[Sample],
    device: torch.device,
    config: TrainingConfig,
    log_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> TrainingResult:
    """Train the network returning the fitted model and metrics."""

    train_samples, val_samples = split_dataset(
        samples, validation_ratio=config.validation_ratio, seed=config.seed
    )
    train_loader, val_loader = create_dataloaders(
        train_samples,
        val_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = build_model(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, config.epochs + 1):
        if cancel_event and cancel_event.is_set():
            if log_callback:
                log_callback("Entrenamiento cancelado por el usuario.")
            break
        start_time = time.time()
        train_loss, train_accuracy = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = _evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        if log_callback:
            elapsed = time.time() - start_time
            log_callback(
                _format_metrics(
                    epoch,
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "tiempo": elapsed,
                    },
                )
            )

    confusion = compute_confusion_matrix(model, val_loader, device)
    model.eval()
    metadata = ModelMetadata()
    return TrainingResult(model=model, history=history, confusion=confusion, metadata=metadata)


def compute_confusion_matrix(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    matrix = confusion_matrix(all_labels, all_preds, labels=list(range(9)))
    return matrix
