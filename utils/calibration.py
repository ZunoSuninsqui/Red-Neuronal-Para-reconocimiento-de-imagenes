"""Probability calibration helpers (temperature scaling, ECE, plots)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib optional in some environments
    plt = None  # type: ignore[assignment]

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow optional in some environments
    tf = None  # type: ignore[assignment]


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, *, epochs: int = 250, lr: float = 0.01) -> float:
        """Optimise the temperature parameter using the validation set."""

        if tf is None:
            raise RuntimeError("TensorFlow es requerido para ajustar la temperatura.")

        temperature = tf.Variable(self.temperature, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        logits_tf = tf.convert_to_tensor(logits, dtype=tf.float32)
        labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                scaled = logits_tf / temperature
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    labels_tf, scaled, from_logits=True
                )
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, [temperature])
            optimizer.apply_gradients(zip(grads, [temperature]))

        self.temperature = float(tf.clip_by_value(temperature, 1e-3, 100.0).numpy())
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.transform(logits)
        return softmax(scaled)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(str(self.temperature))

    @classmethod
    def load(cls, path: Path | str) -> "TemperatureScaler":
        return cls(float(Path(path).read_text().strip()))


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def top_k_predictions(probs: np.ndarray, *, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    top_probs = np.take_along_axis(probs, indices, axis=1)
    order = np.argsort(-top_probs, axis=1)
    sorted_indices = np.take_along_axis(indices, order, axis=1)
    sorted_probs = np.take_along_axis(top_probs, order, axis=1)
    return sorted_indices, sorted_probs


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if not np.any(mask):
            continue
        accuracy = np.mean(predictions[mask] == labels[mask])
        avg_confidence = np.mean(confidences[mask])
        weight = np.mean(mask)
        ece += np.abs(accuracy - avg_confidence) * weight
    return float(ece)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    accuracies = np.zeros(n_bins, dtype=np.float32)
    avg_confidences = np.zeros(n_bins, dtype=np.float32)

    for idx, (bin_lower, bin_upper) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if not np.any(mask):
            accuracies[idx] = np.nan
            avg_confidences[idx] = np.nan
            continue
        accuracies[idx] = np.mean(predictions[mask] == labels[mask])
        avg_confidences[idx] = np.mean(confidences[mask])

    return avg_confidences, accuracies


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
    title: str | None = None,
    output_path: Path | str | None = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib no está disponible para generar gráficas.")

    avg_conf, accuracies = reliability_curve(probs, labels, n_bins=n_bins)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")

    mask = ~np.isnan(avg_conf)
    ax.bar(bins[:-1][mask], accuracies[mask], width=1 / n_bins, alpha=0.6, align="edge", label="Exactitud")
    ax.plot(avg_conf[mask], accuracies[mask], marker="o", color="black", linestyle="none")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confianza media")
    ax.set_ylabel("Exactitud")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200)
    plt.close(fig)


__all__ = [
    "TemperatureScaler",
    "expected_calibration_error",
    "plot_reliability_diagram",
    "reliability_curve",
    "softmax",
    "top_k_predictions",
]
