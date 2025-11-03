from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
pytest.importorskip("tensorflow")
from PIL import Image, ImageDraw

from utils.calibration import TemperatureScaler, expected_calibration_error, softmax, top_k_predictions
from utils.transforms import PreprocessConfig, normalise_images, preprocess_pil_image


@pytest.fixture()
def synthetic_digit() -> Image.Image:
    image = Image.new("L", (256, 256), color=255)
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 60, 200, 200), fill=0)
    return image


def test_preprocess_centres_digit(synthetic_digit: Image.Image) -> None:
    config = PreprocessConfig(canvas_size=(200, 200))
    processed = preprocess_pil_image(synthetic_digit, config=config)
    assert processed.shape == (200, 200, 1)

    mask = processed.squeeze() > 0.2
    coords = np.argwhere(mask)
    y_mean, x_mean = coords.mean(axis=0)
    assert 80 < x_mean < 120
    assert 80 < y_mean < 120


def test_normalise_images() -> None:
    array = np.random.rand(10, 5, 5, 1).astype(np.float32)
    mean = float(array.mean())
    std = float(array.std() + 1e-8)
    normalised = normalise_images(array, mean, std)
    assert np.isclose(normalised.mean(), 0.0, atol=1e-6)


def test_temperature_scaler_improves_ece() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(128, 10)).astype(np.float32)
    labels = rng.integers(0, 10, size=(128,))

    probs = softmax(logits)
    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, labels, epochs=50, lr=0.05)
    assert temperature > 0

    calibrated = scaler.predict_proba(logits)
    ece_before = expected_calibration_error(probs, labels)
    ece_after = expected_calibration_error(calibrated, labels)
    assert ece_after <= ece_before + 1e-4


def test_top_k_predictions_shapes() -> None:
    probs = np.tile(np.linspace(0.1, 1.0, 10), (2, 1))
    indices, values = top_k_predictions(probs, k=3)
    assert indices.shape == (2, 3)
    assert values.shape == (2, 3)
    assert np.all(values[:, 0] >= values[:, 1])
