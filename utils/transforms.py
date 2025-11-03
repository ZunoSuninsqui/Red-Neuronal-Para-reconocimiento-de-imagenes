"""Image preprocessing utilities shared between training and inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

DEFAULT_CANVAS_SIZE: Tuple[int, int] = (200, 200)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration describing how to normalise input digits."""

    canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE
    threshold: float = 0.1
    padding: int = 12
    blur_radius: float = 0.0


def _ensure_grayscale(image: Image.Image) -> Image.Image:
    if image.mode != "L":
        return image.convert("L")
    return image


def _auto_invert(image: Image.Image) -> Image.Image:
    """Invert the image if the background appears darker than the foreground."""

    arr = np.asarray(image, dtype=np.float32)
    mean_intensity = arr.mean() / 255.0
    # For most samples the background is white (high intensity).  We invert when the
    # mean intensity is larger than 0.5 so that strokes end up having large values.
    if mean_intensity > 0.5:
        return ImageOps.invert(image)
    return image


def _apply_optional_blur(image: Image.Image, blur_radius: float) -> Image.Image:
    if blur_radius <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def _find_digit_bbox(arr: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    mask = arr > threshold
    if not mask.any():
        # Fallback: keep the full image when the drawing is too faint.
        return (0, 0, arr.shape[1], arr.shape[0])

    ys, xs = np.where(mask)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, y_min, x_max + 1, y_max + 1


def _crop_and_pad(image: Image.Image, bbox: Tuple[int, int, int, int], padding: int) -> Image.Image:
    cropped = image.crop(bbox)
    if padding <= 0:
        return cropped

    pad = (padding, padding, padding, padding)
    return ImageOps.expand(cropped, border=pad, fill=255)


def _resize_with_aspect_ratio(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    width, height = size
    image.thumbnail((width, height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", size, color=255)
    offset_x = (width - image.width) // 2
    offset_y = (height - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))
    return canvas


def preprocess_pil_image(
    image: Image.Image,
    *,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Convert a raw PIL image into a normalised ``np.ndarray``.

    The function mimics the transformations applied during training so that
    inference receives tensors with the same statistics.
    """

    if config is None:
        config = PreprocessConfig()

    gray = _ensure_grayscale(image)
    inverted = _auto_invert(gray)
    blurred = _apply_optional_blur(inverted, config.blur_radius)

    arr = np.asarray(blurred, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    # Normalise contrast by stretching the histogram.
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
    bbox = _find_digit_bbox(arr, threshold=config.threshold)

    processed = _crop_and_pad(blurred, bbox, padding=config.padding)
    resized = _resize_with_aspect_ratio(processed, config.canvas_size)
    output = np.asarray(resized, dtype=np.float32) / 255.0
    return output[..., np.newaxis]


def preprocess_image_path(path: Path | str, *, config: PreprocessConfig | None = None) -> np.ndarray:
    with Image.open(path) as image:
        return preprocess_pil_image(image, config=config)


def batch_preprocess(paths: Iterable[Path], *, config: PreprocessConfig | None = None) -> np.ndarray:
    processed = [preprocess_image_path(path, config=config) for path in paths]
    return np.stack(processed, axis=0)


def compute_dataset_statistics(images: np.ndarray) -> Tuple[float, float]:
    """Return mean and standard deviation for normalisation."""

    mean = float(images.mean())
    std = float(images.std() + 1e-8)
    return mean, std


def normalise_images(images: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (images - mean) / max(std, 1e-6)


__all__ = [
    "DEFAULT_CANVAS_SIZE",
    "PreprocessConfig",
    "batch_preprocess",
    "compute_dataset_statistics",
    "normalise_images",
    "preprocess_image_path",
    "preprocess_pil_image",
]
