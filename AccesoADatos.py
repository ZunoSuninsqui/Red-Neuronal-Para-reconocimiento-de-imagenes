"""Utilities for loading the hand-drawn digit dataset.

The dataset is organised by student folders and contains PNG images whose
filenames start with the Spanish word for the digit that they represent
(e.g. "cero1.png", "dos15.png", etc.).  This module offers a convenient
interface to gather every image, normalise it and prepare the labels so the
training script can focus on modelling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

# Mapping from the Spanish prefix found in each filename to the numeric label.
SPANISH_DIGIT_PREFIXES = {
    "cero": 0,
    "uno": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
}

DIGIT_CLASS_NAMES: Sequence[str] = (
    "0 - Cero",
    "1 - Uno",
    "2 - Dos",
    "3 - Tres",
    "4 - Cuatro",
    "5 - Cinco",
    "6 - Seis",
    "7 - Siete",
    "8 - Ocho",
    "9 - Nueve",
)


@dataclass(frozen=True)
class DatasetSummary:
    """Simple container with dataset information."""

    image_count: int
    class_distribution: Sequence[int]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        histogram = ", ".join(
            f"{DIGIT_CLASS_NAMES[i]}: {count}"
            for i, count in enumerate(self.class_distribution)
        )
        return f"DatasetSummary(image_count={self.image_count}, {histogram})"


def _infer_label_from_name(file_name: str) -> int:
    """Return the numeric label encoded in the filename.

    The images follow the naming scheme ``<spanish-digit><identifier>.png``.
    We simply look for the matching prefix at the start of the filename.
    """

    lower_name = file_name.lower()
    for prefix, value in SPANISH_DIGIT_PREFIXES.items():
        if lower_name.startswith(prefix):
            return value
    raise ValueError(
        f"No se pudo inferir la etiqueta para la imagen '{file_name}'. "
        "Asegúrate de que el nombre comience con la palabra del número en español."
    )


def _load_image(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    """Load a PNG image, convert it to grayscale and resize it."""

    with Image.open(path) as image:
        return (
            np.asarray(
                image.convert("L").resize(image_size, Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            / 255.0
        )


def iter_image_files(dataset_root: Path) -> Iterable[Path]:
    """Yield every PNG image inside the student folders."""

    dataset_root = Path(dataset_root)
    for student_dir in sorted(dataset_root.iterdir()):
        images_dir = student_dir / "imagenes"
        if not images_dir.is_dir():
            continue
        for image_path in sorted(images_dir.glob("*.png")):
            yield image_path


def load_canvas_digit_dataset(
    dataset_root: Path,
    image_size: Tuple[int, int] = (200,200),
) -> Tuple[np.ndarray, np.ndarray, Sequence[str], DatasetSummary]:
    """Load every image from the dataset directory.

    Parameters
    ----------
    dataset_root:
        Path to the ``Dataset`` directory shipped with the project.
    image_size:
        Target size (width, height) for each image.  The originals vary in
        resolution, so we normalise them to a fixed size that is suitable for a
        convolutional neural network.

    Returns
    -------
    images:
        Array of shape ``(N, H, W, 1)`` containing grayscale images normalised
        to the ``[0, 1]`` range.
    labels:
        Array of length ``N`` with the numeric label corresponding to each
        image.
    class_names:
        Human-friendly names for each class, useful when generating reports.
    summary:
        Quick statistics about the dataset (number of images and histogram per
        class).
    """

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta del dataset en: {dataset_root!s}"
        )

    images: List[np.ndarray] = []
    labels: List[int] = []
    class_distribution = [0 for _ in DIGIT_CLASS_NAMES]

    for image_path in iter_image_files(dataset_root):
        label = _infer_label_from_name(image_path.stem)
        image = _load_image(image_path, image_size=image_size)
        images.append(image[..., np.newaxis])
        labels.append(label)
        class_distribution[label] += 1

    if not images:
        raise RuntimeError(
            "No se encontraron imágenes dentro del dataset. Verifica la estructura"
            " de carpetas (cada estudiante debe tener una subcarpeta 'imagenes')."
        )

    images_array = np.stack(images, axis=0)
    labels_array = np.asarray(labels, dtype=np.int64)
    summary = DatasetSummary(
        image_count=len(images),
        class_distribution=tuple(class_distribution),
    )
    return images_array, labels_array, DIGIT_CLASS_NAMES, summary


__all__ = [
    "DIGIT_CLASS_NAMES",
    "DatasetSummary",
    "SPANISH_DIGIT_PREFIXES",
    "iter_image_files",
    "load_canvas_digit_dataset",
]
