"""CLI utility to run inference on new digit drawings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tensorflow import keras

from utils.calibration import TemperatureScaler, softmax, top_k_predictions
from utils.transforms import PreprocessConfig, preprocess_image_path

DEFAULT_CONFIG = Path("configs/default.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clasifica un dígito dibujado a mano.")
    parser.add_argument("image", type=Path, help="Ruta a la imagen (PNG/JPG).")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Archivo de configuración utilizado para localizar el modelo.",
    )
    parser.add_argument("--model", type=Path, default=None, help="Ruta al checkpoint del modelo.")
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("classes.json"),
        help="Archivo JSON con los nombres de las clases.",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=None,
        help="Archivo JSON con la media y desviación estándar utilizadas en entrenamiento.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Archivo con la temperatura calculada durante la calibración.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Número de clases a mostrar.")
    parser.add_argument(
        "--raw", action="store_true", help="Muestra la salida como JSON crudo para integraciones."
    )
    parser.add_argument(
        "--no-calibration", action="store_true", help="Deshabilita la calibración de temperatura.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Dict[str, str]]:
    from train import load_config as load_training_config  # Reutilizamos el lector YAML

    cfg = load_training_config(path)
    return cfg


def load_stats(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text())
    return {"mean": float(payload["mean"]), "std": float(payload["std"])}


def detect_empty_image(image: np.ndarray, threshold: float = 0.01) -> bool:
    energy = float(np.mean(image))
    return energy < threshold


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]
    model_cfg = cfg.get("model", {})
    input_size = tuple(model_cfg.get("input_size", [200, 200, 1]))
    height, width, _ = input_size
    preprocess_config = PreprocessConfig(canvas_size=(width, height))

    model_path = args.model or Path(paths_cfg["checkpoint"])
    stats_path = args.dataset_stats or Path(paths_cfg["dataset_stats"])
    calibration_path = args.calibration or Path(paths_cfg["calibration"])

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(
            "No se encontraron las estadísticas del dataset. Ejecuta 'python train.py' primero."
        )

    stats = load_stats(stats_path)
    classes: List[str] = json.loads(args.classes.read_text())

    image_array = preprocess_image_path(args.image, config=preprocess_config)

    if detect_empty_image(image_array):
        print("La imagen parece vacía o con poca tinta. Resultado poco fiable.")

    std = stats["std"] if stats["std"] != 0 else 1.0
    normalised = (image_array - stats["mean"]) / std
    batch = normalised[np.newaxis, ...]

    model = keras.models.load_model(model_path)
    logits = model.predict(batch, verbose=0)

    if not args.no_calibration and calibration_path.exists():
        scaler = TemperatureScaler.load(calibration_path)
        probs = scaler.predict_proba(logits)
    else:
        probs = softmax(logits)

    topk = max(1, min(args.topk, probs.shape[1]))
    indices, top_probs = top_k_predictions(probs, k=topk)

    predictions = [
        {"class_id": int(idx), "class_name": classes[int(idx)], "probability": float(prob)}
        for idx, prob in zip(indices[0], top_probs[0])
    ]

    if args.raw:
        print(json.dumps({"predictions": predictions}, indent=2, ensure_ascii=False))
    else:
        best = predictions[0]
        print(f"Predicción: {best['class_id']} ({best['class_name']}) -> {best['probability']:.2%}")
        print("Top-k:")
        for pred in predictions:
            print(f" - {pred['class_name']}: {pred['probability']:.2%}")


if __name__ == "__main__":
    main()
