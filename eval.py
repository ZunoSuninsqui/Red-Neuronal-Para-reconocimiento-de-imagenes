"""Evaluation script for the canvas digit model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from AccesoADatos import DIGIT_CLASS_NAMES, load_canvas_digit_dataset
from train import load_config, save_confusion_matrix
from utils.calibration import TemperatureScaler, expected_calibration_error, plot_reliability_diagram, softmax
from utils.transforms import PreprocessConfig, normalise_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa el modelo guardado en el conjunto de validación.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Ruta al archivo de configuración utilizada durante el entrenamiento.",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Semilla para recrear el split de validación.")
    parser.add_argument("--output", type=Path, default=Path("modelos/eval_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"] if "paths" in cfg else cfg

    dataset_path = Path(paths_cfg["dataset"])  # type: ignore[index]
    stats_path = Path(paths_cfg["dataset_stats"])  # type: ignore[index]
    model_path = Path(paths_cfg["checkpoint"])  # type: ignore[index]
    calibration_path = Path(paths_cfg["calibration"])  # type: ignore[index]

    stats_payload = json.loads(stats_path.read_text())
    mean = float(stats_payload["mean"])
    std = float(stats_payload["std"])

    model_cfg = cfg.get("model", {})
    input_size = tuple(model_cfg.get("input_size", [200, 200, 1]))
    height, width, _ = input_size
    preprocess_config = PreprocessConfig(canvas_size=(width, height))
    images, labels, class_names, summary = load_canvas_digit_dataset(
        dataset_path, preprocess_config=preprocess_config, image_size=(width, height)
    )

    x_train, x_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=args.split_seed,
    )

    x_val = normalise_images(x_val, mean, std)

    model = keras.models.load_model(model_path)
    logits = model.predict(x_val, verbose=0)
    probs = softmax(logits)

    predictions = probs.argmax(axis=1)
    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions, average="macro")
    conf_matrix = confusion_matrix(y_val, predictions)

    save_confusion_matrix(conf_matrix, tuple(DIGIT_CLASS_NAMES), Path(paths_cfg["confusion_matrix"]))

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
    }

    reliability_path = Path(paths_cfg["reliability_diagram"])
    reliability_calibrated_path = Path(paths_cfg.get("reliability_diagram_calibrated", reliability_path))

    if calibration_path.exists():
        scaler = TemperatureScaler.load(calibration_path)
        calibrated_probs = scaler.predict_proba(logits)
        ece = expected_calibration_error(calibrated_probs, y_val)
        metrics["ece_calibrated"] = float(ece)
        plot_reliability_diagram(
            calibrated_probs,
            y_val,
            title="Confiabilidad calibrada",
            output_path=reliability_calibrated_path,
        )
    else:
        ece = expected_calibration_error(probs, y_val)
        metrics["ece"] = float(ece)
        plot_reliability_diagram(
            probs,
            y_val,
            title="Confiabilidad sin calibrar",
            output_path=reliability_path,
        )

    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
