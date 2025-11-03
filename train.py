"""Training entry point for the canvas digit recogniser."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from AccesoADatos import DIGIT_CLASS_NAMES, load_canvas_digit_dataset
from utils.calibration import TemperatureScaler, expected_calibration_error, plot_reliability_diagram, softmax
from utils.seed import seed_everything
from utils.transforms import PreprocessConfig, normalise_images
from models import build_canvas_digit_model


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [root]
    indent_stack = [0]

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        key, _, value = line.partition(":")
        if not _:
            raise ValueError(f"No se pudo interpretar la línea de configuración: {raw_line}")
        key = key.strip()
        value = value.strip()

        while indent < indent_stack[-1]:
            stack.pop()
            indent_stack.pop()

        if not value:
            new_dict: Dict[str, Any] = {}
            stack[-1][key] = new_dict
            stack.append(new_dict)
            indent_stack.append(indent + 2)
            continue

        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            lowered = value.lower()
            if lowered == "true":
                parsed_value = True
            elif lowered == "false":
                parsed_value = False
            else:
                parsed_value = value
        stack[-1][key] = parsed_value
    return root


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        return _parse_simple_yaml(text)
    else:  # pragma: no cover - executed only when PyYAML is installed
        return yaml.safe_load(text)  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y evalúa el modelo de dígitos.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Ruta al archivo de configuración YAML.",
    )
    return parser.parse_args()


def save_confusion_matrix(matrix: np.ndarray, class_names: Tuple[str, ...], path: Path) -> None:
    import matplotlib.pyplot as plt  # local import to keep optional dependency

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión (validación)")

    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:d}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(int(config.get("seed", 42)))

    paths_cfg = config["paths"]
    dataset_path = Path(paths_cfg["dataset"]).resolve()
    model_dir = Path(paths_cfg["model_dir"]).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    input_size = tuple(config["model"]["input_size"])
    height, width, channels = input_size
    preprocess_config = PreprocessConfig(canvas_size=(width, height))

    print("Cargando dataset desde", dataset_path)
    images, labels, class_names, summary = load_canvas_digit_dataset(
        dataset_path,
        image_size=(width, height),
        preprocess_config=preprocess_config,
    )

    print(f"Total de imágenes: {summary.image_count}")
    x_train, x_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=int(config.get("seed", 42)),
    )

    mean = float(x_train.mean())
    std = float(x_train.std() + 1e-8)
    x_train = normalise_images(x_train, mean, std)
    x_val = normalise_images(x_val, mean, std)

    stats_path = Path(paths_cfg["dataset_stats"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_payload = {"mean": mean, "std": std}
    stats_path.write_text(json.dumps(stats_payload, indent=2))

    model = build_canvas_digit_model(
        input_shape=(height, width, channels),
        base_filters=int(config["model"]["base_filters"]),
        dropout=float(config["model"]["dropout"]),
        weight_decay=float(config["model"]["weight_decay"]),
        augmentation_params=config.get("augmentation", {}),
    )

    optimizer = keras.optimizers.Adam(learning_rate=float(config["training"]["initial_learning_rate"]))
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(paths_cfg["checkpoint"]),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=int(config["training"]["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(config["training"]["reduce_lr_factor"]),
            patience=int(config["training"]["reduce_lr_patience"]),
            min_lr=float(config["training"]["min_learning_rate"]),
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        callbacks=callbacks,
        verbose=2,
    )

    checkpoint_path = Path(paths_cfg["checkpoint"])
    if checkpoint_path.exists():
        model = keras.models.load_model(checkpoint_path)

    logits_val = model.predict(x_val, verbose=0)
    probs_val = softmax(logits_val)

    y_pred = probs_val.argmax(axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_val, y_pred)

    confusion_path = Path(paths_cfg["confusion_matrix"])
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(conf_matrix, tuple(DIGIT_CLASS_NAMES), confusion_path)

    metrics_payload: Dict[str, Any] = {
        "history": history.history,
        "dataset": {
            "total_images": summary.image_count,
            "class_distribution": list(summary.class_distribution),
            "class_names": list(class_names),
            "mean": mean,
            "std": std,
        },
        "validation_metrics": {
            "accuracy": accuracy,
            "f1_macro": f1,
        },
    }

    calibration_cfg = config.get("calibration", {})
    scaler = TemperatureScaler()
    temperature = scaler.fit(
        logits_val,
        y_val,
        epochs=int(calibration_cfg.get("epochs", 250)),
        lr=float(calibration_cfg.get("learning_rate", 0.01)),
    )
    calibrated_probs = scaler.predict_proba(logits_val)

    ece_before = expected_calibration_error(probs_val, y_val, n_bins=int(calibration_cfg.get("n_bins", 15)))
    ece_after = expected_calibration_error(
        calibrated_probs, y_val, n_bins=int(calibration_cfg.get("n_bins", 15))
    )

    metrics_payload["calibration"] = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }

    reliability_path = Path(paths_cfg["reliability_diagram"])
    reliability_path.parent.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(
        probs_val,
        y_val,
        n_bins=int(calibration_cfg.get("n_bins", 15)),
        title="Confiabilidad antes de calibrar",
        output_path=reliability_path,
    )

    calibrated_reliability_path = Path(paths_cfg.get("reliability_diagram_calibrated", reliability_path))
    calibrated_reliability_path.parent.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(
        calibrated_probs,
        y_val,
        n_bins=int(calibration_cfg.get("n_bins", 15)),
        title="Confiabilidad después de calibrar",
        output_path=calibrated_reliability_path,
    )

    scaler.save(paths_cfg["calibration"])

    metrics_path = Path(paths_cfg["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False))

    classes_path = Path(paths_cfg["classes"])
    if not classes_path.exists():
        classes_path.write_text(json.dumps(list(class_names), indent=2, ensure_ascii=False))

    print("Entrenamiento completado.")
    print(json.dumps(metrics_payload["validation_metrics"], indent=2))


if __name__ == "__main__":
    main()
