"""Script principal para entrenar una red neuronal con el dataset de dígitos.

El objetivo es reconocer números dibujados a mano en un canvas.  Para facilitar
su uso desde la línea de comandos, basta con ejecutar ``python main.py`` y el
script se encargará de:

1. Cargar todas las imágenes de la carpeta :mod:`Dataset`.
2. Normalizarlas a un tamaño uniforme y dividirlas en conjuntos de entrenamiento
   y validación.
3. Entrenar una red neuronal convolucional (CNN) utilizando TensorFlow/Keras.
4. Guardar el modelo entrenado y mostrar métricas básicas de rendimiento.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from AccesoADatos import (
    DIGIT_CLASS_NAMES,
    DatasetSummary,
    load_canvas_digit_dataset,
)
try:
    # La importación es diferida: solo abrimos la interfaz tras entrenar.
    from canvas_app import launch_canvas_app
except Exception:  # pragma: no cover - tkinter puede faltar en algunos entornos
    launch_canvas_app = None  # type: ignore[assignment]

RANDOM_STATE = 42
IMAGE_SIZE: Tuple[int, int] = (200, 200)
EPOCHS = 20
BATCH_SIZE = 32
MODEL_DIR = Path("modelos")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "canvas_digit_cnn.keras"
METRICS_PATH = MODEL_DIR / "metricas_entrenamiento.json"


@dataclass
class TrainingArtifacts:
    """Contenedor con los resultados clave del entrenamiento."""

    model: keras.Model
    metrics: Dict[str, List[float]]
    summary: DatasetSummary


def build_canvas_digit_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """Create a simple convolutional neural network for digit recognition."""

    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.05),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            data_augmentation,
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="canvas_digit_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def summarise_dataset(summary: DatasetSummary) -> None:
    """Print a short report about the dataset composition."""

    print("Resumen del dataset:")
    print(f"Total de imágenes: {summary.image_count}")
    for class_name, count in zip(DIGIT_CLASS_NAMES, summary.class_distribution):
        print(f" - {class_name}: {count}")


def _format_distribution(summary: DatasetSummary) -> List[str]:
    return [f" • {class_name}: {count}" for class_name, count in zip(DIGIT_CLASS_NAMES, summary.class_distribution)]


def format_training_overview(
    summary: DatasetSummary, metrics: Dict[str, List[float]]
) -> str:
    """Genera un informe legible con los resultados más relevantes."""

    lines = [
        "Total de imágenes: {summary.image_count}".format(summary=summary),
        "Distribución por clase:",
        *_format_distribution(summary),
        "",
    ]

    if metrics.get("accuracy"):
        lines.append(
            f"Exactitud en entrenamiento (última época): {metrics['accuracy'][-1]:.2%}"
        )
    if metrics.get("val_accuracy"):
        best_val_acc = max(metrics["val_accuracy"])
        best_epoch = metrics["val_accuracy"].index(best_val_acc) + 1
        lines.append(
            f"Mejor exactitud en validación: {best_val_acc:.2%} (época {best_epoch})"
        )
    if metrics.get("val_loss"):
        best_val_loss = min(metrics["val_loss"])
        best_loss_epoch = metrics["val_loss"].index(best_val_loss) + 1
        lines.append(
            f"Menor pérdida en validación: {best_val_loss:.4f} (época {best_loss_epoch})"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena la red neuronal de dígitos y, opcionalmente, abre la interfaz "
            "gráfica para probar el modelo entrenado."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sin-interfaz",
        action="store_true",
        help="Evita abrir el canvas tras finalizar el entrenamiento.",
    )
    group.add_argument(
        "--solo-interfaz",
        action="store_true",
        help="Omite el entrenamiento y abre directamente la interfaz (requiere modelo previo).",
    )
    return parser.parse_args()


def train_model() -> TrainingArtifacts:
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "Dataset"

    print("Cargando imágenes...")
    images, labels, class_names, summary = load_canvas_digit_dataset(
        dataset_dir, image_size=IMAGE_SIZE
    )
    summarise_dataset(summary)

    x_train, x_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=labels,
    )
    class_weight = {int(i): float(weight) for i, weight in enumerate(class_weight_values)}

    print("Preparando el modelo...")
    model = build_canvas_digit_model(images.shape[1:], len(class_names))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    print("Iniciando entrenamiento...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    print("Evaluando en el conjunto de validación...")
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Pérdida de validación: {val_loss:.4f}")
    print(f"Exactitud de validación: {val_accuracy:.4%}")

    print(f"Guardando el modelo en: {MODEL_PATH}")
    model.save(MODEL_PATH)

    training_metrics: Dict[str, List[float]] = {
        "loss": history.history["loss"],
        "accuracy": history.history["accuracy"],
        "val_loss": history.history["val_loss"],
        "val_accuracy": history.history["val_accuracy"],
    }
    metrics_payload = {
        **training_metrics,
        "dataset": {
            "total_imagenes": summary.image_count,
            "class_distribution": list(summary.class_distribution),
            "class_names": list(DIGIT_CLASS_NAMES),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False))
    print(f"Métricas almacenadas en: {METRICS_PATH}")
    return TrainingArtifacts(model=model, metrics=training_metrics, summary=summary)


def _load_summary_from_metrics() -> str | None:
    if not METRICS_PATH.exists():
        return None

    try:
        payload = json.loads(METRICS_PATH.read_text())
    except json.JSONDecodeError:
        return None

    dataset_info = payload.get("dataset")
    if not dataset_info:
        return None

    summary = DatasetSummary(
        image_count=int(dataset_info.get("total_imagenes", 0)),
        class_distribution=tuple(
            int(x) for x in dataset_info.get("class_distribution", [])
        ),
    )

    metrics: Dict[str, List[float]] = {
        "loss": payload.get("loss", []) or [],
        "accuracy": payload.get("accuracy", []) or [],
        "val_loss": payload.get("val_loss", []) or [],
        "val_accuracy": payload.get("val_accuracy", []) or [],
    }

    return format_training_overview(summary, metrics)


def _train_and_prepare_summary() -> Tuple[keras.Model, str]:
    artifacts = train_model()
    summary_text = format_training_overview(artifacts.summary, artifacts.metrics)
    return artifacts.model, summary_text


def main() -> None:
    args = parse_args()

    if args.solo_interfaz:
        if launch_canvas_app is None:
            raise RuntimeError(
                "tkinter no está disponible en este entorno, no es posible abrir la interfaz."
            )
        print("Abriendo la interfaz gráfica usando el modelo almacenado...")
        training_summary = _load_summary_from_metrics()
        launch_canvas_app(
            model=None,
            training_summary=training_summary,
            retrain_callback=_train_and_prepare_summary,
        )
        return

    trained_model, training_summary = _train_and_prepare_summary()

    print("\nResumen del entrenamiento actual:\n")
    print(training_summary)

    if args.sin_interfaz:
        return

    if launch_canvas_app is None:
        print(
            "Entrenamiento finalizado, pero tkinter no está disponible para abrir la "
            "interfaz gráfica."
        )
        return

    print("Entrenamiento completado. Iniciando la interfaz de dibujo...")
    launch_canvas_app(
        model=trained_model,
        training_summary=training_summary,
        retrain_callback=_train_and_prepare_summary,
    )
if __name__ == "__main__":
    # Limit TensorFlow logs to keep the output tidy.
    tf.get_logger().setLevel("INFO")
    main()
