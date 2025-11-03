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

import json
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from AccesoADatos import (
    DIGIT_CLASS_NAMES,
    DatasetSummary,
    load_canvas_digit_dataset,
)

RANDOM_STATE = 42
IMAGE_SIZE: Tuple[int, int] = (64, 64)
EPOCHS = 20
BATCH_SIZE = 32
MODEL_DIR = Path("modelos")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "canvas_digit_cnn.keras"
METRICS_PATH = MODEL_DIR / "metricas_entrenamiento.json"


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


def main() -> None:
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
        callbacks=callbacks,
        verbose=2,
    )

    print("Evaluando en el conjunto de validación...")
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Pérdida de validación: {val_loss:.4f}")
    print(f"Exactitud de validación: {val_accuracy:.4%}")

    print(f"Guardando el modelo en: {MODEL_PATH}")
    model.save(MODEL_PATH)

    training_metrics = {
        "loss": history.history["loss"],
        "accuracy": history.history["accuracy"],
        "val_loss": history.history["val_loss"],
        "val_accuracy": history.history["val_accuracy"],
    }
    METRICS_PATH.write_text(json.dumps(training_metrics, indent=2, ensure_ascii=False))
    print(f"Métricas almacenadas en: {METRICS_PATH}")


if __name__ == "__main__":
    # Limit TensorFlow logs to keep the output tidy.
    tf.get_logger().setLevel("INFO")
    main()
