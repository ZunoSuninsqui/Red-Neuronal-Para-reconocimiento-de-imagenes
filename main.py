"""Script principal para entrenar una red neuronal con el dataset de dígitos.

El objetivo es reconocer números dibujados a mano en un canvas.  Para facilitar
su uso desde la línea de comandos, basta con ejecutar ``python main.py`` y el
script se encargará de:

1. Cargar todas las imágenes de la carpeta :mod:`Dataset`.
2. Normalizarlas a un tamaño uniforme y dividirlas en conjuntos de entrenamiento
   y validación.
3. Entrenar una red neuronal convolucional (CNN) utilizando TensorFlow/Keras.
4. Guardar el modelo entrenado, mostrar métricas básicas de rendimiento y abrir
   una interfaz Tkinter con un canvas para dibujar un número y obtener su
   predicción en tiempo real.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk

from PIL import Image, ImageDraw

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
CANVAS_SIZE = 280  # Pixels for the Tkinter drawing canvas
STROKE_WIDTH = 12


def _start_prediction_gui(
    model: keras.Model,
    image_size: Tuple[int, int],
    class_names: Tuple[str, ...],
) -> None:
    """Launch a Tkinter window to draw digits and obtain predictions."""

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - requires display environment
        print(
            "No fue posible iniciar la interfaz gráfica de Tkinter. "
            "Asegúrate de ejecutar el script en un entorno con servidor gráfico.",
        )
        print(f"Detalle: {exc}")
        return
    root.title("Prueba de la red neuronal")

    info_label = tk.Label(
        root,
        text="Dibuja un número en el lienzo y pulsa 'Predecir' para obtener el resultado",
    )
    info_label.pack(pady=(10, 5))

    canvas_frame = tk.Frame(root)
    canvas_frame.pack(padx=10, pady=5)

    canvas = tk.Canvas(
        canvas_frame,
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        bg="white",
        borderwidth=2,
        relief="ridge",
    )
    canvas.pack()

    pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
    drawer = ImageDraw.Draw(pil_image)
    last_point = {"x": None, "y": None}

    def _reset_last_point(_: tk.Event) -> None:
        last_point["x"] = None
        last_point["y"] = None

    def _draw(event: tk.Event) -> None:
        x, y = event.x, event.y
        if last_point["x"] is None:
            last_point["x"], last_point["y"] = x, y
            return

        canvas.create_line(
            last_point["x"],
            last_point["y"],
            x,
            y,
            fill="black",
            width=STROKE_WIDTH,
            capstyle=tk.ROUND,
            smooth=True,
        )
        drawer.line(
            [(last_point["x"], last_point["y"]), (x, y)],
            fill=0,
            width=STROKE_WIDTH,
        )
        last_point["x"], last_point["y"] = x, y

    canvas.bind("<ButtonRelease-1>", _reset_last_point)
    canvas.bind("<B1-Motion>", _draw)

    result_var = tk.StringVar(value="Resultado: ")
    result_label = tk.Label(root, textvariable=result_var, font=("Helvetica", 14))
    result_label.pack(pady=10)

    def _predict() -> None:
        resized = pil_image.resize(image_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(resized, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = image_array[np.newaxis, ..., np.newaxis]

        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])
        result_var.set(
            f"Resultado: {class_names[predicted_index]} (confianza {confidence:.1%})"
        )

    def _clear_canvas() -> None:
        canvas.delete("all")
        drawer.rectangle([(0, 0), (CANVAS_SIZE, CANVAS_SIZE)], fill=255)
        result_var.set("Resultado: ")

    button_frame = tk.Frame(root)
    button_frame.pack(pady=(0, 10))

    predict_button = tk.Button(button_frame, text="Predecir", command=_predict)
    predict_button.grid(row=0, column=0, padx=5)

    clear_button = tk.Button(button_frame, text="Limpiar", command=_clear_canvas)
    clear_button.grid(row=0, column=1, padx=5)

    exit_button = tk.Button(button_frame, text="Cerrar", command=root.destroy)
    exit_button.grid(row=0, column=2, padx=5)

    root.mainloop()


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
    class_names = tuple(class_names)
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

    print("Abriendo la interfaz gráfica para realizar pruebas en el canvas...")
    _start_prediction_gui(model, IMAGE_SIZE, class_names)


if __name__ == "__main__":
    # Limit TensorFlow logs to keep the output tidy.
    tf.get_logger().setLevel("INFO")
    main()
