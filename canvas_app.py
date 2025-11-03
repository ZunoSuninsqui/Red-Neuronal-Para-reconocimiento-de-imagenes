from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras
from tkinter import (
    BOTH,
    LEFT,
    RIGHT,
    Tk,
    Canvas,
    Frame,
    Label,
    StringVar,
    messagebox,
    ttk,
)

from AccesoADatos import DIGIT_CLASS_NAMES

MODEL_PATH = Path(__file__).resolve().parent / "modelos" / "canvas_digit_cnn.keras"
CANVAS_SIZE = 200
BRUSH_RADIUS = 10
MODEL_INPUT_SIZE: Tuple[int, int] = (200, 200)


@dataclass(frozen=True)
class Prediction:
    """Representa la probabilidad para un dígito concreto."""

    label: str
    probability: float


class CanvasDigitRecognizer:
    """Interfaz principal para dibujar y clasificar dígitos."""

    def __init__(
        self,
        root: Tk,
        *,
        model: keras.Model | None = None,
        training_summary: str | None = None,
        retrain_callback: Callable[[], Tuple[keras.Model, str]] | None = None,
    ) -> None:
        self.root = root
        root.title("Reconocimiento de dígitos")
        root.resizable(False, False)

        self.model = model if model is not None else self._load_model(MODEL_PATH)
        self._training_summary_text = training_summary
        self._retrain_callback = retrain_callback
        self._retraining_thread: threading.Thread | None = None
        self._init_widgets()

    # ------------------------------------------------------------------
    # Configuración de la interfaz
    def _init_widgets(self) -> None:
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=BOTH, expand=True)

        self.canvas = Canvas(
            main_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self.canvas.pack(side=LEFT, padx=(0, 10))

        controls = Frame(main_frame)
        controls.pack(side=RIGHT, fill=BOTH)

        info_label = Label(
            controls,
            text=(
                "Dibuja un número entre 0 y 9 con el botón izquierdo del ratón.\n"
                "Pulsa en *Predecir* para que la red neuronal intente reconocerlo."
            ),
            justify="left",
            wraplength=220,
        )
        info_label.pack(pady=(0, 10))

        button_frame = Frame(controls)
        button_frame.pack(pady=(0, 10))

        ttk.Button(button_frame, text="Predecir", command=self.predict_digit).pack(
            side=LEFT, padx=(0, 5)
        )
        ttk.Button(button_frame, text="Limpiar", command=self.clear_canvas).pack(
            side=LEFT, padx=(0, 5)
        )
        if self._retrain_callback is not None:
            ttk.Button(
                button_frame,
                text="Reentrenar modelo",
                command=self.retrain_model,
            ).pack(side=LEFT)

        self.prediction_var = StringVar(value="Predicción: —")
        prediction_label = Label(controls, textvariable=self.prediction_var, font=("Arial", 16))
        prediction_label.pack(pady=(5, 10))

        self.probabilities_var = StringVar(value="")
        probabilities_label = Label(
            controls,
            textvariable=self.probabilities_var,
            justify="left",
            font=("Courier New", 10),
        )
        probabilities_label.pack(anchor="w")

        self.status_var = StringVar(value="")
        status_label = Label(
            controls,
            textvariable=self.status_var,
            justify="left",
            foreground="#555555",
        )
        status_label.pack(anchor="w", pady=(5, 0))

        self.progress = ttk.Progressbar(controls, mode="indeterminate")

        # Imagen auxiliar donde replicamos lo dibujado en el canvas usando Pillow.
        self._drawing_surface = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self._draw = ImageDraw.Draw(self._drawing_surface)

        self._last_position: Tuple[int, int] | None = None
        self.canvas.bind("<ButtonPress-1>", self._start_drawing)
        self.canvas.bind("<B1-Motion>", self._draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drawing)

        Label(controls, text="Resumen del entrenamiento", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(15, 0)
        )
        self.training_summary_var = StringVar(
            value=self._training_summary_text
            or "No hay información de entrenamiento disponible."
        )
        summary_label = Label(
            controls,
            textvariable=self.training_summary_var,
            justify="left",
            font=("Courier New", 9),
            wraplength=220,
        )
        summary_label.pack(anchor="w", pady=(0, 10))

    # ------------------------------------------------------------------
    # Lógica de inferencia
    @staticmethod
    def _load_model(model_path: Path) -> keras.Model:
        if not model_path.exists():
            raise FileNotFoundError(
                "No se encontró el modelo entrenado. Ejecuta primero 'python main.py' "
                "para entrenarlo y generarlo en la carpeta 'modelos'."
            )
        return keras.models.load_model(model_path)

    @staticmethod
    def _format_probabilities(predictions: Iterable[Prediction]) -> str:
        return "\n".join(
            f"{prediction.label:>12}: {prediction.probability:6.2%}"
            for prediction in predictions
        )

    def predict_digit(self) -> None:
        image = self._prepare_image_for_model()
        probabilities = self.model.predict(image, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])

        self.prediction_var.set(
            f"Predicción: {best_index} ({confidence:.1%})"
        )

        predictions = [
            Prediction(label=name, probability=float(prob))
            for name, prob in zip(DIGIT_CLASS_NAMES, probabilities)
        ]
        self.probabilities_var.set(self._format_probabilities(predictions))

    def retrain_model(self) -> None:
        if self._retrain_callback is None:
            return
        if self._retraining_thread is not None and self._retraining_thread.is_alive():
            return

        self.status_var.set("Reentrenando modelo, por favor espera...")
        if not self.progress.winfo_ismapped():
            self.progress.pack(fill="x", pady=(2, 10))
        self.progress.start(10)

        self._retraining_thread = threading.Thread(
            target=self._execute_retraining,
            daemon=True,
        )
        self._retraining_thread.start()

    def _execute_retraining(self) -> None:
        try:
            model, summary_text = self._retrain_callback()  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - feedback visual
            self.root.after(0, self._handle_retraining_error, exc)
        else:
            self.root.after(0, self._handle_retraining_success, model, summary_text)

    def _handle_retraining_success(self, model: keras.Model, summary_text: str) -> None:
        self.model = model
        self.training_summary_var.set(summary_text)
        self.status_var.set("Reentrenamiento completado correctamente.")
        self._finish_retraining()

    def _handle_retraining_error(self, exc: Exception) -> None:
        self.status_var.set("Error durante el reentrenamiento.")
        messagebox.showerror("Error de entrenamiento", str(exc))
        self._finish_retraining()

    def _finish_retraining(self) -> None:
        self.progress.stop()
        if self.progress.winfo_ismapped():
            self.progress.pack_forget()
        self._retraining_thread = None

    def _prepare_image_for_model(self) -> np.ndarray:
        """Convierte la imagen dibujada en el formato esperado por la CNN."""

        resized = self._drawing_surface.resize(
            MODEL_INPUT_SIZE,
            resample=Image.Resampling.NEAREST,
        )
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        image_array = image_array.reshape((1, MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 1))
        return image_array

    # ------------------------------------------------------------------
    # Gestión del dibujo sobre el canvas
    def _start_drawing(self, event) -> None:  # type: ignore[override]
        self._last_position = (event.x, event.y)
        self._draw_dot(event.x, event.y)

    def _draw_stroke(self, event) -> None:  # type: ignore[override]
        if self._last_position is None:
            self._last_position = (event.x, event.y)
        x0, y0 = self._last_position
        x1, y1 = event.x, event.y
        self.canvas.create_line(
            x0,
            y0,
            x1,
            y1,
            fill="black",
            width=BRUSH_RADIUS * 2,
            capstyle="round",
            smooth=True,
        )
        self._draw.line([(x0, y0), (x1, y1)], fill=0, width=BRUSH_RADIUS * 2)
        self._last_position = (x1, y1)

    def _draw_dot(self, x: int, y: int) -> None:
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self._draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=0)

    def _stop_drawing(self, _event) -> None:  # type: ignore[override]
        self._last_position = None

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self._drawing_surface.paste(255, (0, 0, CANVAS_SIZE, CANVAS_SIZE))
        self.probabilities_var.set("")
        self.prediction_var.set("Predicción: —")


def launch_canvas_app(
    model: keras.Model | None = None,
    *,
    training_summary: str | None = None,
    retrain_callback: Callable[[], Tuple[keras.Model, str]] | None = None,
) -> None:
    """Inicia la interfaz gráfica, reutilizando un modelo ya entrenado si se proporciona."""

    root = Tk()
    CanvasDigitRecognizer(
        root,
        model=model,
        training_summary=training_summary,
        retrain_callback=retrain_callback,
    )
    root.mainloop()


if __name__ == "__main__":
    launch_canvas_app()
