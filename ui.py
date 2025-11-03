"""Tkinter user interface for the digit recognition project."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageDraw, ImageTk

from data_loader import DatasetScanner, DatasetSummary, Sample
from model import ModelMetadata, load_model, save_model
from predict import predict_tensor
from preprocess import run_pipeline
from train import TrainingConfig, TrainingResult, train_model
from utils import enqueue_log, forward_exception, get_device, queue_log_writer, set_seed

BRUSH_WIDTH = 20
CANVAS_SIZE = 280
PREVIEW_SIZE = 140


class DigitRecognizerApp:
    """Encapsulates the Tkinter GUI and background tasks."""

    def __init__(self) -> None:
        set_seed()
        self.root = tk.Tk()
        self.root.title("Reconocimiento de Dígitos 1-9")
        self.device = get_device()

        self.log_queue: "queue.Queue[dict]" = queue.Queue()
        self.training_thread: Optional[threading.Thread] = None
        self.training_cancel_event: Optional[threading.Event] = None
        self.training_config = TrainingConfig()

        self.dataset_samples: list[Sample] = []
        self.dataset_summary: Optional[DatasetSummary] = None

        self.model = None
        self.metadata: Optional[ModelMetadata] = None
        self.latest_processed_image = None
        self.latest_tensor = None

        self._init_ui()
        self._setup_logging()

    # ------------------------------------------------------------------ UI SETUP
    def _init_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        self.canvas = tk.Canvas(
            main_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            relief="solid",
            borderwidth=1,
        )
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        self.canvas.bind("<ButtonPress-1>", self._on_draw_start)
        self.canvas.bind("<B1-Motion>", self._on_draw)
        self.canvas.bind("<ButtonRelease-1>", self._on_draw_end)

        self.drawing_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw_context = ImageDraw.Draw(self.drawing_image)
        self.last_point: Optional[tuple[int, int]] = None

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)

        preview_label = ttk.Label(right_frame, text="Vista previa (28×28)")
        preview_label.grid(row=0, column=0, sticky="w")

        self.preview_canvas = ttk.Label(right_frame)
        self.preview_canvas.grid(row=1, column=0, sticky="nsew")

        self.prediction_var = tk.StringVar(value="Predicción: -")
        self.confidence_var = tk.StringVar(value="Confianza: -")
        prediction_label = ttk.Label(
            right_frame, textvariable=self.prediction_var, font=("Arial", 20, "bold")
        )
        prediction_label.grid(row=2, column=0, pady=(10, 0), sticky="w")
        confidence_label = ttk.Label(right_frame, textvariable=self.confidence_var)
        confidence_label.grid(row=3, column=0, sticky="w")

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="ew")
        for idx in range(7):
            button_frame.columnconfigure(idx, weight=1)

        ttk.Button(button_frame, text="Borrar", command=self.clear_canvas).grid(
            row=0, column=0, padx=2, sticky="ew"
        )
        ttk.Button(button_frame, text="Preprocesar", command=self.preprocess_canvas).grid(
            row=0, column=1, padx=2, sticky="ew"
        )
        ttk.Button(button_frame, text="Predecir", command=self.predict_canvas_digit).grid(
            row=0, column=2, padx=2, sticky="ew"
        )
        ttk.Button(button_frame, text="Entrenar", command=self.start_training).grid(
            row=0, column=3, padx=2, sticky="ew"
        )
        ttk.Button(button_frame, text="Cancelar", command=self.cancel_training).grid(
            row=0, column=4, padx=2, sticky="ew"
        )
        ttk.Button(button_frame, text="Cargar Modelo", command=self.load_model_dialog).grid(
            row=0, column=5, padx=2, sticky="ew"
        )
        ttk.Button(button_frame, text="Guardar Modelo", command=self.save_model_dialog).grid(
            row=0, column=6, padx=2, sticky="ew"
        )

        ttk.Button(button_frame, text="Seleccionar Dataset", command=self.select_dataset).grid(
            row=1, column=0, columnspan=7, pady=(5, 0), sticky="ew"
        )

        log_label = ttk.Label(main_frame, text="Logs")
        log_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.log_widget = ScrolledText(main_frame, height=12, state="disabled")
        self.log_widget.grid(row=4, column=0, columnspan=2, sticky="nsew")

    def _setup_logging(self) -> None:
        self.root.after(100, self._process_log_queue)

    # ------------------------------------------------------------------ DRAWING
    def _on_draw_start(self, event: tk.Event) -> None:
        self.last_point = (event.x, event.y)

    def _on_draw(self, event: tk.Event) -> None:
        if self.last_point is None:
            self.last_point = (event.x, event.y)
            return
        x0, y0 = self.last_point
        x1, y1 = event.x, event.y
        self.canvas.create_line(x0, y0, x1, y1, width=BRUSH_WIDTH, fill="black", capstyle=tk.ROUND, smooth=True)
        self.draw_context.line((x0, y0, x1, y1), fill=0, width=BRUSH_WIDTH)
        self.last_point = (x1, y1)

    def _on_draw_end(self, _event: tk.Event) -> None:
        self.last_point = None

    # ------------------------------------------------------------------ ACTIONS
    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.drawing_image.paste(255, box=None)
        self.preview_canvas.configure(image="")
        self.preview_canvas.image = None
        self.latest_processed_image = None
        self.latest_tensor = None
        self.prediction_var.set("Predicción: -")
        self.confidence_var.set("Confianza: -")

    def preprocess_canvas(self) -> None:
        result = run_pipeline(self.drawing_image)
        self.latest_processed_image = result.image
        self.latest_tensor = result.tensor
        preview = result.image.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
        photo = ImageTk.PhotoImage(preview)
        self.preview_canvas.configure(image=photo)
        self.preview_canvas.image = photo
        enqueue_log(self.log_queue, "Imagen preprocesada correctamente.")

    def predict_canvas_digit(self) -> None:
        if self.model is None:
            messagebox.showinfo("Modelo no disponible", "Entrena o carga un modelo antes de predecir.")
            return
        if self.latest_tensor is None:
            self.preprocess_canvas()
        if self.latest_tensor is None:
            messagebox.showwarning("Imagen vacía", "Dibuja un dígito antes de predecir.")
            return
        label, confidence = predict_tensor(self.model, self.latest_tensor, self.device)
        self.prediction_var.set(f"Predicción: {label}")
        self.confidence_var.set(f"Confianza: {confidence:.2%}")
        enqueue_log(self.log_queue, f"Predicción: {label} (confianza {confidence:.2%})")

    def select_dataset(self) -> None:
        folder = filedialog.askdirectory(title="Seleccionar carpeta del dataset")
        if not folder:
            return
        path = Path(folder)
        scanner = DatasetScanner()
        samples, summary = scanner.scan(path)
        if not samples:
            messagebox.showerror("Dataset vacío", "No se encontraron imágenes válidas en la carpeta seleccionada.")
            return
        self.dataset_samples = samples
        self.dataset_summary = summary
        enqueue_log(self.log_queue, f"Dataset seleccionado: {path}")
        for digit in range(1, 10):
            count = summary.class_counts.get(digit, 0)
            enqueue_log(self.log_queue, f"Clase {digit}: {count} imágenes")
        if summary.total_images:
            enqueue_log(self.log_queue, f"Total de imágenes: {summary.total_images}")
        self._validate_dataset(show_dialog=True)

    def _validate_dataset(self, show_dialog: bool = False) -> bool:
        if self.dataset_summary is None:
            if show_dialog:
                messagebox.showwarning("Dataset no seleccionado", "Selecciona un dataset válido antes de entrenar.")
            return False
        missing = self.dataset_summary.missing_classes
        if missing:
            message = "Faltan clases en el dataset: " + ", ".join(map(str, missing))
            enqueue_log(self.log_queue, message, level="warning")
            if show_dialog:
                messagebox.showerror("Dataset incompleto", message)
            return False
        insufficient = {
            digit: count
            for digit, count in self.dataset_summary.class_counts.items()
            if count < self.training_config.min_images_per_class
        }
        if insufficient:
            message = "Clases con pocas imágenes: " + ", ".join(
                f"{digit} ({count})" for digit, count in insufficient.items()
            )
            enqueue_log(self.log_queue, message, level="warning")
            if show_dialog:
                proceed = messagebox.askyesno(
                    "Advertencia",
                    message + "\n¿Deseas continuar con el entrenamiento?",
                    icon="warning",
                )
                if not proceed:
                    return False
        return True

    def start_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Entrenamiento en curso", "Espera a que finalice el entrenamiento actual o cancélalo.")
            return
        if not self._validate_dataset(show_dialog=True):
            return
        if not self.dataset_samples:
            messagebox.showwarning("Dataset vacío", "Selecciona un dataset con imágenes válidas.")
            return
        enqueue_log(self.log_queue, "Iniciando entrenamiento...")
        self.training_cancel_event = threading.Event()
        self.training_thread = threading.Thread(target=self._run_training, daemon=True)
        self.training_thread.start()

    def cancel_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive() and self.training_cancel_event:
            self.training_cancel_event.set()
            enqueue_log(self.log_queue, "Solicitud de cancelación enviada.")

    def _run_training(self) -> None:
        try:
            log_writer = queue_log_writer(self.log_queue)
            result = train_model(
                self.dataset_samples,
                device=self.device,
                config=self.training_config,
                log_callback=log_writer,
                cancel_event=self.training_cancel_event,
            )
            status = "training_cancelled" if self.training_cancel_event and self.training_cancel_event.is_set() else "training_finished"
            self.log_queue.put({"type": "status", "payload": (status, result)})
        except Exception as exc:  # pragma: no cover - background thread
            forward_exception(self.log_queue, exc, prefix="Error durante el entrenamiento")
        finally:
            self.log_queue.put({"type": "status", "payload": ("training_done", None)})

    def load_model_dialog(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Cargar modelo",
            filetypes=(("Modelos PyTorch", "*.pt *.pth"), ("Todos", "*.*")),
        )
        if not file_path:
            return
        try:
            model, metadata = load_model(Path(file_path), device=self.device)
        except Exception as exc:  # pragma: no cover - file IO
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {exc}")
            return
        self.model = model
        self.metadata = metadata
        enqueue_log(self.log_queue, f"Modelo cargado desde {file_path}")

    def save_model_dialog(self) -> None:
        if self.model is None:
            messagebox.showinfo("Sin modelo", "Entrena o carga un modelo antes de guardar.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Guardar modelo",
            defaultextension=".pt",
            filetypes=(("Modelos PyTorch", "*.pt"), ("Modelos PyTorch", "*.pth"), ("Todos", "*.*")),
        )
        if not file_path:
            return
        try:
            save_model(Path(file_path), self.model, self.metadata or ModelMetadata())
        except Exception as exc:  # pragma: no cover - file IO
            messagebox.showerror("Error", f"No se pudo guardar el modelo: {exc}")
            return
        enqueue_log(self.log_queue, f"Modelo guardado en {file_path}")

    # ------------------------------------------------------------------ LOGGING
    def _process_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item["type"] == "log":
                    self._append_log(item["payload"].level.upper(), item["payload"].message)
                elif item["type"] == "error":
                    self._append_log("ERROR", item["payload"])
                elif item["type"] == "status":
                    status, payload = item["payload"]
                    if status == "training_finished" and isinstance(payload, TrainingResult):
                        self._handle_training_result(payload)
                    elif status == "training_cancelled":
                        self._append_log("INFO", "Entrenamiento cancelado.")
                    if status in {"training_finished", "training_cancelled", "training_done"}:
                        self.training_thread = None
                        self.training_cancel_event = None
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_log_queue)

    def _append_log(self, level: str, message: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", f"[{level}] {message}\n")
        self.log_widget.configure(state="disabled")
        self.log_widget.see("end")

    def _handle_training_result(self, result: TrainingResult) -> None:
        self.model = result.model
        self.metadata = result.metadata
        enqueue_log(self.log_queue, "Entrenamiento finalizado.")
        last_acc = result.history.get("val_accuracy", [])
        if last_acc:
            enqueue_log(self.log_queue, f"Exactitud validación final: {last_acc[-1]:.2%}")
        enqueue_log(self.log_queue, f"Matriz de confusión:\n{result.confusion}")

    # ------------------------------------------------------------------ PUBLIC
    def run(self) -> None:
        self.root.mainloop()
