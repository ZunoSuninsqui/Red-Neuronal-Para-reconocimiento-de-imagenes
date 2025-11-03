"""Utilities for logging, reproducibility and device selection."""
from __future__ import annotations

import logging
import queue
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch


LOGGER_NAME = "digit_app"


@dataclass
class LogEntry:
    """Represents a message that should be displayed in the UI log."""

    level: str
    message: str


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a module level logger."""

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_device() -> torch.device:
    """Return the optimal device available for computation."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Seed every relevant random number generator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enqueue_log(
    output_queue: "queue.Queue[dict]", message: str, level: str = "info"
) -> None:
    """Push a log message into the UI queue."""

    output_queue.put({"type": "log", "payload": LogEntry(level=level, message=message)})


def forward_exception(
    output_queue: "queue.Queue[dict]", exc: BaseException, prefix: Optional[str] = None
) -> None:
    """Send a formatted exception to the UI queue."""

    message = f"{prefix}: {exc}" if prefix else str(exc)
    output_queue.put({"type": "error", "payload": message})


def queue_log_writer(output_queue: "queue.Queue[dict]") -> Callable[[str], None]:
    """Return a callback that enqueues log strings."""

    def _writer(message: str) -> None:
        enqueue_log(output_queue, message)

    return _writer
