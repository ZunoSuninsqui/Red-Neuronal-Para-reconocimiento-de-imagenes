"""Utilities to seed every random number generator used in training."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow optional during tests
    tf = None  # type: ignore[assignment]


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow to obtain reproducible runs.

    Parameters
    ----------
    seed:
        Integer seed shared across libraries.
    deterministic:
        When ``True`` (the default) enables TensorFlow deterministic
        operations whenever the backend supports them.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if tf is None:
        return

    tf.random.set_seed(seed)
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - depending on TF version
            pass
