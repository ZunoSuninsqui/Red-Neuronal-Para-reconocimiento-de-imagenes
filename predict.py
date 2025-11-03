"""Inference helpers for the digit recogniser."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from preprocess import preprocess_image, image_to_tensor


def predict_tensor(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> Tuple[int, float]:
    """Return the predicted class index (1-9) and confidence."""

    model.eval()
    with torch.no_grad():
        tensor = tensor.unsqueeze(0).to(device)
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(dim=1)
    return predicted.item() + 1, confidence.item()


def predict_image(model: torch.nn.Module, image, device: torch.device) -> Tuple[int, float]:
    tensor = image_to_tensor(preprocess_image(image))
    return predict_tensor(model, tensor, device)
