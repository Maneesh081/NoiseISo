"""Small shared helpers."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make random number generation reproducible for this run.

    It reseeds Python's random, NumPy and PyTorch (cpu + every GPU), and tells
    cuDNN to prefer deterministic convolutions. This is what lets you get the
    same results on a second run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
