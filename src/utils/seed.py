"""Reproducibility utilities for seeding random number generators."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for deterministic behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Enable deterministic algorithms in PyTorch
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_seed_from_config(config: dict) -> int:
    """Extract seed from config dict."""
    return config.get("seed", 42)

