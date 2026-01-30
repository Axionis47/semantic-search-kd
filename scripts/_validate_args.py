"""Shared argument validation for scripts."""

import sys
from pathlib import Path


def validate_path_exists(path_str: str, label: str) -> None:
    """Validate that a file or directory path exists."""
    if not Path(path_str).exists():
        print(f"Error: {label} does not exist: {path_str}", file=sys.stderr)
        sys.exit(1)


def validate_positive_int(value: int, label: str) -> None:
    """Validate that an integer is positive."""
    if value <= 0:
        print(f"Error: {label} must be positive, got {value}", file=sys.stderr)
        sys.exit(1)


def validate_positive_float(value: float, label: str) -> None:
    """Validate that a float is positive."""
    if value <= 0:
        print(f"Error: {label} must be positive, got {value}", file=sys.stderr)
        sys.exit(1)


def validate_port(port: int) -> None:
    """Validate port number range."""
    if not (1 <= port <= 65535):
        print(f"Error: port must be in range [1, 65535], got {port}", file=sys.stderr)
        sys.exit(1)


def validate_device(device: str) -> None:
    """Validate device string."""
    if device not in ("cpu",) and not device.startswith("cuda"):
        print(f"Error: device must be 'cpu' or 'cuda[:N]', got '{device}'", file=sys.stderr)
        sys.exit(1)
