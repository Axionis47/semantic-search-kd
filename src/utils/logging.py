"""Logging configuration and utilities."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "10 days",
) -> None:
    """
    Configure loguru logger with console and optional file output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        rotation: Log rotation size/time
        retention: How long to keep old logs
    """
    # Remove default handler
    logger.remove()

    # Add console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging configured: level={log_level}, file={log_file}")


def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)

