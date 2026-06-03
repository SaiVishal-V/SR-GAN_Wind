"""
Logging setup for WindGapGAN.

Provides consistent, configurable logging across all modules.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
    log_filename: str = "windgapgan.log",
) -> logging.Logger:
    """
    Configure the root logger with console and optional file handlers.

    Args:
        log_dir: Directory for log files. If None, only console logging is used.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_filename: Name of the log file.

    Returns:
        The configured root logger.
    """
    root_logger = logging.getLogger()

    # Avoid duplicate handlers on re-initialization
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)-40s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Use this instead of logging.getLogger() directly
    to ensure consistent naming conventions.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)
