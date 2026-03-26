"""
Unified Logging Utilities

Provides consistent logging across the entire pipeline with:
- Color-coded output for terminal
- Structured step logging
- Integration with Python's logging module

Usage:
    from src.utils.logging import get_logger, log_step, log_info, log_ok, log_warn, log_fail

    # Simple logging
    log_step("Training model")
    log_info("Loading dataset...")
    log_ok("Training completed!")
    log_warn("Low memory warning")
    log_fail("Critical error occurred")

    # Standard Python logger
    logger = get_logger(__name__)
    logger.info("Standard logging message")
"""

import logging
import sys
from datetime import datetime
from typing import Optional


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Standard colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


def _supports_color() -> bool:
    """Check if terminal supports color output."""
    if not hasattr(sys.stdout, 'isatty'):
        return False
    if not sys.stdout.isatty():
        return False
    return True


_USE_COLOR = _supports_color()


def _colorize(text: str, color: str) -> str:
    """Apply color to text if supported."""
    if _USE_COLOR:
        return f"{color}{text}{Colors.RESET}"
    return text


def _timestamp() -> str:
    """Return a consistent local timestamp for console output."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _emit(level_tag: str, message: str, color: str) -> None:
    """Emit a single professional console log line."""
    prefix = f"{_timestamp()} | {level_tag:<5}"
    print(f"{_colorize(prefix, color)} | {message}")


def log_step(message: str) -> None:
    """Log a major step/section header."""
    separator = "-" * 70
    print()
    _emit("STEP", separator, Colors.CYAN)
    _emit("STEP", message, Colors.BOLD + Colors.CYAN)
    _emit("STEP", separator, Colors.CYAN)


def log_info(message: str) -> None:
    """Log an informational message."""
    _emit("INFO", message, Colors.BLUE)


def log_ok(message: str) -> None:
    """Log a success message."""
    _emit("OK", message, Colors.GREEN)


def log_warn(message: str) -> None:
    """Log a warning message."""
    _emit("WARN", message, Colors.YELLOW)


def log_fail(message: str) -> None:
    """Log an error/failure message."""
    _emit("FAIL", message, Colors.RED)


def log_debug(message: str) -> None:
    """Log a debug message."""
    _emit("DEBUG", message, Colors.DIM)


def log_progress(current: int, total: int, message: str = "") -> None:
    """Log a progress update."""
    percentage = (current / total) * 100 if total > 0 else 0
    bar_length = 30
    filled_length = int(bar_length * current // total) if total > 0 else 0
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    prefix = _colorize(f"{_timestamp()} | PROG ", Colors.CYAN)
    pct = _colorize(f"{percentage:5.1f}%", Colors.GREEN)
    print(f"\r{prefix} | [{bar}] {pct} {message}", end='', flush=True)
    if current >= total:
        print()  # Newline at completion


class ColoredFormatter(logging.Formatter):
    """Logging formatter with color support."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM,
        logging.INFO: Colors.BLUE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BG_RED + Colors.WHITE,
    }

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if _USE_COLOR:
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def get_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist
    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Set format
        if format_string is None:
            format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

        formatter = ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


# Module-level logger
_module_logger: Optional[logging.Logger] = None


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup root logging configuration.

    Call this once at application startup.

    Args:
        level: Logging level for root logger

    Returns:
        Root logger instance
    """
    global _module_logger

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    _module_logger = logging.getLogger("llm_pipeline")
    _module_logger.setLevel(level)

    return _module_logger
