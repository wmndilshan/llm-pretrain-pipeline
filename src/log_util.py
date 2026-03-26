"""Backward-compatible import shim for centralized logging helpers."""

from .utils.logging import (
    get_logger,
    log_debug,
    log_fail,
    log_info,
    log_ok,
    log_step,
    log_warn,
    setup_logging,
)

step = log_step
info = log_info
ok = log_ok
warn = log_warn
fail = log_fail
debug = log_debug

__all__ = [
    "get_logger",
    "setup_logging",
    "log_debug",
    "log_step",
    "log_info",
    "log_ok",
    "log_warn",
    "log_fail",
    "step",
    "info",
    "ok",
    "warn",
    "fail",
    "debug",
]
