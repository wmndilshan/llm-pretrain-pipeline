"""Backward-compatible import shim for the canonical trainer module."""

from .core.trainer import CosineWarmupScheduler, Trainer

__all__ = ["Trainer", "CosineWarmupScheduler"]

