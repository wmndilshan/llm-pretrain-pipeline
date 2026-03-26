"""Backward-compatible import shim for the canonical checkpoint module."""

from .pipeline.checkpoint import CheckpointManager, CheckpointMetadata

__all__ = ["CheckpointManager", "CheckpointMetadata"]

