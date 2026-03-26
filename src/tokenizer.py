"""Backward-compatible import shim for the canonical tokenizer module."""

from .core.tokenizer import BPETokenizer

__all__ = ["BPETokenizer"]

