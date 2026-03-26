"""Backward-compatible import shim for the canonical enhanced model module."""

from .core.models.enhanced_model import EnhancedGPTModel, EnhancedModelConfig

ModelConfig = EnhancedModelConfig

__all__ = ["EnhancedGPTModel", "EnhancedModelConfig", "ModelConfig"]

