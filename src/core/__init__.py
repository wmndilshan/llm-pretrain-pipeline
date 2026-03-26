"""
Core training components.

This package contains:
- models: Model architectures (base, enhanced, professional)
- trainer: Training loop with gradient accumulation, AMP, checkpointing
- tokenizer: Legacy BPE tokenizer implementation
- tokenization: Unified tokenizer manager for merged pipeline profiles
- dataset: Memory-mapped dataset loading
"""

from .models import (
    GPTModel,
    EnhancedGPTModel,
    ProfessionalTransformerModel,
    ModelArchitectureConfig,
    ModelConfig,
    get_model,
    get_model_from_config,
    list_models,
)
from .trainer import Trainer, CosineWarmupScheduler
from .tokenizer import BPETokenizer
from .tokenization import TokenizerManager
from .dataset import TokenDataset, create_dataloader, InfiniteDataLoader

__all__ = [
    'GPTModel',
    'EnhancedGPTModel',
    'ProfessionalTransformerModel',
    'ModelArchitectureConfig',
    'ModelConfig',
    'get_model',
    'get_model_from_config',
    'list_models',
    'Trainer',
    'CosineWarmupScheduler',
    'BPETokenizer',
    'TokenizerManager',
    'TokenDataset',
    'create_dataloader',
    'InfiniteDataLoader',
]
