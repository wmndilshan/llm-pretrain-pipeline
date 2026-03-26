"""
LLM Training Pipeline

A professional, modular pipeline for training transformer language models.

Package Structure:
- core/: Core training components (models, trainer, tokenizer, dataset)
- pipeline/: Pipeline orchestration (preprocessing, checkpoint, modal_training)
- orchestration/: Scheduling and budget management
- inference/: Inference server
- utils/: Shared utilities (config, logging)

Quick Start:
    from src.core.models import get_model, ModelConfig, list_models
    from src.core.trainer import Trainer
    from src.pipeline.preprocessing import DataPreprocessor

    # List available model architectures
    print(list_models())  # ['base', 'enhanced', 'professional']

    # Create model
    config = ModelConfig(architecture='enhanced', vocab_size=10000)
    model = get_model('enhanced', config)
"""

__version__ = "2.0.0"

# Convenience imports
from .core.models import (
    GPTModel,
    EnhancedGPTModel,
    ProfessionalTransformerModel,
    ModelArchitectureConfig,
    ModelConfig,
    get_model,
    get_model_from_config,
    list_models,
    ModelVersionManager,
)
from .core.trainer import Trainer, CosineWarmupScheduler
from .core.tokenizer import BPETokenizer
from .core.tokenization import TokenizerManager
from .core.dataset import TokenDataset, create_dataloader, InfiniteDataLoader

from .pipeline.preprocessing import DataPreprocessor, PreprocessingState
from .pipeline.checkpoint import CheckpointManager, CheckpointMetadata

from .orchestration.scheduler import MonthlyScheduler
from .orchestration.budget_tracker import BudgetTracker

from .utils.config import load_env, load_yaml_config, get_config
from .utils.logging import log_step, log_info, log_ok, log_warn, log_fail

__all__ = [
    # Version
    '__version__',

    # Models
    'GPTModel',
    'EnhancedGPTModel',
    'ProfessionalTransformerModel',
    'ModelArchitectureConfig',
    'ModelConfig',
    'get_model',
    'get_model_from_config',
    'list_models',
    'ModelVersionManager',

    # Training
    'Trainer',
    'CosineWarmupScheduler',

    # Data
    'BPETokenizer',
    'TokenizerManager',
    'TokenDataset',
    'create_dataloader',
    'InfiniteDataLoader',
    'DataPreprocessor',
    'PreprocessingState',

    # Checkpoint
    'CheckpointManager',
    'CheckpointMetadata',

    # Orchestration
    'MonthlyScheduler',
    'BudgetTracker',

    # Utils
    'load_env',
    'load_yaml_config',
    'get_config',
    'log_step',
    'log_info',
    'log_ok',
    'log_warn',
    'log_fail',
]
