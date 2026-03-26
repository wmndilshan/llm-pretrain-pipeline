"""
Model Registry and Factory

Provides a unified interface for selecting between model architectures:
- base: GPT-2 style transformer (simple, fast training)
- enhanced: Modern optimizations (RoPE, GQA, Flash Attention, RMSNorm, SwiGLU)
- professional: merged profile architecture using enterprise config semantics

Usage:
    from src.core.models import get_model, ModelConfig, list_models

    # List available models
    print(list_models())  # ['base', 'enhanced']

    # Create model using factory
    model = get_model('enhanced', config)

    # Or use config-based selection
    config = ModelConfig(architecture='enhanced', vocab_size=32000, ...)
    model = get_model_from_config(config)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import re
import shutil

from .base_model import GPTModel
from .configuration import ModelArchitectureConfig
from .enhanced_model import EnhancedGPTModel, EnhancedModelConfig
from .professional_transformer import ProfessionalTransformerModel


@dataclass
class ModelConfig:
    """Unified model configuration for all architectures."""

    # Architecture selection
    architecture: str = "base"  # "base", "enhanced", or "professional"

    # Common parameters
    vocab_size: int = 32000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 8
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1

    # Enhanced model specific (ignored for base model)
    use_rotary_embeddings: bool = True
    use_flash_attention: bool = True
    use_grouped_query_attention: bool = True
    gqa_num_kv_heads: int = 2
    use_rms_norm: bool = True
    use_swiglu: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False
    model_name: str = ""
    parameter_count: int = 0
    architecture_family: str = "decoder-only-transformer"

    # Model versioning
    version_models: bool = True  # Keep old best_model.pt versions

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        if self.d_ff < self.d_model:
            raise ValueError(f"d_ff ({self.d_ff}) should be >= d_model ({self.d_model})")
        if self.use_grouped_query_attention and self.num_heads % self.gqa_num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by gqa_num_kv_heads when GQA is enabled")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'architecture': self.architecture,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'max_seq_length': self.max_seq_length,
            'dropout': self.dropout,
            'use_rotary_embeddings': self.use_rotary_embeddings,
            'use_flash_attention': self.use_flash_attention,
            'use_grouped_query_attention': self.use_grouped_query_attention,
            'gqa_num_kv_heads': self.gqa_num_kv_heads,
            'use_rms_norm': self.use_rms_norm,
            'use_swiglu': self.use_swiglu,
            'gradient_checkpointing': self.gradient_checkpointing,
            'compile_model': self.compile_model,
            'model_name': self.model_name,
            'parameter_count': self.parameter_count,
            'architecture_family': self.architecture_family,
            'version_models': self.version_models,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Model registry
_MODEL_REGISTRY: Dict[str, type] = {
    'base': GPTModel,
    'enhanced': EnhancedGPTModel,
    'professional': ProfessionalTransformerModel,
}


def list_models() -> List[str]:
    """List available model architectures."""
    return list(_MODEL_REGISTRY.keys())


def get_model(architecture: str, config: ModelConfig) -> nn.Module:
    """
    Factory function to create a model instance.

    Args:
        architecture: Model architecture name ('base' or 'enhanced')
        config: Model configuration

    Returns:
        Initialized model instance
    """
    if architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list_models()}")

    model_class = _MODEL_REGISTRY[architecture]

    if architecture == 'base':
        return model_class(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout
        )
    elif architecture == 'enhanced':
        enhanced_config = EnhancedModelConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout,
            use_rotary_embeddings=config.use_rotary_embeddings,
            use_flash_attention=config.use_flash_attention,
            use_grouped_query_attention=config.use_grouped_query_attention,
            gqa_num_kv_heads=config.gqa_num_kv_heads,
            use_rms_norm=config.use_rms_norm,
            use_swiglu=config.use_swiglu,
            gradient_checkpointing=config.gradient_checkpointing
        )
        return model_class(enhanced_config)
    elif architecture == 'professional':
        professional_config = ModelArchitectureConfig(
            model_name=config.model_name or "professional-transformer",
            architecture_family=config.architecture_family,
            estimated_parameter_count=config.parameter_count or 0,
            embedding_dimension=config.d_model,
            attention_head_count=config.num_heads,
            transformer_layer_count=config.num_layers,
            feed_forward_dimension=config.d_ff,
            vocabulary_size=config.vocab_size,
            maximum_sequence_length=config.max_seq_length,
            use_rotary_position_embeddings=config.use_rotary_embeddings,
            use_grouped_query_attention=config.use_grouped_query_attention,
            grouped_query_kv_head_count=config.gqa_num_kv_heads,
            use_flash_attention=config.use_flash_attention,
            use_rms_normalization=config.use_rms_norm,
            use_swiglu_activation=config.use_swiglu,
            dropout_probability=config.dropout,
            attention_dropout_probability=config.dropout,
            residual_dropout_probability=config.dropout,
            enable_gradient_checkpointing=config.gradient_checkpointing,
            compile_model=config.compile_model,
            version_models=config.version_models,
        )
        return model_class(professional_config)

    raise ValueError(f"Unknown architecture: {architecture}")


def get_model_from_config(config: ModelConfig) -> nn.Module:
    """Create model from unified config (uses config.architecture)."""
    return get_model(config.architecture, config)


class ModelVersionManager:
    """
    Manages model versioning to preserve previous best models.

    Instead of overwriting best_model.pt, creates versioned copies:
    - best_model_v1.pt (first training)
    - best_model_v2.pt (second training)
    - best_model.pt -> symlink/copy to latest

    This allows rollback and comparison between training runs.
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.model_dir / ".model_versions.json"

    def get_next_version(self) -> int:
        """Get next available version number."""
        existing_versions = self._get_existing_versions()
        return max(existing_versions, default=0) + 1

    def _get_existing_versions(self) -> List[int]:
        """Find all existing version numbers."""
        versions = []
        pattern = re.compile(r'best_model_v(\d+)\.pt')

        for file in self.model_dir.glob('best_model_v*.pt'):
            match = pattern.match(file.name)
            if match:
                versions.append(int(match.group(1)))

        return sorted(versions)

    def save_versioned_model(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: ModelConfig,
        metrics: Dict[str, Any],
        copy_as_latest: bool = True
    ) -> Tuple[str, int]:
        """
        Save model with version number.

        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            config: Model configuration
            metrics: Training metrics (val_loss, etc.)
            copy_as_latest: Also save as best_model.pt

        Returns:
            (path_to_versioned_model, version_number)
        """
        version = self.get_next_version()
        versioned_path = self.model_dir / f"best_model_v{version}.pt"
        latest_path = self.model_dir / "best_model.pt"

        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'metrics': metrics,
            'version': version,
            'architecture': config.architecture,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save versioned model
        torch.save(checkpoint, versioned_path)

        # Copy as latest
        if copy_as_latest:
            shutil.copy(versioned_path, latest_path)

        return str(versioned_path), version

    def load_model(
        self,
        version: Optional[int] = None,
        device: str = 'cpu'
    ) -> Tuple[nn.Module, ModelConfig, Dict[str, Any]]:
        """
        Load model by version number (or latest).

        Args:
            version: Specific version to load, or None for latest
            device: Device to load model to

        Returns:
            (model, config, metrics)
        """
        if version is None:
            model_path = self.model_dir / "best_model.pt"
        else:
            model_path = self.model_dir / f"best_model_v{version}.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        # Reconstruct config
        config = ModelConfig.from_dict(checkpoint['config'])

        # Create and load model
        model = get_model_from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        metrics = checkpoint.get('metrics', {})

        return model, config, metrics

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available model versions with their metrics."""
        versions = []

        for version in self._get_existing_versions():
            model_path = self.model_dir / f"best_model_v{version}.pt"
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                versions.append({
                    'version': version,
                    'path': str(model_path),
                    'architecture': checkpoint.get('architecture', 'unknown'),
                    'metrics': checkpoint.get('metrics', {}),
                })
            except Exception:
                versions.append({
                    'version': version,
                    'path': str(model_path),
                    'error': 'Could not load checkpoint',
                })

        return versions

    def get_best_model_path(self) -> Optional[Path]:
        """Get path to latest best model."""
        latest_path = self.model_dir / "best_model.pt"
        if latest_path.exists():
            return latest_path

        # Fallback to highest version
        versions = self._get_existing_versions()
        if versions:
            return self.model_dir / f"best_model_v{max(versions)}.pt"

        return None


# Convenience exports
__all__ = [
    'GPTModel',
    'EnhancedGPTModel',
    'EnhancedModelConfig',
    'ProfessionalTransformerModel',
    'ModelArchitectureConfig',
    'ModelConfig',
    'get_model',
    'get_model_from_config',
    'list_models',
    'ModelVersionManager',
]
