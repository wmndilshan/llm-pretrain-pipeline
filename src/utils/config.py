"""
Centralized Environment and Configuration Management

This module consolidates all environment loading and configuration handling,
eliminating duplicate code across the codebase.

Usage:
    from src.utils.config import load_env, get_config, load_yaml_config

    # Load environment variables (call once at startup)
    load_env()

    # Load YAML configuration
    config = load_yaml_config('configs/config.yaml')

    # Get specific config value with default
    batch_size = get_config('training.batch_size', default=32)
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


# Root directory of the project
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """Get the project root directory."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        # Try to find project root by looking for key files
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / 'setup.py').exists() or (parent / 'main.py').exists():
                _PROJECT_ROOT = parent
                break
        if _PROJECT_ROOT is None:
            _PROJECT_ROOT = Path.cwd()
    return _PROJECT_ROOT


def load_env(env_file: Optional[str] = None) -> bool:
    """
    Load environment variables from .env file.

    Automatically handles:
    - Project root .env
    - Current directory .env
    - HF_TOKEN to HUGGING_FACE_HUB_TOKEN aliasing

    Args:
        env_file: Optional specific .env file path

    Returns:
        True if any .env file was loaded
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        # dotenv not installed, skip loading
        return False

    loaded = False
    root = get_project_root()

    # Load from specific file if provided
    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
            loaded = True

    # Load from project root
    root_env = root / '.env'
    if root_env.exists():
        load_dotenv(root_env)
        loaded = True

    # Load from current directory if different
    cwd_env = Path.cwd() / '.env'
    if cwd_env.exists() and cwd_env != root_env:
        load_dotenv(cwd_env)
        loaded = True

    # Handle HuggingFace token aliasing
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token and not os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token

    return loaded


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    # If relative path, try from project root
    if not config_path.is_absolute():
        root = get_project_root()
        if (root / config_path).exists():
            config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return normalize_config(config or {})


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config files from merged donor schemas into the canonical runtime schema.

    Supports:
    - legacy flat runtime configs
    - professional nested profile configs (`model_architecture`, `training_configuration`, ...)
    """
    if 'model_architecture' in config:
        return _normalize_professional_profile(config)
    return config


def _normalize_professional_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert professional nested config format into the canonical runtime schema."""
    model_arch = config.get('model_architecture', {})
    dimensions = model_arch.get('dimensions', {})
    features = model_arch.get('architecture_features', {})
    regularization = model_arch.get('regularization', {})
    optimization = model_arch.get('optimization', {})

    training_cfg = config.get('training_configuration', {})
    optimizer = training_cfg.get('optimizer', {})
    lr_schedule = training_cfg.get('learning_rate_schedule', {})
    batch_cfg = training_cfg.get('batch_configuration', {})
    duration = training_cfg.get('duration', {})
    monitoring = training_cfg.get('monitoring', {})
    checkpoint_mgmt = training_cfg.get('checkpoint_management', {})
    data_loading = training_cfg.get('data_loading', {})

    dataset_cfg = config.get('dataset_configuration', {})
    training_dataset = dataset_cfg.get('training_dataset', {})
    preprocessing = dataset_cfg.get('preprocessing', {})
    split_ratios = dataset_cfg.get('split_ratios', {})

    infra_cfg = config.get('infrastructure_configuration', {})
    compute = infra_cfg.get('compute', {})
    modal_gpu = infra_cfg.get('modal_gpu', {})

    budget_cfg = config.get('budget_configuration', {})
    monitoring_cfg = config.get('monitoring_configuration', {})
    logging_cfg = monitoring_cfg.get('logging', {})
    metrics_cfg = monitoring_cfg.get('metrics', {})
    inference_cfg = config.get('inference_configuration', {})

    model_name = model_arch.get('name', model_arch.get('model_name', 'professional-transformer'))
    parameter_count = model_arch.get('parameter_count', model_arch.get('estimated_parameter_count', 0))
    dataset_name = training_dataset.get('name', 'roneneldan/TinyStories')

    normalized = {
        'dataset': {
            'name': dataset_name,
            'split_ratios': {
                'train': split_ratios.get('train', 0.9),
                'validation': split_ratios.get('validation', 0.05),
                'test': split_ratios.get('test', 0.05),
            },
            'max_seq_length': preprocessing.get(
                'maximum_sequence_length',
                dimensions.get('maximum_sequence_length', 1024),
            ),
            'cache_dir': training_dataset.get('cache_directory', './data/cache'),
            'processed_dir': training_dataset.get('processed_directory', './data/processed'),
            'tokenizer_backend': 'manager',
        },
        'model': {
            'architecture': 'professional',
            'model_name': model_name,
            'parameter_count': parameter_count,
            'architecture_family': model_arch.get('architecture_family', 'decoder-only-transformer'),
            'vocab_size': dimensions.get('vocabulary_size', 50000),
            'd_model': dimensions.get('embedding_dimension', 768),
            'num_heads': dimensions.get('attention_head_count', 12),
            'num_layers': dimensions.get('transformer_layer_count', 16),
            'd_ff': dimensions.get('feed_forward_dimension', 3072),
            'dropout': regularization.get('dropout_probability', 0.1),
            'max_seq_length': dimensions.get('maximum_sequence_length', 1024),
            'use_rotary_embeddings': features.get('use_rotary_position_embeddings', True),
            'use_flash_attention': features.get('use_flash_attention', True),
            'use_grouped_query_attention': features.get('use_grouped_query_attention', True),
            'gqa_num_kv_heads': features.get('grouped_query_kv_head_count', 4),
            'use_rms_norm': features.get('use_rms_normalization', True),
            'use_swiglu': features.get('use_swiglu_activation', True),
            'gradient_checkpointing': optimization.get('enable_gradient_checkpointing', True),
            'compile_model': optimization.get('enable_model_compilation', optimization.get('compile_model', False)),
            'version_models': True,
            'enhanced': {
                'use_rotary_embeddings': features.get('use_rotary_position_embeddings', True),
                'use_flash_attention': features.get('use_flash_attention', True),
                'use_grouped_query_attention': features.get('use_grouped_query_attention', True),
                'gqa_num_kv_heads': features.get('grouped_query_kv_head_count', 4),
                'use_rms_norm': features.get('use_rms_normalization', True),
                'use_swiglu': features.get('use_swiglu_activation', True),
                'gradient_checkpointing': optimization.get('enable_gradient_checkpointing', True),
            },
        },
        'training': {
            'batch_size': batch_cfg.get('batch_size_per_device', 8) or 8,
            'learning_rate': optimizer.get('learning_rate', 3e-4),
            'weight_decay': optimizer.get('weight_decay_coefficient', 0.1),
            'beta1': optimizer.get('optimizer_beta1', 0.9),
            'beta2': optimizer.get('optimizer_beta2', 0.95),
            'grad_clip': optimizer.get('gradient_clipping_threshold', 1.0),
            'warmup_steps': lr_schedule.get('warmup_steps_count', 1000),
            'max_steps': duration.get('total_training_steps', lr_schedule.get('total_training_steps', 10000)),
            'eval_interval': monitoring.get('evaluation_interval_steps', 500),
            'save_interval': monitoring.get('checkpoint_save_interval_steps', 1000),
            'log_interval': monitoring.get('logging_interval_steps', 10),
            'gradient_accumulation_steps': batch_cfg.get('gradient_accumulation_steps', 1),
            'patience': duration.get('early_stopping_patience', 5) or 5,
            'target_effective_batch_size': batch_cfg.get('target_effective_batch_size'),
        },
        'checkpoint': {
            'save_dir': checkpoint_mgmt.get('checkpoint_directory', './models/checkpoints'),
            'keep_last_n': checkpoint_mgmt.get('keep_last_n_checkpoints', 3),
            'keep_best': checkpoint_mgmt.get('keep_best_checkpoint', True),
            'resume_from_latest': True,
        },
        'hardware': {
            'device': compute.get('target_device', 'cuda'),
            'mixed_precision': compute.get('enable_mixed_precision', True),
            'num_workers': data_loading.get('num_workers', 4),
            'pin_memory': data_loading.get('pin_memory', True),
            'modal_gpu': modal_gpu.get('gpu_type', 'A10G'),
        },
        'logging': {
            'log_dir': logging_cfg.get('log_directory', './logs'),
            'tensorboard': metrics_cfg.get('enable_tensorboard', True),
            'wandb': metrics_cfg.get('enable_wandb', False),
            'wandb_project': metrics_cfg.get('wandb_project'),
            'wandb_entity': metrics_cfg.get('wandb_entity'),
        },
        'budget': {
            'monthly_limit': budget_cfg.get('monthly_budget_usd', 30.0),
            'safety_margin': budget_cfg.get('safety_margin_percentage', 90) / 100.0,
        },
        'data_loading': {
            'prefetch_factor': data_loading.get('prefetch_factor'),
            'persistent_workers': data_loading.get('persistent_workers', False),
        },
        'inference': {
            'default_max_new_tokens': inference_cfg.get('generation', {}).get('default_max_new_tokens', 256),
            'default_temperature': inference_cfg.get('generation', {}).get('default_temperature', 0.8),
            'default_top_k': inference_cfg.get('generation', {}).get('default_top_k', 50),
            'default_top_p': inference_cfg.get('generation', {}).get('default_top_p', 0.95),
            'default_repetition_penalty': inference_cfg.get('generation', {}).get('default_repetition_penalty', 1.0),
        },
    }

    phases = []
    phase_mapping = {
        'phase_1_foundation': ('Foundation', 'foundation', None, 'foundation_checkpoint.pt'),
        'phase_2_expansion': ('Expansion', 'expansion', 'foundation_checkpoint.pt', 'expanded_checkpoint.pt'),
        'phase_3_specialization': ('Specialization', 'specialization', 'expanded_checkpoint.pt', 'production_checkpoint.pt'),
    }
    for key, value in budget_cfg.get('training_phases', {}).items():
        name, phase_name, requires_checkpoint, checkpoint_name = phase_mapping.get(
            key,
            (key.replace('_', ' ').title(), key, None, f"{key}.pt"),
        )
        phases.append({
            'name': name,
            'phase': phase_name,
            'dataset': value.get('dataset', dataset_name),
            'max_steps': value.get('training_steps', normalized['training']['max_steps']),
            'gpu_type': value.get('gpu_type', normalized['hardware']['modal_gpu']),
            'estimated_cost': value.get('estimated_cost_usd', 0.0),
            'description': value.get('description', ''),
            'checkpoint_name': checkpoint_name,
            'requires_checkpoint': requires_checkpoint,
        })
    if phases:
        normalized['progressive_training'] = {'phases': phases}

    return normalized


def get_config(
    key: str,
    config: Optional[Dict[str, Any]] = None,
    default: Any = None
) -> Any:
    """
    Get a configuration value using dot notation.

    Args:
        key: Dot-separated key path (e.g., 'training.batch_size')
        config: Configuration dictionary (if None, uses default config)
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    if config is None:
        # Try to load default config
        try:
            config = load_yaml_config('configs/config.yaml')
        except FileNotFoundError:
            return default

    keys = key.split('.')
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment."""
    return os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')


def get_modal_token() -> Optional[str]:
    """Get Modal token from environment."""
    return os.environ.get('MODAL_TOKEN_ID')


def get_modal_secret() -> Optional[str]:
    """Get Modal secret from environment."""
    return os.environ.get('MODAL_TOKEN_SECRET')


class Config:
    """
    Configuration wrapper class with convenient access methods.

    Usage:
        config = Config.from_yaml('configs/config.yaml')
        batch_size = config.get('training.batch_size', default=32)
        training_config = config.training  # Returns training section as dict
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'Config':
        """Load config from YAML file."""
        return cls(load_yaml_config(path))

    def get(self, key: str, default: Any = None) -> Any:
        """Get value using dot notation."""
        return get_config(key, self._data, default)

    def __getattr__(self, name: str) -> Any:
        """Access top-level config sections as attributes."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        return self._data.get(name, {})

    def to_dict(self) -> Dict[str, Any]:
        """Return raw config dictionary."""
        return self._data.copy()


# Auto-load environment on import
load_env()
