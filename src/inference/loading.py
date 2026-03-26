"""
Helpers for loading trained checkpoints into the unified inference stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from ..core.models import ModelConfig, get_model_from_config


def build_model_config_from_checkpoint(checkpoint: Dict[str, Any]) -> ModelConfig:
    """Normalize checkpoint config payloads into the canonical runtime model config."""
    raw_config = checkpoint.get("config", {})
    raw_model_config = raw_config.get("model", raw_config) if isinstance(raw_config, dict) else {}
    if not isinstance(raw_model_config, dict):
        raw_model_config = {}

    enhanced_cfg = raw_model_config.get("enhanced", {})
    if not isinstance(enhanced_cfg, dict):
        enhanced_cfg = {}

    return ModelConfig(
        architecture=raw_model_config.get("architecture", "base"),
        model_name=raw_model_config.get("model_name", ""),
        parameter_count=raw_model_config.get("parameter_count", 0),
        architecture_family=raw_model_config.get(
            "architecture_family",
            "decoder-only-transformer",
        ),
        vocab_size=raw_model_config.get("vocab_size", 10000),
        d_model=raw_model_config.get("d_model", raw_model_config.get("embedding_dimension", 384)),
        num_heads=raw_model_config.get(
            "num_heads",
            raw_model_config.get("n_heads", raw_model_config.get("attention_head_count", 6)),
        ),
        num_layers=raw_model_config.get(
            "num_layers",
            raw_model_config.get("n_layers", raw_model_config.get("transformer_layer_count", 6)),
        ),
        d_ff=raw_model_config.get("d_ff", raw_model_config.get("feed_forward_dimension", 1536)),
        max_seq_length=raw_model_config.get(
            "max_seq_length",
            raw_model_config.get("max_seq_len", raw_model_config.get("maximum_sequence_length", 256)),
        ),
        dropout=raw_model_config.get("dropout", raw_model_config.get("dropout_probability", 0.0)),
        use_rotary_embeddings=raw_model_config.get(
            "use_rotary_embeddings",
            enhanced_cfg.get("use_rotary_embeddings", True),
        ),
        use_flash_attention=raw_model_config.get(
            "use_flash_attention",
            enhanced_cfg.get("use_flash_attention", True),
        ),
        use_grouped_query_attention=raw_model_config.get(
            "use_grouped_query_attention",
            enhanced_cfg.get("use_grouped_query_attention", True),
        ),
        gqa_num_kv_heads=raw_model_config.get(
            "gqa_num_kv_heads",
            enhanced_cfg.get("gqa_num_kv_heads", 2),
        ),
        use_rms_norm=raw_model_config.get(
            "use_rms_norm",
            enhanced_cfg.get("use_rms_norm", True),
        ),
        use_swiglu=raw_model_config.get(
            "use_swiglu",
            enhanced_cfg.get("use_swiglu", True),
        ),
        gradient_checkpointing=raw_model_config.get(
            "gradient_checkpointing",
            enhanced_cfg.get("gradient_checkpointing", False),
        ),
        compile_model=raw_model_config.get("compile_model", False),
        version_models=raw_model_config.get("version_models", True),
    )


def load_checkpoint_model(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, ModelConfig, Dict[str, Any]]:
    """Load a checkpoint and reconstruct the matching model architecture."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model_config = build_model_config_from_checkpoint(checkpoint)
    model = get_model_from_config(model_config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, model_config, checkpoint

