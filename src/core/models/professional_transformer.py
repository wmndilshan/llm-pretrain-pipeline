"""
Professional transformer wrapper that uses enterprise-style configuration with the
canonical enhanced runtime implementation.
"""

from __future__ import annotations

import torch.nn as nn

from .configuration import ModelArchitectureConfig
from .enhanced_model import EnhancedGPTModel, EnhancedModelConfig


class ProfessionalTransformerModel(nn.Module):
    """Adapter that exposes the canonical trainer API with professional config semantics."""

    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.professional_config = config
        self.runtime_config = EnhancedModelConfig(
            vocab_size=config.vocabulary_size,
            d_model=config.embedding_dimension,
            num_heads=config.attention_head_count,
            num_layers=config.transformer_layer_count,
            d_ff=config.feed_forward_dimension,
            max_seq_length=config.maximum_sequence_length,
            dropout=config.dropout_probability,
            use_rotary_embeddings=config.use_rotary_position_embeddings,
            use_flash_attention=config.use_flash_attention,
            use_grouped_query_attention=config.use_grouped_query_attention,
            gqa_num_kv_heads=config.grouped_query_kv_head_count,
            use_rms_norm=config.use_rms_normalization,
            use_swiglu=config.use_swiglu_activation,
            gradient_checkpointing=config.enable_gradient_checkpointing,
        )
        self.model = EnhancedGPTModel(self.runtime_config)
        self.vocab_size = config.vocabulary_size

    def forward(self, input_ids, targets=None, cache=None):
        return self.model(input_ids, targets=targets, cache=cache)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def count_parameters(self) -> int:
        return self.model.count_parameters()
