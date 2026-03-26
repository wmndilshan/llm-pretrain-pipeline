"""
Professional model architecture configuration merged from the enterprise/profile repos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class ModelArchitectureConfig:
    """Validated architecture config with professional presets."""

    model_name: str = "medium-200m-transformer"
    architecture_family: str = "decoder-only-transformer"
    model_size_category: Literal["small", "medium", "large", "xlarge"] = "medium"
    estimated_parameter_count: int = 200_000_000

    embedding_dimension: int = 768
    attention_head_count: int = 12
    transformer_layer_count: int = 16
    feed_forward_dimension: int = 3072
    vocabulary_size: int = 50000
    maximum_sequence_length: int = 1024

    use_rotary_position_embeddings: bool = True
    use_grouped_query_attention: bool = True
    grouped_query_kv_head_count: int = 4
    use_flash_attention: bool = True
    use_rms_normalization: bool = True
    use_swiglu_activation: bool = True
    use_tied_embeddings: bool = True

    dropout_probability: float = 0.1
    attention_dropout_probability: float = 0.1
    residual_dropout_probability: float = 0.1

    enable_gradient_checkpointing: bool = True
    mixed_precision_format: str = "bf16"
    compile_model: bool = False
    version_models: bool = True

    def __post_init__(self) -> None:
        if self.embedding_dimension % self.attention_head_count != 0:
            raise ValueError(
                f"embedding_dimension ({self.embedding_dimension}) must be divisible by "
                f"attention_head_count ({self.attention_head_count})"
            )
        if self.use_grouped_query_attention:
            if self.grouped_query_kv_head_count > self.attention_head_count:
                raise ValueError("grouped_query_kv_head_count cannot exceed attention_head_count")
            if self.attention_head_count % self.grouped_query_kv_head_count != 0:
                raise ValueError(
                    "attention_head_count must be divisible by grouped_query_kv_head_count"
                )
        if self.feed_forward_dimension < self.embedding_dimension:
            raise ValueError("feed_forward_dimension should be >= embedding_dimension")
        for name, value in {
            "dropout_probability": self.dropout_probability,
            "attention_dropout_probability": self.attention_dropout_probability,
            "residual_dropout_probability": self.residual_dropout_probability,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

    @property
    def head_dimension(self) -> int:
        return self.embedding_dimension // self.attention_head_count

    def estimate_parameters(self) -> int:
        embedding_params = self.vocabulary_size * self.embedding_dimension
        attention_params = self.transformer_layer_count * 4 * self.embedding_dimension * self.embedding_dimension
        ffn_params = self.transformer_layer_count * 3 * self.embedding_dimension * self.feed_forward_dimension
        return embedding_params + attention_params + ffn_params

    def to_model_config_kwargs(self) -> Dict[str, Any]:
        return {
            "architecture": "professional",
            "vocab_size": self.vocabulary_size,
            "d_model": self.embedding_dimension,
            "num_heads": self.attention_head_count,
            "num_layers": self.transformer_layer_count,
            "d_ff": self.feed_forward_dimension,
            "max_seq_length": self.maximum_sequence_length,
            "dropout": self.dropout_probability,
            "use_rotary_embeddings": self.use_rotary_position_embeddings,
            "use_flash_attention": self.use_flash_attention,
            "use_grouped_query_attention": self.use_grouped_query_attention,
            "gqa_num_kv_heads": self.grouped_query_kv_head_count,
            "use_rms_norm": self.use_rms_normalization,
            "use_swiglu": self.use_swiglu_activation,
            "gradient_checkpointing": self.enable_gradient_checkpointing,
            "compile_model": self.compile_model,
            "version_models": self.version_models,
            "model_name": self.model_name,
            "parameter_count": self.estimated_parameter_count,
            "architecture_family": self.architecture_family,
        }

    @classmethod
    def from_model_config(cls, config: Any) -> "ModelArchitectureConfig":
        return cls(
            model_name=getattr(config, "model_name", "merged-transformer"),
            architecture_family=getattr(config, "architecture_family", "decoder-only-transformer"),
            estimated_parameter_count=getattr(config, "parameter_count", 0) or 0,
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
            compile_model=getattr(config, "compile_model", False),
            version_models=config.version_models,
        )

    @classmethod
    def create_small_85m(cls) -> "ModelArchitectureConfig":
        return cls(
            model_name="small-85m-transformer",
            model_size_category="small",
            estimated_parameter_count=85_000_000,
            embedding_dimension=512,
            attention_head_count=8,
            transformer_layer_count=8,
            feed_forward_dimension=2048,
            vocabulary_size=32000,
            maximum_sequence_length=512,
            grouped_query_kv_head_count=2,
            enable_gradient_checkpointing=False,
        )

    @classmethod
    def create_medium_200m(cls) -> "ModelArchitectureConfig":
        return cls()

    @classmethod
    def create_large_350m(cls) -> "ModelArchitectureConfig":
        return cls(
            model_name="large-350m-transformer",
            model_size_category="large",
            estimated_parameter_count=350_000_000,
            embedding_dimension=1024,
            attention_head_count=16,
            transformer_layer_count=24,
            feed_forward_dimension=4096,
            vocabulary_size=50000,
            maximum_sequence_length=2048,
            grouped_query_kv_head_count=4,
        )

    @classmethod
    def create_xlarge_500m(cls) -> "ModelArchitectureConfig":
        return cls(
            model_name="xlarge-500m-transformer",
            model_size_category="xlarge",
            estimated_parameter_count=500_000_000,
            embedding_dimension=1280,
            attention_head_count=20,
            transformer_layer_count=28,
            feed_forward_dimension=5120,
            vocabulary_size=50000,
            maximum_sequence_length=2048,
            grouped_query_kv_head_count=5,
        )
