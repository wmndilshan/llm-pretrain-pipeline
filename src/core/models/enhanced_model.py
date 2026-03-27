"""
Enhanced GPT Model with Modern Optimizations

Improvements over base model:
1. Rotary Position Embeddings (RoPE)
2. Flash Attention
3. Grouped Query Attention (GQA)
4. RMSNorm (faster than LayerNorm)
5. SwiGLU activation (better than ReLU)
6. Gradient checkpointing support
7. torch.compile() compatibility
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class EnhancedModelConfig:
    """Enhanced model configuration."""
    vocab_size: int = 32000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 8
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1

    # Advanced features
    use_rotary_embeddings: bool = True
    use_flash_attention: bool = True
    use_grouped_query_attention: bool = True
    gqa_num_kv_heads: int = 2  # For GQA
    use_rms_norm: bool = True  # Faster than LayerNorm
    use_swiglu: bool = True    # Better than ReLU

    # Training optimizations
    gradient_checkpointing: bool = False


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Faster than LayerNorm, same quality.
    Used in: LLaMA, PaLM, Chinchilla

    RMS(x) = x / sqrt(mean(x^2) + eps)
    y = RMS(x) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            normalized: [batch, seq_len, dim]
        """
        # Compute RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # Normalize and scale
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Better than absolute position embeddings:
    - Relative position information
    - No length extrapolation issues
    - Used in: GPT-Neo, PaLM, LLaMA

    Paper: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_length: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached cos/sin values."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            # Position indices
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)

            # Frequency matrix
            freqs = torch.outer(t, self.inv_freq)

            # Concatenate for rotation
            emb = torch.cat([freqs, freqs], dim=-1)

            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (unused, for interface compatibility)
            seq_len: Sequence length

        Returns:
            cos, sin: Rotation matrices [seq_len, dim]
        """
        self._update_cache(seq_len, x.device)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for RoPE: rotate half the dimensions."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key."""
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]

    # Expand dimensions
    cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(1)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Middle ground between:
    - Multi-Head Attention (MHA): All heads have separate K, V
    - Multi-Query Attention (MQA): All heads share K, V

    GQA: Groups of heads share K, V

    Benefits:
    - 2x faster inference than MHA
    - 30% less memory than MHA
    - <2% quality loss vs MHA

    Paper: https://arxiv.org/abs/2305.13245
    Used in: LLaMA-2, Mistral
    """

    def __init__(
        self,
        config: EnhancedModelConfig,
        use_rotary: bool = True
    ):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.gqa_num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.use_rotary = use_rotary

        assert config.d_model % config.num_heads == 0
        assert config.num_heads % config.gqa_num_kv_heads == 0

        self.num_queries_per_kv = config.num_heads // config.gqa_num_kv_heads

        # Projections
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # RoPE
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_seq_length=config.max_seq_length
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_flash: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, 1, seq_len, seq_len]
            cache: (k_cache, v_cache) for inference
            use_flash: Use Flash Attention if available

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated cache
        """
        batch_size, seq_len, _ = x.shape

        # Projections
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)  # [batch, seq_len, num_kv_heads * head_dim]
        V = self.W_v(x)  # [batch, seq_len, num_kv_heads * head_dim]

        # Reshape for attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.use_rotary:
            cos, sin = self.rotary_emb(x, seq_len)
            # Expand K for RoPE (need to match Q dimensions temporarily)
            K_expanded = K.repeat_interleave(self.num_queries_per_kv, dim=1)
            Q, K_expanded = apply_rotary_pos_emb(Q, K_expanded, cos, sin)
            # Average back to get rotated K
            K = K_expanded.view(batch_size, self.num_heads, seq_len, self.head_dim)
            K = K.view(batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim).mean(dim=2)

        # Handle cache
        if cache is not None:
            k_cache, v_cache = cache
            K = torch.cat([k_cache, K], dim=2)
            V = torch.cat([v_cache, V], dim=2)
            new_cache = (K, V)
        else:
            new_cache = None

        # Expand K, V to match number of query heads
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Scaled dot-product attention
        if use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's built-in Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=(mask is None)  # Use causal mask if no mask provided
            )
        else:
            # Standard attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, V)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)

        return output, new_cache


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    Better than ReLU for transformers.
    Used in: PaLM, LLaMA

    SwiGLU(x) = Swish(xW) * (xV)
    where Swish(x) = x * sigmoid(x)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        return self.w2(F.silu(self.w(x)) * self.v(x))


class TransformerBlock(nn.Module):
    """Enhanced transformer block with modern optimizations."""

    def __init__(self, config: EnhancedModelConfig):
        super().__init__()

        # Attention
        self.attention = GroupedQueryAttention(
            config,
            use_rotary=config.use_rotary_embeddings
        )

        # Feed-forward
        if config.use_swiglu:
            self.ffn = SwiGLU(config.d_model, config.d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model)
            )

        # Normalization
        if config.use_rms_norm:
            self.ln1 = RMSNorm(config.d_model)
            self.ln2 = RMSNorm(config.d_model)
        else:
            self.ln1 = nn.LayerNorm(config.d_model)
            self.ln2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.use_flash = config.use_flash_attention

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm + Attention + Residual
        norm_x = self.ln1(x)
        attn_out, new_cache = self.attention(norm_x, mask, cache, self.use_flash)
        x = x + self.dropout(attn_out)

        # Pre-norm + FFN + Residual
        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)

        return x, new_cache


class EnhancedGPTModel(nn.Module):
    """
    Enhanced GPT model with modern optimizations.

    Improvements:
    - RoPE instead of absolute position embeddings
    - GQA instead of MHA (2x faster inference)
    - RMSNorm instead of LayerNorm (faster)
    - SwiGLU instead of ReLU (better quality)
    - Flash Attention support (2-3x faster)
    - Gradient checkpointing (train larger models)
    """

    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # No position embeddings if using RoPE
        if not config.use_rotary_embeddings:
            self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final norm
        if config.use_rms_norm:
            self.ln_f = RMSNorm(config.d_model)
        else:
            self.ln_f = nn.LayerNorm(config.d_model)

        # Output head (weight tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using scaled initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[list] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Args:
            input_ids: [batch, seq_len]
            targets: [batch, seq_len]
            cache: List of KV caches for each layer

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided)
            new_cache: Updated KV caches
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Position embeddings (if not using RoPE)
        if not self.config.use_rotary_embeddings:
            pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            pos = pos.unsqueeze(0).expand(batch_size, -1)
            x = x + self.position_embedding(pos)

        x = self.dropout(x)

        # Causal mask
        mask = self._create_causal_mask(seq_len, input_ids.device)

        # Pass through transformer blocks with gradient checkpointing
        new_cache = []
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None and i < len(cache) else None

            if self.config.gradient_checkpointing and self.training:
                # Use gradient checkpointing
                x, layer_cache = torch.utils.checkpoint.checkpoint(
                    block, x, mask, block_cache,
                    use_reentrant=False
                )
            else:
                x, layer_cache = block(x, mask, block_cache)

            new_cache.append(layer_cache)

        # Final norm
        x = self.ln_f(x)

        # Output logits
        logits = self.lm_head(x)

        # Calculate loss
        loss = None
        if targets is not None:
            # Next-token prediction: token t predicts token t+1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-1
            )

        return logits, loss, new_cache if cache is not None else None

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text with KV caching."""
        self.eval()
        cache = None

        for _ in range(max_new_tokens):
            idx_cond = input_ids if cache is None else input_ids[:, -1:]

            logits, _, cache = self(idx_cond, cache=cache)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Export
__all__ = ['EnhancedGPTModel', 'EnhancedModelConfig']
