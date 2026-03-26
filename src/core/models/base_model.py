"""
GPT-2 Style Transformer Language Model

Mathematical Foundation:
- Self-Attention: Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
- Multi-Head Attention: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
- Feed-Forward: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
- Layer Norm: LayerNorm(x) = gamma(x - mu)/sigma + beta
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism

    Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V

    Properties:
    - Scale invariance through sqrt(d_k) normalization
    - Permutation equivariance
    - Parallel computation across heads
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, 1, seq_len, seq_len]
            cache: KV cache for inference (k_cache, v_cache)

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated KV cache
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections: [batch, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Handle KV caching for autoregressive generation
        if cache is not None:
            k_cache, v_cache = cache
            K = torch.cat([k_cache, K], dim=1)
            V = torch.cat([v_cache, V], dim=1)
            new_cache = (K, V)
        else:
            new_cache = None

        # Reshape for multi-head attention
        # [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # scores: [batch, num_heads, seq_len, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch, num_heads, seq_len, d_k]
        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        # [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output, new_cache


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Typically: d_ff = 4 x d_model (expansion factor of 4)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block with Pre-LayerNorm

    Architecture:
    x -> LayerNorm -> MultiHeadAttention -> + -> LayerNorm -> FFN -> + -> output
    |_________________^                       |_____________^
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Causal mask [batch, 1, seq_len, seq_len]
            cache: KV cache for inference

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated cache
        """
        # Pre-LayerNorm + Self-Attention + Residual
        norm_x = self.ln1(x)
        attn_out, new_cache = self.attention(norm_x, mask, cache)
        x = x + self.dropout(attn_out)

        # Pre-LayerNorm + FFN + Residual
        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)

        return x, new_cache


class GPTModel(nn.Module):
    """
    GPT-2 Style Causal Language Model

    Architecture:
    - Token Embeddings + Position Embeddings
    - N x Transformer Blocks
    - Layer Normalization
    - Language Modeling Head (weight tying)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Language modeling head (weight tying with token embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)

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
            targets: [batch, seq_len] for training
            cache: List of KV caches for each layer

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided)
            new_cache: Updated KV caches
        """
        batch_size, seq_len = input_ids.shape

        # Create position indices
        if cache is not None and len(cache) > 0:
            # For cached inference, only use last position
            pos = torch.arange(cache[0][0].shape[1], cache[0][0].shape[1] + seq_len,
                             dtype=torch.long, device=input_ids.device)
        else:
            pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)

        pos = pos.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        # Create causal mask
        mask = self._create_causal_mask(seq_len, input_ids.device)

        # Pass through transformer blocks
        new_cache = []
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None and i < len(cache) else None
            x, layer_cache = block(x, mask, block_cache)
            new_cache.append(layer_cache)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1  # Ignore padding
            )

        return logits, loss, new_cache if cache is not None else None

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask to prevent attending to future tokens.

        Returns: [1, 1, seq_len, seq_len] mask
        """
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
        """
        Autoregressive generation with KV caching.

        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (keep only top k tokens)
            top_p: Nucleus sampling (keep tokens with cumulative prob > p)

        Returns:
            generated: [batch, seq_len + max_new_tokens]
        """
        self.eval()
        cache = None

        for _ in range(max_new_tokens):
            # Get predictions (use cache after first iteration)
            if cache is None:
                idx_cond = input_ids
            else:
                idx_cond = input_ids[:, -1:]

            logits, _, cache = self(idx_cond, cache=cache)

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
