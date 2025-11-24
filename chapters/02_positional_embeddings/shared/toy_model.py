"""
Toy Transformer Model for Testing Positional Embeddings

A minimal transformer implementation for demonstrating positional encoding effects.
Small enough to visualize, large enough to be meaningful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        # Self-attention with residual
        if return_attention:
            attn_output, attn_weights = self.attention(
                self.norm1(x), mask, bias, rope_freqs, return_attention=True
            )
            x = x + self.dropout(attn_output)
        else:
            attn_output = self.attention(self.norm1(x), mask, bias, rope_freqs)
            x = x + self.dropout(attn_output)
            attn_weights = None

        # Feed-forward with residual
        x = x + self.dropout(self.feed_forward(self.norm2(x)))

        if return_attention:
            return x, attn_weights
        return x


class ToyTransformer(nn.Module):
    """
    Minimal transformer for testing positional embeddings.

    Supports multiple positional encoding strategies:
    - Sinusoidal
    - Learned
    - RoPE (applied in attention)
    - ALiBi (applied in attention)
    - None (for ablation study)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        position_encoding: str = "learned"  # "sinusoidal", "learned", "rope", "alibi", "none"
    ):
        """
        Initialize toy transformer.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            position_encoding: Type of positional encoding
        """
        super().__init__()
        self.d_model = d_model
        self.position_encoding = position_encoding

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (only for sinusoidal and learned)
        if position_encoding == "sinusoidal":
            # Register as buffer (not a parameter)
            self.register_buffer(
                "position_embedding",
                self._create_sinusoidal_embeddings(max_seq_len, d_model)
            )
        elif position_encoding == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Store parameters for RoPE and ALiBi
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

    def _create_sinusoidal_embeddings(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional embeddings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Args:
            x: Input token IDs (batch, seq_len)
            mask: Attention mask
            return_attention: Whether to return attention weights

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            attention_weights: (optional) List of attention weights per layer
        """
        batch_size, seq_len = x.shape

        # Token embeddings
        token_emb = self.token_embedding(x) * (self.d_model ** 0.5)

        # Add positional embeddings (for sinusoidal and learned)
        if self.position_encoding in ["sinusoidal", "learned"]:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_emb = self.position_embedding if self.position_encoding == "sinusoidal" \
                else self.position_embedding(positions)
            if self.position_encoding == "sinusoidal":
                pos_emb = pos_emb[:seq_len, :].unsqueeze(0)
            x_emb = token_emb + pos_emb
        else:
            x_emb = token_emb

        x_emb = self.dropout(x_emb)

        # Prepare RoPE frequencies if needed
        rope_freqs = None
        if self.position_encoding == "rope":
            from .attention import precompute_rope_frequencies
            rope_freqs = precompute_rope_frequencies(
                seq_len, self.d_k, device=x.device
            )

        # Prepare ALiBi bias if needed
        bias = None
        if self.position_encoding == "alibi":
            from .attention import get_alibi_bias
            bias = get_alibi_bias(seq_len, self.n_heads, device=x.device)

        # Pass through transformer blocks
        attention_weights_list = []
        x_out = x_emb
        for block in self.blocks:
            if return_attention:
                x_out, attn_weights = block(
                    x_out, mask, bias, rope_freqs, return_attention=True
                )
                attention_weights_list.append(attn_weights)
            else:
                x_out = block(x_out, mask, bias, rope_freqs)

        # Output projection
        logits = self.output_projection(x_out)

        if return_attention:
            return logits, attention_weights_list
        return logits, None

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get final embeddings (token + position) for visualization.

        Args:
            x: Input token IDs (batch, seq_len)

        Returns:
            Embeddings tensor (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        token_emb = self.token_embedding(x) * (self.d_model ** 0.5)

        if self.position_encoding in ["sinusoidal", "learned"]:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_emb = self.position_embedding if self.position_encoding == "sinusoidal" \
                else self.position_embedding(positions)
            if self.position_encoding == "sinusoidal":
                pos_emb = pos_emb[:seq_len, :].unsqueeze(0)
            return token_emb + pos_emb
        else:
            return token_emb


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
