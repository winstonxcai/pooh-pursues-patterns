"""
Multi-Head Self-Attention Implementation

This module provides a flexible attention implementation that can be used with:
- Sinusoidal positional embeddings
- Learned positional embeddings
- RoPE (Rotary Position Embeddings)
- ALiBi (Attention with Linear Biases)
- No positional encoding (for ablation study)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with flexible positional encoding support.

    Supports:
    - Standard attention (with additive position embeddings)
    - Attention with biases (for ALiBi)
    - Attention with rotated Q/K (for RoPE)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

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
        bias: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len) or (seq_len, seq_len)
            bias: Attention bias for ALiBi (n_heads, seq_len, seq_len)
            rope_freqs: Precomputed frequencies for RoPE (seq_len, d_k//2)
            return_attention: Whether to return attention weights

        Returns:
            output: Attention output (batch, seq_len, d_model)
            attention_weights: (optional) Attention weights (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape

        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, n_heads, seq_len, d_k)

        # Apply RoPE if provided
        if rope_freqs is not None:
            Q = apply_rotary_embedding(Q, rope_freqs)
            K = apply_rotary_embedding(K, rope_freqs)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # Shape: (batch, n_heads, seq_len, seq_len)

        # Add ALiBi bias if provided
        if bias is not None:
            # bias shape: (n_heads, seq_len, seq_len)
            scores = scores + bias.unsqueeze(0)  # Broadcast over batch

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # Shape: (batch, n_heads, seq_len, d_k)

        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, d_model)
        output = self.W_o(context)

        if return_attention:
            return output, attention_weights
        return output


def apply_rotary_embedding(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to input tensor.

    This implements RoPE by treating consecutive dimension pairs as complex numbers
    and applying rotation via complex multiplication.

    Args:
        x: Input tensor (batch, n_heads, seq_len, d_k)
        freqs: Precomputed frequencies (seq_len, d_k//2)

    Returns:
        Rotated tensor with same shape as input
    """
    # Reshape x to pair dimensions: (batch, n_heads, seq_len, d_k//2, 2)
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)

    # Convert to complex numbers
    x_complex = torch.view_as_complex(x_reshaped.float())
    # Shape: (batch, n_heads, seq_len, d_k//2)

    # Create complex rotation
    # freqs shape: (seq_len, d_k//2)
    freqs_complex = torch.polar(
        torch.ones_like(freqs),
        freqs
    )  # e^(i*theta)

    # Broadcast and apply rotation
    # x_complex: (batch, n_heads, seq_len, d_k//2)
    # freqs_complex: (seq_len, d_k//2) -> (1, 1, seq_len, d_k//2)
    x_rotated = x_complex * freqs_complex.unsqueeze(0).unsqueeze(0)

    # Convert back to real representation
    x_out = torch.view_as_real(x_rotated)
    # Shape: (batch, n_heads, seq_len, d_k//2, 2)

    # Flatten last two dimensions
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x)


def precompute_rope_frequencies(
    seq_len: int,
    d_k: int,
    base: float = 10000.0,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Precompute rotation frequencies for RoPE.

    Args:
        seq_len: Maximum sequence length
        d_k: Dimension of each head
        base: Base for frequency computation (default: 10000)
        device: Device to create tensor on

    Returns:
        Frequency tensor (seq_len, d_k//2)
    """
    # Compute base frequencies for each dimension pair
    # theta_i = 1 / (base^(2i/d_k)) for i in [0, d_k/2)
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
    # Shape: (d_k//2,)

    # Compute position indices
    t = torch.arange(seq_len, device=device).float()
    # Shape: (seq_len,)

    # Compute frequencies: pos * theta_i
    freqs = torch.outer(t, inv_freq)
    # Shape: (seq_len, d_k//2)

    return freqs


def get_alibi_bias(
    seq_len: int,
    n_heads: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Compute ALiBi attention bias.

    ALiBi adds a linearly decreasing bias to attention scores based on
    distance between query and key positions.

    Args:
        seq_len: Sequence length
        n_heads: Number of attention heads
        device: Device to create tensor on

    Returns:
        Bias tensor (n_heads, seq_len, seq_len)
    """
    # Compute distance matrix |i - j|
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    distances = torch.abs(positions - positions.T).float()
    # Shape: (seq_len, seq_len)

    # Compute head-specific slopes: 2^(-8/n * head_idx)
    slopes = get_alibi_slopes(n_heads, device)
    # Shape: (n_heads,)

    # Apply slopes to distances: -slope * distance
    # slopes: (n_heads, 1, 1), distances: (seq_len, seq_len)
    bias = -slopes.view(n_heads, 1, 1) * distances.unsqueeze(0)
    # Shape: (n_heads, seq_len, seq_len)

    return bias


def get_alibi_slopes(n_heads: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Compute head-specific slopes for ALiBi.

    Uses geometric sequence: m_h = 2^(-8h/n_heads) for h in [1, n_heads]

    Args:
        n_heads: Number of attention heads
        device: Device to create tensor on

    Returns:
        Slopes tensor (n_heads,)
    """
    # Compute slopes as geometric sequence
    # Start at 2^(-8/n_heads), decrease by factor of 2^(8/n_heads) each step
    slopes = 2.0 ** (-8 * torch.arange(1, n_heads + 1, device=device).float() / n_heads)
    return slopes


def create_causal_mask(seq_len: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Create causal (autoregressive) attention mask.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Mask tensor (seq_len, seq_len) where mask[i, j] = 1 if j <= i else 0
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask
