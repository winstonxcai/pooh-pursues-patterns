"""
Attention Utility Functions

Core functions for implementing and analyzing attention mechanisms:
- Scaled dot-product attention
- Attention mask creation
- Attention statistics and metrics
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V

    This is the fundamental attention operation from "Attention Is All You Need".

    Args:
        Q: Query tensor (batch, ..., seq_len, d_k)
        K: Key tensor (batch, ..., seq_len, d_k)
        V: Value tensor (batch, ..., seq_len, d_v)
        mask: Optional attention mask (seq_len, seq_len) or compatible shape
              1 = allowed, 0 = blocked (will be converted to -inf)
        dropout: Dropout probability (default: 0.0)

    Returns:
        output: Attention output (batch, ..., seq_len, d_v)
        attention_weights: Attention weights after softmax (batch, ..., seq_len, seq_len)

    The scaling factor √d_k prevents the dot products from growing too large,
    which would push softmax into regions with extremely small gradients.
    """
    # Get dimension of keys for scaling
    d_k = Q.size(-1)

    # Compute attention scores: QK^T
    # Q: (..., seq_len_q, d_k)
    # K^T: (..., d_k, seq_len_k)
    # scores: (..., seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Scale by √d_k to prevent saturation
    scores = scores / math.sqrt(d_k)

    # Apply mask if provided (mask out invalid positions)
    if mask is not None:
        # Convert binary mask (1=allowed, 0=blocked) to additive mask
        # 0 -> 0, 1 -> -inf, so after adding: valid scores unchanged, invalid -> -inf
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to get attention weights (sum to 1 over keys)
    attention_weights = F.softmax(scores, dim=-1)

    # Handle NaN from softmax of all -inf (can happen with certain masks)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    # Apply dropout if specified (randomly zero out some attention weights)
    if dropout > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout)

    # Compute weighted sum of values
    # attention_weights: (..., seq_len_q, seq_len_k)
    # V: (..., seq_len_k, d_v)
    # output: (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal (lower-triangular) attention mask for autoregressive models.

    The mask prevents positions from attending to future positions:
    - Position i can attend to positions 0, 1, ..., i (past and self)
    - Position i cannot attend to positions i+1, i+2, ... (future)

    Args:
        seq_len: Sequence length
        device: Device to create mask on (default: CPU)

    Returns:
        Causal mask (seq_len, seq_len) with 1s on and below diagonal, 0s above
        Example for seq_len=4:
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]
    """
    # Create lower triangular matrix (includes diagonal)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask for sequences of different lengths.

    Args:
        lengths: Tensor of sequence lengths (batch_size,)
        max_len: Maximum sequence length

    Returns:
        Padding mask (batch_size, max_len) with 1 for valid positions, 0 for padding
    """
    batch_size = lengths.size(0)
    # Create range tensor: [0, 1, 2, ..., max_len-1]
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, max_len)
    # Broadcast comparison: position < length for each sequence
    mask = positions < lengths.unsqueeze(1)  # (batch_size, max_len)
    return mask.long()


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of attention distributions.

    High entropy = uniform/diffuse attention (attending to many positions)
    Low entropy = focused/sparse attention (attending to few positions)

    Args:
        attention_weights: Attention weights (..., seq_len_q, seq_len_k)

    Returns:
        Entropy values (..., seq_len_q) - one entropy per query position
    """
    # Clip to avoid log(0)
    weights_clipped = torch.clamp(attention_weights, min=1e-9)

    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(weights_clipped * torch.log(weights_clipped), dim=-1)

    return entropy


def compute_attention_sparsity(attention_weights: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Compute sparsity of attention (fraction of weights above threshold).

    Args:
        attention_weights: Attention weights (..., seq_len_q, seq_len_k)
        threshold: Threshold for considering a weight "significant"

    Returns:
        Sparsity values (..., seq_len_q) - fraction of significant weights per query
    """
    # Count weights above threshold
    significant = (attention_weights > threshold).float()
    sparsity = significant.mean(dim=-1)

    return sparsity


def compute_attention_distance(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute average attention distance for each query position.

    Measures how far (in positions) attention spreads on average.

    Args:
        attention_weights: Attention weights (batch, n_heads, seq_len, seq_len)

    Returns:
        Average distances (batch, n_heads, seq_len)
    """
    seq_len = attention_weights.size(-1)
    device = attention_weights.device

    # Create position indices [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device).float()

    # Compute expected position: sum over (weight * position)
    expected_pos = torch.sum(attention_weights * positions.view(1, 1, 1, -1), dim=-1)

    # Compute distance from query position
    query_positions = positions.view(1, 1, -1)
    distances = torch.abs(expected_pos - query_positions)

    return distances


def verify_attention_weights(attention_weights: torch.Tensor, tol: float = 1e-5) -> bool:
    """
    Verify that attention weights are valid probability distributions.

    Checks:
    1. All weights are non-negative
    2. Weights sum to 1 across the key dimension
    3. No NaN or Inf values

    Args:
        attention_weights: Attention weights (..., seq_len_q, seq_len_k)
        tol: Tolerance for sum-to-one check

    Returns:
        True if valid, False otherwise
    """
    # Check for NaN or Inf
    if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
        print("ERROR: Attention weights contain NaN or Inf")
        return False

    # Check non-negative
    if (attention_weights < 0).any():
        print("ERROR: Attention weights contain negative values")
        return False

    # Check sum to 1 (along key dimension)
    sums = attention_weights.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=tol):
        print(f"ERROR: Attention weights don't sum to 1 (max deviation: {(sums - 1).abs().max().item()})")
        return False

    return True


def get_top_k_attention(
    attention_weights: torch.Tensor,
    k: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-k attended positions for each query.

    Args:
        attention_weights: Attention weights (..., seq_len_q, seq_len_k)
        k: Number of top positions to return

    Returns:
        values: Top-k attention weights (..., seq_len_q, k)
        indices: Indices of top-k positions (..., seq_len_q, k)
    """
    values, indices = torch.topk(attention_weights, k=k, dim=-1)
    return values, indices


def attention_rollout(attention_matrices: list) -> torch.Tensor:
    """
    Compute attention rollout across multiple layers.

    Traces information flow by multiplying attention matrices across layers.
    From: "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)

    Args:
        attention_matrices: List of attention matrices, one per layer
                           Each: (batch, n_heads, seq_len, seq_len)

    Returns:
        Rolled-out attention (batch, seq_len, seq_len)
    """
    # Average over heads for each layer
    layer_attentions = [attn.mean(dim=1) for attn in attention_matrices]

    # Multiply matrices from first to last layer
    rollout = layer_attentions[0]
    for layer_attn in layer_attentions[1:]:
        rollout = torch.matmul(rollout, layer_attn)

    return rollout
