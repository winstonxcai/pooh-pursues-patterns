"""
RoPE - Rotary Position Embeddings (Su et al., 2021)

Used in LLaMA, PaLM, GPT-NeoX, and other modern LLMs.

Key idea: Instead of adding position to embeddings, rotate query and key vectors
based on their positions. This makes relative position information explicit in
the dot product.

Properties:
- No added parameters
- Encodes relative position naturally
- Excellent extrapolation/interpolation
- Better than sinusoidal for long sequences
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    get_device,
    set_seed,
    load_toy_sequence,
    tokens_to_ids,
    DATA_PROCESSED_DIR,
    print_section
)
from shared.attention import precompute_rope_frequencies, apply_rotary_embedding


def demonstrate_rope():
    """Demonstrate RoPE positional encoding."""
    print_section("RoPE (Rotary Position Embeddings) Demo")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    # Parameters
    d_model = 64
    n_heads = 4
    d_k = d_model // n_heads
    max_len = 128
    seq_len = 6

    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_k (per head): {d_k}")
    print(f"  max_len: {max_len}")
    print(f"  Parameters: 0 (rotation is deterministic)\n")

    # Precompute RoPE frequencies
    freqs = precompute_rope_frequencies(max_len, d_k, device=device)
    print(f"RoPE frequencies shape: {freqs.shape}")
    print(f"  (max_len, d_k//2) = ({max_len}, {d_k//2})\n")

    # Load toy sequence
    tokens, vocab = load_toy_sequence()
    print(f"Toy sequence: {' '.join(tokens)}")
    print(f"Vocabulary size: {len(vocab)}\n")

    # Create dummy query and key vectors
    batch_size = 1
    Q = torch.randn(batch_size, n_heads, seq_len, d_k, device=device)
    K = torch.randn(batch_size, n_heads, seq_len, d_k, device=device)

    print(f"Before RoPE:")
    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}\n")

    # Apply RoPE
    freqs_for_seq = freqs[:seq_len, :]
    Q_rotated = apply_rotary_embedding(Q, freqs_for_seq)
    K_rotated = apply_rotary_embedding(K, freqs_for_seq)

    print(f"After RoPE:")
    print(f"  Q_rotated shape: {Q_rotated.shape}")
    print(f"  K_rotated shape: {K_rotated.shape}\n")

    # Verify magnitude preservation
    q_mag_before = Q.norm(dim=-1).mean().item()
    q_mag_after = Q_rotated.norm(dim=-1).mean().item()
    print(f"Magnitude preservation:")
    print(f"  Average Q magnitude before: {q_mag_before:.4f}")
    print(f"  Average Q magnitude after: {q_mag_after:.4f}")
    print(f"  Difference: {abs(q_mag_before - q_mag_after):.6f} (should be ~0)\n")

    # Compute attention scores (without and with RoPE)
    scores_no_rope = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    scores_with_rope = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1)) / (d_k ** 0.5)

    print(f"Attention scores:")
    print(f"  Without RoPE shape: {scores_no_rope.shape}")
    print(f"  With RoPE shape: {scores_with_rope.shape}\n")

    # Show that RoPE encodes relative position
    print("Relative Position Property:")
    print("RoPE makes attention scores depend on (query_pos - key_pos)\n")

    # For head 0, position pairs
    head = 0
    for q_pos in [0, 1, 2]:
        for k_pos in [0, 1, 2]:
            score = scores_with_rope[0, head, q_pos, k_pos].item()
            rel_pos = q_pos - k_pos
            print(f"  Q[{q_pos}] · K[{k_pos}] (relative={rel_pos:+2d}): {score:7.4f}")
        print()

    # Save frequencies for visualization
    save_path = DATA_PROCESSED_DIR / "rope_frequencies.pt"
    torch.save({
        'frequencies': freqs.cpu(),
        'd_model': d_model,
        'd_k': d_k,
        'max_len': max_len,
        'n_heads': n_heads
    }, save_path)
    print(f"Saved RoPE frequencies to: {save_path}")

    # Demonstrate rotation at different positions
    print("\nRotation Angles at Different Positions:")
    print("(for first dimension pair, dim_idx=0)")
    print()

    # Compute base frequency for first dimension pair
    base_freq = freqs[0, 0].item()
    print(f"Base frequency: {base_freq:.6f}")
    print()

    for pos in [0, 1, 5, 10, 50]:
        if pos < max_len:
            angle = freqs[pos, 0].item()
            angle_degrees = angle * 180 / np.pi
            print(f"  Position {pos:2d}: angle = {angle:.4f} rad = {angle_degrees:7.2f}°")

    print()

    # Show wavelengths across dimension pairs
    print("Wavelengths across dimension pairs:")
    for dim_pair in range(0, d_k // 2, 2):
        # Compute wavelength: how many positions for 2π rotation
        freq = freqs[1, dim_pair].item()  # Frequency at position 1
        wavelength = 2 * np.pi / freq if freq > 0 else float('inf')
        print(f"  Dim pair {dim_pair}: wavelength = {wavelength:8.2f} positions")

    return freqs


class RoPEAttention(nn.Module):
    """Simple self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor):
        """
        Forward pass with RoPE.

        Args:
            x: Input (batch, seq_len, d_model)
            freqs: RoPE frequencies (seq_len, d_k//2)

        Returns:
            Output (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to Q and K
        Q = apply_rotary_embedding(Q, freqs)
        K = apply_rotary_embedding(K, freqs)

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, attn


def test_rope_attention():
    """Test RoPE attention module."""
    print_section("Testing RoPE Attention Module")

    set_seed(42)
    device = get_device()

    d_model = 64
    n_heads = 4
    seq_len = 6
    batch_size = 2

    # Create attention module
    attn = RoPEAttention(d_model, n_heads).to(device)

    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Precompute frequencies
    freqs = precompute_rope_frequencies(seq_len, d_model // n_heads, device=device)

    # Forward pass
    output, attn_weights = attn(x, freqs)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"\nRoPE attention test successful!")


def main():
    parser = argparse.ArgumentParser(description="RoPE Positional Encoding Demo")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--test-attention", action="store_true", help="Test RoPE attention module")
    args = parser.parse_args()

    demonstrate_rope()

    if args.test_attention:
        test_rope_attention()


if __name__ == "__main__":
    main()
