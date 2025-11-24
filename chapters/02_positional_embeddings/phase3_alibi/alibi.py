"""
ALiBi - Attention with Linear Biases (Press et al., 2021)

Used in BLOOM, MPT, and other models optimized for long contexts.

Key idea: Don't use positional embeddings at all! Instead, add position-dependent
biases directly to attention scores based on distance between positions.

Properties:
- Zero positional parameters
- Excellent extrapolation (tested up to 20x training length!)
- Memory efficient
- Simpler than other methods
- Different heads have different slopes (multi-scale attention)
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
from shared.attention import get_alibi_bias, get_alibi_slopes


def demonstrate_alibi():
    """Demonstrate ALiBi attention biases."""
    print_section("ALiBi (Attention with Linear Biases) Demo")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    # Parameters
    seq_len = 12
    n_heads = 8

    print(f"Configuration:")
    print(f"  seq_len: {seq_len}")
    print(f"  n_heads: {n_heads}")
    print(f"  Parameters: 0 (no positional embeddings!)\n")

    # Compute ALiBi slopes
    slopes = get_alibi_slopes(n_heads, device)

    print(f"Head-specific slopes (m_h = 2^(-8h/n_heads)):")
    for h in range(n_heads):
        print(f"  Head {h}: slope = {slopes[h].item():.6f} = 2^{-8*(h+1)/n_heads:.2f}")
    print()

    # Compute ALiBi bias
    bias = get_alibi_bias(seq_len, n_heads, device)

    print(f"ALiBi bias shape: {bias.shape}")
    print(f"  (n_heads, seq_len, seq_len) = ({n_heads}, {seq_len}, {seq_len})\n")

    # Show bias for first head
    print(f"Bias matrix for Head 0 (slope={slopes[0].item():.4f}):")
    print(f"  (rows=query, cols=key)")
    print(bias[0].cpu().numpy())
    print()

    # Demonstrate the linear bias formula: -m * |i - j|
    print("Verifying linear bias formula: bias[i, j] = -m * |i - j|")
    print()

    head = 0
    m = slopes[head].item()
    for i in [0, 1, 5]:
        for j in [0, 1, 5]:
            distance = abs(i - j)
            expected = -m * distance
            actual = bias[head, i, j].item()
            print(f"  Position ({i}, {j}): distance={distance}, "
                  f"expected={expected:.4f}, actual={actual:.4f}")
        print()

    # Show how different heads have different biases
    print("Comparing biases across heads (for query pos=5, key pos=0):")
    print(f"  Distance = |5 - 0| = 5")
    print()

    for h in range(n_heads):
        m_h = slopes[h].item()
        bias_value = bias[h, 5, 0].item()
        print(f"  Head {h}: slope={m_h:.6f}, bias={bias_value:.4f}")
    print()

    # Demonstrate attention score modification
    print("How ALiBi modifies attention scores:")
    print("-" * 80)

    # Create dummy attention scores (before softmax)
    torch.manual_seed(42)
    raw_scores = torch.randn(1, n_heads, seq_len, seq_len, device=device)

    # Add ALiBi bias
    scores_with_alibi = raw_scores + bias.unsqueeze(0)

    print(f"For Head 0, Query position 5:")
    print(f"  {'Key Pos':<10} {'Raw Score':<15} {'ALiBi Bias':<15} {'Final Score':<15}")
    print(f"  {'-'*10:<10} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15}")

    for k in range(min(8, seq_len)):
        raw = raw_scores[0, 0, 5, k].item()
        bias_val = bias[0, 5, k].item()
        final = scores_with_alibi[0, 0, 5, k].item()
        print(f"  {k:<10} {raw:<15.4f} {bias_val:<15.4f} {final:<15.4f}")

    print()
    print("Notice: Bias decreases linearly with distance from query position")
    print("        Nearby keys get less penalty, distant keys get more penalty")
    print()

    # Demonstrate extrapolation capability
    print("Extrapolation to longer sequences:")
    print("-" * 80)

    for test_len in [12, 24, 48, 96]:
        test_bias = get_alibi_bias(test_len, n_heads, device)
        print(f"  Sequence length {test_len:3d}: bias shape = {test_bias.shape}, "
              f"max distance = {test_len-1}, max bias = {test_bias[0].min().item():.2f}")

    print()
    print("ALiBi works seamlessly for any sequence length!")
    print()

    # Save biases for visualization
    save_path = DATA_PROCESSED_DIR / "alibi_biases.pt"
    torch.save({
        'bias': bias.cpu(),
        'slopes': slopes.cpu(),
        'seq_len': seq_len,
        'n_heads': n_heads
    }, save_path)
    print(f"Saved ALiBi biases to: {save_path}")

    return bias, slopes


class ALiBiAttention(nn.Module):
    """Simple self-attention with ALiBi."""

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

    def forward(self, x: torch.Tensor):
        """
        Forward pass with ALiBi.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            output: Output (batch, seq_len, d_model)
            attention_weights: Attention weights (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Add ALiBi bias
        bias = get_alibi_bias(seq_len, self.n_heads, device=x.device)
        scores = scores + bias.unsqueeze(0)

        # Softmax and apply to values
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, attn


def test_alibi_attention():
    """Test ALiBi attention module."""
    print_section("Testing ALiBi Attention Module")

    set_seed(42)
    device = get_device()

    d_model = 64
    n_heads = 8
    seq_len = 12
    batch_size = 2

    # Create attention module (no positional embeddings!)
    attn = ALiBiAttention(d_model, n_heads).to(device)

    # Create dummy input (just token embeddings, no position added)
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Forward pass
    output, attn_weights = attn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print()

    # Test on different sequence lengths (extrapolation)
    print("Testing extrapolation to different sequence lengths:")
    for test_len in [6, 12, 24, 48]:
        x_test = torch.randn(1, test_len, d_model, device=device)
        try:
            output_test, _ = attn(x_test)
            print(f"  Length {test_len:2d}: ✓ Success, output shape = {output_test.shape}")
        except Exception as e:
            print(f"  Length {test_len:2d}: ✗ Failed - {e}")

    print(f"\nALiBi attention test successful!")
    print("Notice: ALiBi works for ANY sequence length without retraining!")


def main():
    parser = argparse.ArgumentParser(description="ALiBi Demo")
    parser.add_argument("--seq-len", type=int, default=12, help="Sequence length")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--test-attention", action="store_true", help="Test ALiBi attention module")
    args = parser.parse_args()

    demonstrate_alibi()

    if args.test_attention:
        test_alibi_attention()


if __name__ == "__main__":
    main()
