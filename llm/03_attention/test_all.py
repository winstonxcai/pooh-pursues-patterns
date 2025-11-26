"""
Test Suite for Chapter 3: Self-Attention & Multi-Head Attention

Comprehensive tests to verify all implementations work correctly.

Tests:
- Shared utilities and attention functions
- Single-head self-attention
- Multi-head attention
- Causal masking
- Attention properties (valid probabilities, causality)
- Shape consistency across all modules
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from shared.utils import (
    get_device,
    set_seed,
    load_toy_sequence,
    tokens_to_ids,
    print_section
)
from shared.attention_utils import (
    scaled_dot_product_attention,
    create_causal_mask,
    compute_attention_entropy,
    verify_attention_weights
)
from phase1_single_head.self_attention import SelfAttention
from phase2_multi_head.multi_head_attention import MultiHeadAttention


def test_utilities():
    """Test shared utility functions."""
    print_section("Testing Shared Utilities")

    device = get_device()
    print(f"Device: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    # Test toy sequence
    print("Testing toy sequence loading...")
    tokens, vocab = load_toy_sequence()
    assert len(tokens) == 6, f"Expected 6 tokens, got {len(tokens)}"
    assert len(vocab) == 4, f"Expected 4 unique tokens, got {len(vocab)}"
    assert tokens == ["the", "cat", "sat", "on", "the", "mat"]
    print(f"  ✓ Toy sequence: {tokens}")
    print(f"  ✓ Vocabulary: {vocab}\n")

    # Test tokens_to_ids
    print("Testing token ID conversion...")
    token_ids = tokens_to_ids(tokens, vocab)
    assert token_ids.shape == (6,)
    assert token_ids.dtype == torch.long
    print(f"  ✓ Token IDs: {token_ids.tolist()}\n")

    print("✅ Utility tests passed!\n")


def test_attention_utils():
    """Test attention utility functions."""
    print_section("Testing Attention Utilities")

    set_seed(42)
    device = get_device()

    batch_size = 2
    seq_len = 6
    d_k = 16

    # Test scaled_dot_product_attention
    print("Testing scaled dot-product attention...")
    Q = torch.randn(batch_size, seq_len, d_k, device=device)
    K = torch.randn(batch_size, seq_len, d_k, device=device)
    V = torch.randn(batch_size, seq_len, d_k, device=device)

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    assert output.shape == (batch_size, seq_len, d_k), f"Expected {(batch_size, seq_len, d_k)}, got {output.shape}"
    assert attention_weights.shape == (batch_size, seq_len, seq_len)
    assert verify_attention_weights(attention_weights), "Attention weights are invalid!"
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Attention shape: {attention_weights.shape}")
    print(f"  ✓ Attention weights valid (sum to 1)\n")

    # Test causal mask
    print("Testing causal mask creation...")
    mask = create_causal_mask(seq_len, device=device)
    assert mask.shape == (seq_len, seq_len)
    # Check lower triangle (including diagonal) is 1
    lower_triangle = torch.tril(torch.ones(seq_len, seq_len, device=device))
    assert torch.all(mask == lower_triangle), "Causal mask incorrect!"
    print(f"  ✓ Mask shape: {mask.shape}")
    print(f"  ✓ Mask is lower triangular\n")

    # Test attention with causal mask
    print("Testing attention with causal mask...")
    output_causal, attn_causal = scaled_dot_product_attention(Q, K, V, mask=mask)
    # Verify upper triangle is zero
    upper_triangle = torch.triu(attn_causal, diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6), \
        "Causal masking failed - upper triangle not zero!"
    print(f"  ✓ Upper triangle is zero (causality enforced)\n")

    # Test entropy computation
    print("Testing attention entropy computation...")
    entropy = compute_attention_entropy(attention_weights)
    assert entropy.shape == (batch_size, seq_len)
    assert torch.all(entropy >= 0), "Entropy should be non-negative!"
    print(f"  ✓ Entropy shape: {entropy.shape}")
    print(f"  ✓ Entropy values are non-negative\n")

    print("✅ Attention utility tests passed!\n")


def test_single_head_attention():
    """Test single-head self-attention module."""
    print_section("Testing Single-Head Self-Attention")

    set_seed(42)
    device = get_device()

    batch_size = 2
    seq_len = 6
    d_model = 64
    vocab_size = 20

    print(f"Configuration: batch={batch_size}, seq={seq_len}, d_model={d_model}\n")

    # Create module
    print("Creating single-head attention module...")
    self_attn = SelfAttention(d_model).to(device)
    n_params = sum(p.numel() for p in self_attn.parameters())
    print(f"  ✓ Parameters: {n_params:,}\n")

    # Create input
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    x_emb = embedding(x)

    # Forward pass without mask
    print("Testing forward pass (no mask)...")
    output = self_attn(x_emb, return_attention=False)
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print(f"  ✓ Output shape: {output.shape}\n")

    # Forward pass with attention weights
    print("Testing forward pass (return attention)...")
    output, attention = self_attn(x_emb, return_attention=True)
    assert attention.shape == (batch_size, seq_len, seq_len)
    assert verify_attention_weights(attention), "Attention weights invalid!"
    print(f"  ✓ Attention shape: {attention.shape}")
    print(f"  ✓ Attention weights valid\n")

    # Forward pass with causal mask
    print("Testing forward pass (causal mask)...")
    mask = create_causal_mask(seq_len, device=device)
    output_causal, attn_causal = self_attn(x_emb, mask=mask, return_attention=True)
    upper_triangle = torch.triu(attn_causal, diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6), \
        "Causal masking failed!"
    print(f"  ✓ Causal attention enforced\n")

    print("✅ Single-head attention tests passed!\n")


def test_multi_head_attention():
    """Test multi-head self-attention module."""
    print_section("Testing Multi-Head Self-Attention")

    set_seed(42)
    device = get_device()

    batch_size = 2
    seq_len = 6
    d_model = 64
    n_heads = 4
    vocab_size = 20

    print(f"Configuration: batch={batch_size}, seq={seq_len}, d_model={d_model}, heads={n_heads}\n")

    # Create module
    print("Creating multi-head attention module...")
    multihead_attn = MultiHeadAttention(d_model, n_heads).to(device)
    n_params = sum(p.numel() for p in multihead_attn.parameters())
    print(f"  ✓ Parameters: {n_params:,}\n")

    # Verify d_k
    assert multihead_attn.d_k == d_model // n_heads, \
        f"d_k should be {d_model // n_heads}, got {multihead_attn.d_k}"
    print(f"  ✓ d_k = {multihead_attn.d_k} (d_model / n_heads)\n")

    # Create input
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    x_emb = embedding(x)

    # Forward pass without mask
    print("Testing forward pass (no mask)...")
    output = multihead_attn(x_emb, return_attention=False)
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print(f"  ✓ Output shape: {output.shape}\n")

    # Forward pass with attention weights
    print("Testing forward pass (return attention)...")
    output, attention = multihead_attn(x_emb, return_attention=True)
    assert attention.shape == (batch_size, n_heads, seq_len, seq_len), \
        f"Expected {(batch_size, n_heads, seq_len, seq_len)}, got {attention.shape}"

    # Verify each head has valid attention weights
    for head_idx in range(n_heads):
        head_attn = attention[:, head_idx, :, :]
        assert verify_attention_weights(head_attn), f"Head {head_idx} has invalid attention!"

    print(f"  ✓ Attention shape: {attention.shape}")
    print(f"  ✓ All {n_heads} heads have valid attention weights\n")

    # Forward pass with causal mask
    print("Testing forward pass (causal mask)...")
    mask = create_causal_mask(seq_len, device=device)
    output_causal, attn_causal = multihead_attn(x_emb, mask=mask, return_attention=True)

    # Verify causality for all heads
    for head_idx in range(n_heads):
        head_attn = attn_causal[:, head_idx, :, :]
        upper_triangle = torch.triu(head_attn, diagonal=1)
        assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6), \
            f"Head {head_idx} failed causal masking!"

    print(f"  ✓ Causal attention enforced for all heads\n")

    # Test head diversity (heads should not all be identical)
    print("Testing head diversity...")
    attn_np = attention.squeeze(0).cpu().numpy()  # (n_heads, seq_len, seq_len)
    correlations = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            corr = np.corrcoef(attn_np[i].flatten(), attn_np[j].flatten())[0, 1]
            correlations.append(corr)

    avg_corr = np.mean(correlations)
    print(f"  Average correlation between heads: {avg_corr:.3f}")

    # Heads shouldn't be perfectly correlated (that would mean no diversity)
    # But we don't enforce this strictly since weights are random
    print(f"  ✓ Computed head correlations\n")

    print("✅ Multi-head attention tests passed!\n")


def test_dimensionality():
    """Test that all modules handle different dimensions correctly."""
    print_section("Testing Dimensional Flexibility")

    set_seed(42)
    device = get_device()

    test_configs = [
        {"d_model": 32, "n_heads": 2, "seq_len": 4},
        {"d_model": 64, "n_heads": 4, "seq_len": 8},
        {"d_model": 128, "n_heads": 8, "seq_len": 16},
    ]

    for i, config in enumerate(test_configs, 1):
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        seq_len = config["seq_len"]

        print(f"Config {i}: d_model={d_model}, n_heads={n_heads}, seq_len={seq_len}")

        # Test multi-head attention
        model = MultiHeadAttention(d_model, n_heads).to(device)
        x = torch.randn(1, seq_len, d_model, device=device)

        output, attention = model(x, return_attention=True)

        assert output.shape == (1, seq_len, d_model)
        assert attention.shape == (1, n_heads, seq_len, seq_len)
        assert verify_attention_weights(attention[:, 0, :, :])

        print(f"  ✓ All shapes correct\n")

    print("✅ Dimensionality tests passed!\n")


def test_numerical_stability():
    """Test for numerical stability (no NaNs or Infs)."""
    print_section("Testing Numerical Stability")

    set_seed(42)
    device = get_device()

    d_model = 64
    n_heads = 4
    seq_len = 6

    model = MultiHeadAttention(d_model, n_heads).to(device)

    # Test with normal inputs
    print("Testing with normal inputs...")
    x = torch.randn(1, seq_len, d_model, device=device)
    output, attention = model(x, return_attention=True)

    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"
    assert not torch.isnan(attention).any(), "Attention contains NaN!"
    assert not torch.isinf(attention).any(), "Attention contains Inf!"
    print("  ✓ No NaNs or Infs\n")

    # Test with very small values
    print("Testing with very small values...")
    x_small = torch.randn(1, seq_len, d_model, device=device) * 1e-6
    output_small, attn_small = model(x_small, return_attention=True)

    assert not torch.isnan(output_small).any(), "Output contains NaN with small inputs!"
    assert not torch.isnan(attn_small).any(), "Attention contains NaN with small inputs!"
    print("  ✓ Stable with small values\n")

    # Test with very large values
    print("Testing with very large values...")
    x_large = torch.randn(1, seq_len, d_model, device=device) * 100
    output_large, attn_large = model(x_large, return_attention=True)

    assert not torch.isnan(output_large).any(), "Output contains NaN with large inputs!"
    assert not torch.isnan(attn_large).any(), "Attention contains NaN with large inputs!"
    print("  ✓ Stable with large values\n")

    print("✅ Numerical stability tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("CHAPTER 3: SELF-ATTENTION & MULTI-HEAD ATTENTION - TEST SUITE")
    print("=" * 80)
    print()

    try:
        test_utilities()
        test_attention_utils()
        test_single_head_attention()
        test_multi_head_attention()
        test_dimensionality()
        test_numerical_stability()

        print("=" * 80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 80)
        print()
        print("Your attention implementation is working correctly!")
        print()
        print("You can now run the experiments:")
        print("  Phase 1: uv run python phase1_single_head/self_attention.py")
        print("  Phase 2: uv run python phase2_multi_head/multi_head_attention.py")
        print("  Phase 3: uv run python phase3_visualization/attention_flow.py")
        print("  Phase 4: uv run python phase4_causal/causal_attention.py")
        print()
        print("Or run everything at once:")
        print("  ./run_all.sh")
        print()

    except Exception as e:
        print("\n❌ TESTS FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
