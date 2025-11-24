"""
Test Script for Chapter 2: Positional Embeddings

Quick test to verify all implementations work correctly.
"""

import sys
from pathlib import Path

# Add shared directory to path
sys.path.append(str(Path(__file__).parent))

from shared.utils import get_device, set_seed, print_section
from shared.attention import (
    precompute_rope_frequencies,
    apply_rotary_embedding,
    get_alibi_bias,
    get_alibi_slopes
)
from shared.toy_model import ToyTransformer, count_parameters

import torch


def test_shared_utilities():
    """Test shared utility functions."""
    print_section("Testing Shared Utilities")

    device = get_device()
    print(f"Device: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    # Test RoPE
    print("Testing RoPE frequencies...")
    freqs = precompute_rope_frequencies(seq_len=10, d_k=16, device=device)
    print(f"  Frequencies shape: {freqs.shape}")
    assert freqs.shape == (10, 8), f"Expected (10, 8), got {freqs.shape}"
    print("  ✓ RoPE frequencies OK\n")

    # Test RoPE rotation
    print("Testing RoPE rotation...")
    Q = torch.randn(2, 4, 10, 16, device=device)  # (batch, heads, seq, d_k)
    Q_rotated = apply_rotary_embedding(Q, freqs)
    print(f"  Input shape: {Q.shape}")
    print(f"  Output shape: {Q_rotated.shape}")
    assert Q.shape == Q_rotated.shape
    print("  ✓ RoPE rotation OK\n")

    # Test ALiBi
    print("Testing ALiBi bias...")
    bias = get_alibi_bias(seq_len=10, n_heads=8, device=device)
    print(f"  Bias shape: {bias.shape}")
    assert bias.shape == (8, 10, 10), f"Expected (8, 10, 10), got {bias.shape}"
    print("  ✓ ALiBi bias OK\n")

    # Test ALiBi slopes
    print("Testing ALiBi slopes...")
    slopes = get_alibi_slopes(n_heads=8, device=device)
    print(f"  Slopes shape: {slopes.shape}")
    print(f"  Slopes: {slopes.cpu().numpy()}")
    assert slopes.shape == (8,), f"Expected (8,), got {slopes.shape}"
    print("  ✓ ALiBi slopes OK\n")

    print("✅ All shared utilities tests passed!\n")


def test_toy_models():
    """Test toy transformer with different position encodings."""
    print_section("Testing Toy Transformer Models")

    set_seed(42)
    device = get_device()

    vocab_size = 50
    d_model = 64
    n_heads = 4
    seq_len = 12
    batch_size = 2

    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    methods = ["sinusoidal", "learned", "rope", "alibi", "none"]

    for method in methods:
        print(f"Testing {method.upper()}...")

        model = ToyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=2,
            max_seq_len=128,
            position_encoding=method
        ).to(device)

        # Count parameters
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")

        # Forward pass
        logits, attention = model(x, return_attention=True)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Attention shape: {attention[0].shape}")

        # Verify shapes
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert attention[0].shape == (batch_size, n_heads, seq_len, seq_len)

        print(f"  ✓ {method.upper()} OK\n")

    print("✅ All toy model tests passed!\n")


def test_extrapolation():
    """Test that methods can handle different sequence lengths."""
    print_section("Testing Length Extrapolation")

    set_seed(42)
    device = get_device()

    vocab_size = 50
    d_model = 64
    n_heads = 4

    test_lengths = [6, 12, 24, 48]

    methods = ["sinusoidal", "rope", "alibi", "none"]

    print(f"Testing sequence lengths: {test_lengths}\n")

    results = {}

    for method in methods:
        print(f"Testing {method.upper()}...")

        # Create model with max length for the longest sequence
        model = ToyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=1,
            max_seq_len=max(test_lengths),
            position_encoding=method
        ).to(device)

        model.eval()

        results[method] = []

        for length in test_lengths:
            x = torch.randint(0, vocab_size, (1, length), device=device)

            try:
                with torch.no_grad():
                    logits, _ = model(x)
                    assert logits.shape == (1, length, vocab_size)
                print(f"  Length {length:2d}: ✓")
                results[method].append(True)
            except Exception as e:
                print(f"  Length {length:2d}: ✗ ({e})")
                results[method].append(False)

        print()

    # Summary
    print("Summary:")
    print("-" * 60)
    print(f"{'Method':<15} ", end='')
    for length in test_lengths:
        print(f"{'L' + str(length):>6}", end='')
    print()
    print("-" * 60)

    for method in methods:
        print(f"{method.upper():<15} ", end='')
        for success in results[method]:
            print(f"{'✓' if success else '✗':>6}", end='')
        print()

    print("-" * 60)
    print("\n✅ Extrapolation tests complete!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("CHAPTER 2: POSITIONAL EMBEDDINGS - TEST SUITE")
    print("=" * 80)
    print()

    try:
        test_shared_utilities()
        test_toy_models()
        test_extrapolation()

        print("=" * 80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 80)
        print()
        print("You can now run individual phase scripts:")
        print("  Phase 1: uv run python phase1_classic/sinusoidal.py")
        print("  Phase 1: uv run python phase1_classic/learned.py")
        print("  Phase 2: uv run python phase2_rope/rope.py")
        print("  Phase 3: uv run python phase3_alibi/alibi.py")
        print("  Phase 4: uv run python phase4_ablation/no_position.py")
        print()

    except Exception as e:
        print("\n❌ TESTS FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
