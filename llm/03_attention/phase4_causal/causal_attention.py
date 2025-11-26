"""
Phase 4: Causal Masking & Autoregressive Attention

Implements causal (autoregressive) attention for language generation.

Key concept: In generation tasks, we must prevent positions from "seeing the future".
Position i can only attend to positions 0, 1, ..., i (not i+1, i+2, ...).

This is achieved with a causal mask:
- Lower triangular matrix with 1s below diagonal (allowed)
- Upper triangle with 0s (blocked, converted to -inf before softmax)

Critical for GPT-style models!
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    get_device,
    set_seed,
    load_toy_sequence,
    tokens_to_ids,
    print_section,
    print_tensor_info,
    DATA_PROCESSED_DIR
)
from shared.attention_utils import (
    create_causal_mask,
    verify_attention_weights
)
from phase2_multi_head.multi_head_attention import MultiHeadAttention


def demonstrate_causal_mask():
    """
    Demonstrate how causal masking works.

    Shows:
    1. What the causal mask looks like
    2. How it's applied to attention scores
    3. Effect on attention weights
    """
    print_section("Demonstrating Causal Mask")

    device = get_device()

    # Create a toy sequence
    tokens, vocab = load_toy_sequence()
    seq_len = len(tokens)

    print(f"Sequence: {' '.join(tokens)}")
    print(f"Length: {seq_len}\n")

    # Create causal mask
    mask = create_causal_mask(seq_len, device=device)

    print("Causal Mask (1 = allowed, 0 = blocked):")
    print("=" * 50)
    print("     ", end="")
    for i, token in enumerate(tokens):
        print(f"{i:>6}", end="")
    print()
    print("     ", end="")
    for token in tokens:
        print(f"{token:>6}", end="")
    print()
    print("-" * 50)

    mask_np = mask.detach().cpu().numpy()
    for i, query_token in enumerate(tokens):
        print(f"{i} {query_token:>3}:", end="")
        for j in range(seq_len):
            val = int(mask_np[i, j])
            if val == 1:
                print(f"{'✓':>6}", end="")
            else:
                print(f"{'✗':>6}", end="")
        print()
    print("=" * 50)
    print()

    print("Key observations:")
    print("  - Diagonal = 1 (tokens can attend to themselves)")
    print("  - Below diagonal = 1 (tokens can attend to past)")
    print("  - Above diagonal = 0 (tokens CANNOT attend to future)")
    print()
    print("Example: Position 2 ('sat') can attend to:")
    print("  ✓ Position 0 ('the')")
    print("  ✓ Position 1 ('cat')")
    print("  ✓ Position 2 ('sat') [self]")
    print("  ✗ Position 3 ('on') [FUTURE - blocked!]")
    print("  ✗ Position 4 ('the') [FUTURE - blocked!]")
    print("  ✗ Position 5 ('mat') [FUTURE - blocked!]")
    print()


def compare_bidirectional_vs_causal():
    """
    Compare bidirectional (full) attention with causal attention.

    Shows how masking changes attention patterns dramatically.
    """
    print_section("Comparing Bidirectional vs Causal Attention")

    set_seed(42)
    device = get_device()

    # Configuration
    d_model = 64
    n_heads = 4
    vocab_size = 20

    # Load sequence
    tokens, vocab = load_toy_sequence()
    token_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)
    seq_len = len(tokens)

    print(f"Sequence: {' '.join(tokens)}")
    print(f"Configuration: d_model={d_model}, n_heads={n_heads}\n")

    # Create embeddings
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    x = embedding(token_ids)

    # Create multi-head attention
    attention = MultiHeadAttention(d_model, n_heads).to(device)

    # 1. Bidirectional (no mask)
    print("Running BIDIRECTIONAL attention (no mask)...\n")
    output_bidir, attn_bidir = attention(x, mask=None, return_attention=True)

    print_tensor_info("Bidirectional output", output_bidir)
    print_tensor_info("Bidirectional attention", attn_bidir)
    print()

    # 2. Causal (with mask)
    print("Running CAUSAL attention (with mask)...\n")
    causal_mask = create_causal_mask(seq_len, device=device)
    output_causal, attn_causal = attention(x, mask=causal_mask, return_attention=True)

    print_tensor_info("Causal output", output_causal)
    print_tensor_info("Causal attention", attn_causal)
    print()

    # Compare attention patterns
    print("Comparison:")
    print("=" * 70)

    # Average over heads and batch for simplicity
    attn_bidir_avg = attn_bidir.mean(dim=(0, 1)).detach().cpu().numpy()  # (seq_len, seq_len)
    attn_causal_avg = attn_causal.mean(dim=(0, 1)).detach().cpu().numpy()

    print("\nBidirectional attention (averaged over heads):")
    print("-" * 50)
    print("       ", end="")
    for token in tokens:
        print(f"{token:>6}", end="")
    print()
    for i, query_token in enumerate(tokens):
        print(f"{query_token:>6}:", end="")
        for j in range(seq_len):
            print(f"{attn_bidir_avg[i, j]:>6.2f}", end="")
        print()

    print("\nCausal attention (averaged over heads):")
    print("-" * 50)
    print("       ", end="")
    for token in tokens:
        print(f"{token:>6}", end="")
    print()
    for i, query_token in enumerate(tokens):
        print(f"{query_token:>6}:", end="")
        for j in range(seq_len):
            print(f"{attn_causal_avg[i, j]:>6.2f}", end="")
        print()
    print()

    # Verify causal property
    print("Verifying causal property (upper triangle should be all zeros):")
    upper_triangle = torch.triu(attn_causal.mean(dim=1), diagonal=1)  # Above diagonal
    max_future_attention = upper_triangle.max().item()
    print(f"  Maximum attention to future positions: {max_future_attention:.10f}")

    if max_future_attention < 1e-6:
        print("  ✓ Causal property verified! No leakage from future positions.\n")
    else:
        print(f"  ✗ WARNING: Future attention detected! ({max_future_attention})\n")

    # Save both for visualization
    save_path = DATA_PROCESSED_DIR / "phase4_causal_comparison.pt"
    torch.save({
        'attention_bidirectional': attn_bidir.cpu(),
        'attention_causal': attn_causal.cpu(),
        'output_bidirectional': output_bidir.cpu(),
        'output_causal': output_causal.cpu(),
        'tokens': tokens,
        'n_heads': n_heads
    }, save_path)
    print(f"Saved comparison data to: {save_path}\n")


def demonstrate_autoregressive_generation():
    """
    Demonstrate why causal masking is essential for generation.

    In autoregressive generation:
    1. Start with prompt
    2. Predict next token
    3. Append predicted token
    4. Repeat

    Causal masking ensures we never "cheat" by seeing future tokens.
    """
    print_section("Autoregressive Generation with Causal Masking")

    print("In language generation (GPT-style models):")
    print()
    print("Step 1: Given prompt: 'the cat'")
    print("  → Predict next token: 'sat'")
    print("  → MUST ONLY use 'the' and 'cat' (not future tokens!)")
    print()
    print("Step 2: Context: 'the cat sat'")
    print("  → Predict next token: 'on'")
    print("  → MUST ONLY use 'the', 'cat', 'sat'")
    print()
    print("Step 3: Context: 'the cat sat on'")
    print("  → Predict next token: 'the'")
    print("  → MUST ONLY use tokens 0-3")
    print()
    print("And so on...")
    print()
    print("Causal masking enforces this constraint!")
    print()
    print("Without causal masking:")
    print("  ✗ Model could 'see the future' during training")
    print("  ✗ Would learn to cheat instead of learning patterns")
    print("  ✗ Would fail at test time (future isn't available)")
    print()
    print("With causal masking:")
    print("  ✓ Model learns to predict from past only")
    print("  ✓ Training matches inference conditions")
    print("  ✓ Generalizes to generation")
    print()


def run_causal_attention_demo():
    """Run all causal attention demonstrations."""
    print_section("Phase 4: Causal Attention Demonstrations", char="=")

    demonstrate_causal_mask()
    compare_bidirectional_vs_causal()
    demonstrate_autoregressive_generation()

    print_section("Causal Attention Demos Complete!", char="=")
    print("Key takeaways:")
    print("  1. Causal mask prevents attention to future positions")
    print("  2. Essential for autoregressive generation (GPT-style models)")
    print("  3. Upper triangle of attention matrix becomes all zeros")
    print("  4. Training with causal mask matches inference conditions")
    print()
    print("Next step:")
    print("  Run visualize.py to see causal mask structure and comparisons")
    print()


if __name__ == "__main__":
    run_causal_attention_demo()
