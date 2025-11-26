"""
Phase 4 Visualizations: Causal Masking

Creates visualizations to understand causal masking:
1. Causal mask structure - shows the triangular mask pattern
2. Bidirectional vs Causal comparison - side-by-side attention patterns

These visualizations clearly show how masking prevents future attention.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    get_device,
    setup_plotting_style,
    save_figure,
    print_section,
    DATA_PROCESSED_DIR
)
from shared.attention_utils import create_causal_mask
from shared.visualization_utils import plot_attention_heatmap


def visualize_causal_mask_structure():
    """
    Visualize the causal mask itself.

    Shows the triangular pattern that enforces causality.
    """
    print_section("Visualizing Causal Mask Structure")

    device = get_device()

    # Create mask for toy sequence
    seq_len = 6
    tokens = ["the", "cat", "sat", "on", "the", "mat"]

    mask = create_causal_mask(seq_len, device=device)
    mask_np = mask.detach().cpu().numpy()

    print(f"Causal mask for sequence of length {seq_len}")
    print(f"Tokens: {tokens}\n")

    # Setup plotting
    setup_plotting_style()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Binary mask (1 = allowed, 0 = blocked)
    im1 = ax1.imshow(mask_np, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(seq_len))
    ax1.set_yticks(range(seq_len))
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    ax1.set_yticklabels(tokens)
    ax1.set_xlabel('Key Position (attending from)', fontsize=12)
    ax1.set_ylabel('Query Position (attending to)', fontsize=12)
    ax1.set_title('Causal Mask Structure\n(Green = Allowed, Red = Blocked)',
                  fontsize=14, fontweight='bold')

    # Add grid
    ax1.set_xticks(np.arange(seq_len) - 0.5, minor=True)
    ax1.set_yticks(np.arange(seq_len) - 0.5, minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=2)

    # Add text annotations
    for i in range(seq_len):
        for j in range(seq_len):
            if mask_np[i, j] == 1:
                text = "✓"
                color = "black"
            else:
                text = "✗"
                color = "white"
            ax1.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=16, fontweight='bold')

    plt.colorbar(im1, ax=ax1, label='Mask Value')

    # Plot 2: Show which positions each query can attend to
    ax2.axis('off')

    # Create text explanation
    explanation = "Causal Attention Rules:\n" + "=" * 45 + "\n\n"
    explanation += "Each position can attend to:\n\n"

    for i, token in enumerate(tokens):
        allowed_positions = [j for j in range(seq_len) if mask_np[i, j] == 1]
        allowed_tokens = [tokens[j] for j in allowed_positions]

        explanation += f"Position {i} ('{token}'):\n"
        explanation += f"  ✓ Can see: {allowed_positions}\n"
        explanation += f"  ✓ Tokens: {allowed_tokens}\n"
        explanation += f"  ✗ Cannot see: {[j for j in range(seq_len) if j > i]}\n\n"

    explanation += "=" * 45 + "\n\n"
    explanation += "Key insight:\n"
    explanation += "Position i can ONLY attend to positions 0..i\n"
    explanation += "(past and present, never future!)"

    ax2.text(0.1, 0.95, explanation, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    save_figure(fig, "causal_mask_structure.png", subdir="phase4", dpi=300)

    print("\n✓ Causal mask structure visualization complete!")
    print("  The mask shows:")
    print("  - Lower triangle (including diagonal): Allowed (green)")
    print("  - Upper triangle: Blocked (red)")
    print("  - This enforces causality: can't see future!\n")


def visualize_bidirectional_vs_causal_comparison():
    """
    Create side-by-side comparison of bidirectional and causal attention.

    Shows dramatically how masking changes the attention pattern.
    """
    print_section("Visualizing Bidirectional vs Causal Comparison")

    # Load comparison data
    data_path = DATA_PROCESSED_DIR / "phase4_causal_comparison.pt"

    if not data_path.exists():
        print(f"Error: Comparison data not found at {data_path}")
        print("Please run causal_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attn_bidir = data['attention_bidirectional'].squeeze(0).detach().cpu().numpy()  # (n_heads, seq_len, seq_len)
    attn_causal = data['attention_causal'].squeeze(0).detach().cpu().numpy()
    tokens = data['tokens']
    n_heads = data['n_heads']

    # Average over heads for cleaner visualization
    attn_bidir_avg = attn_bidir.mean(axis=0)  # (seq_len, seq_len)
    attn_causal_avg = attn_causal.mean(axis=0)

    print(f"Loaded attention patterns")
    print(f"  Tokens: {tokens}")
    print(f"  Heads: {n_heads}")
    print(f"  Averaged over heads for visualization\n")

    # Setup plotting
    setup_plotting_style()

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Bidirectional
    plot_attention_heatmap(
        attn_bidir_avg,
        tokens,
        title="Bidirectional Attention\n(Full - can see all positions)",
        ax=ax1
    )

    # Plot 2: Causal
    plot_attention_heatmap(
        attn_causal_avg,
        tokens,
        title="Causal Attention\n(Masked - can only see past)",
        ax=ax2
    )

    # Add main title
    fig.suptitle('Bidirectional vs Causal Attention Comparison',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    save_figure(fig, "bidirectional_vs_causal.png", subdir="phase4", dpi=300)

    print("\n✓ Comparison visualization complete!")
    print("  Key differences:")
    print("  - Bidirectional: Full attention matrix (uses all positions)")
    print("  - Causal: Upper triangle is zero (future positions blocked)")
    print("  - Notice how attention redistributes in causal mode!")
    print()

    # Quantitative comparison
    print("Quantitative comparison:")
    print("-" * 70)

    # Count non-zero elements
    threshold = 0.01
    bidir_nonzero = (attn_bidir_avg > threshold).sum()
    causal_nonzero = (attn_causal_avg > threshold).sum()
    total_elements = attn_bidir_avg.size

    print(f"Non-zero elements (threshold={threshold}):")
    print(f"  Bidirectional: {bidir_nonzero} / {total_elements} ({bidir_nonzero/total_elements*100:.1f}%)")
    print(f"  Causal: {causal_nonzero} / {total_elements} ({causal_nonzero/total_elements*100:.1f}%)")
    print()

    # Verify upper triangle is zero in causal
    seq_len = len(tokens)
    upper_triangle_sum = 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            upper_triangle_sum += attn_causal_avg[i, j]

    print(f"Sum of upper triangle in causal attention: {upper_triangle_sum:.10f}")
    if upper_triangle_sum < 1e-6:
        print("  ✓ Verified: No attention to future positions!\n")
    else:
        print(f"  ✗ Warning: Some future attention detected!\n")


def visualize_multihead_causal():
    """
    Show causal attention for each head separately.

    Demonstrates that masking applies uniformly to all heads.
    """
    print_section("Visualizing Multi-Head Causal Attention")

    # Load comparison data
    data_path = DATA_PROCESSED_DIR / "phase4_causal_comparison.pt"

    if not data_path.exists():
        print(f"Error: Comparison data not found at {data_path}")
        print("Please run causal_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attn_causal = data['attention_causal'].squeeze(0).detach().cpu().numpy()  # (n_heads, seq_len, seq_len)
    tokens = data['tokens']
    n_heads = data['n_heads']

    print(f"Visualizing causal attention for {n_heads} heads\n")

    # Setup plotting
    setup_plotting_style()

    # Create grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for head_idx in range(n_heads):
        head_attn = attn_causal[head_idx]

        # Plot on subplot
        ax = axes[head_idx]
        plot_attention_heatmap(
            head_attn,
            tokens,
            title=f'Head {head_idx + 1} - Causal Attention',
            ax=ax
        )

        # Verify upper triangle is zero
        upper_triangle_sum = np.triu(head_attn, k=1).sum()
        print(f"Head {head_idx + 1}: Upper triangle sum = {upper_triangle_sum:.10f}")

    fig.suptitle('Multi-Head Causal Attention\n(All heads respect causality)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    save_figure(fig, "multihead_causal.png", subdir="phase4", dpi=300)

    print("\n✓ Multi-head causal visualization complete!")
    print("  All heads show triangular pattern (upper triangle = 0)")
    print("  Causal mask applies uniformly to all heads\n")


def create_all_visualizations():
    """Create all Phase 4 visualizations."""
    print_section("Phase 4: Causal Masking Visualizations", char="=")

    # Create visualizations
    visualize_causal_mask_structure()
    visualize_bidirectional_vs_causal_comparison()
    visualize_multihead_causal()

    print_section("All Phase 4 Visualizations Complete!", char="=")
    print("Output files:")
    print("  - visualizations/plots/03_attention/phase4/causal_mask_structure.png")
    print("  - visualizations/plots/03_attention/phase4/bidirectional_vs_causal.png")
    print("  - visualizations/plots/03_attention/phase4/multihead_causal.png")
    print()
    print("Key insights:")
    print("  1. Causal mask creates triangular attention pattern")
    print("  2. Upper triangle is completely zeroed out")
    print("  3. Prevents information leakage from future positions")
    print("  4. Essential for autoregressive generation (GPT models)")
    print()
    print("🎉 All 4 phases complete!")
    print()
    print("Summary of what you've learned:")
    print("  • Phase 1: Single-head self-attention mechanism")
    print("  • Phase 2: Multi-head attention and specialization")
    print("  • Phase 3: Advanced visualization techniques")
    print("  • Phase 4: Causal masking for generation")
    print()
    print("Next chapter: Transformers, QKV, and Stacking")
    print()


if __name__ == "__main__":
    create_all_visualizations()
