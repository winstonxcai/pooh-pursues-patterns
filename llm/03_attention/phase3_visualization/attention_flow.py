"""
Phase 3: Attention Flow Diagrams

Advanced visualization showing attention as information flow between tokens.

Instead of heatmaps, this visualization shows:
- Tokens as nodes
- Attention weights as arrows (thickness = weight)
- Only shows significant connections (filters weak attention)

This makes it easy to see at a glance which tokens are influencing which others.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (DATA_PROCESSED_DIR, print_section, save_figure,
                          setup_plotting_style)
from shared.visualization_utils import plot_attention_flow_arrows


def visualize_single_head_flow():
    """
    Create attention flow diagram for single-head attention.

    Shows information flow as arrows between tokens.
    Arrow thickness corresponds to attention weight.
    """
    print_section("Visualizing Single-Head Attention Flow")

    # Load Phase 1 data (single-head attention)
    data_path = DATA_PROCESSED_DIR / "phase1_attention_weights.pt"

    if not data_path.exists():
        print(f"Error: Single-head attention data not found at {data_path}")
        print("Please run phase1_single_head/self_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attention_weights = data['attention_weights'].squeeze(0).detach().cpu().numpy()  # (seq_len, seq_len)
    tokens = data['tokens']

    print(f"Loaded attention weights: shape {attention_weights.shape}")
    print(f"Tokens: {tokens}\n")

    # Count significant connections
    threshold = 0.15
    n_significant = (attention_weights > threshold).sum()
    total_connections = attention_weights.size

    print(f"Threshold for display: {threshold:.2f}")
    print(f"Significant connections: {n_significant} / {total_connections}")
    print(f"Showing {n_significant / total_connections * 100:.1f}% of connections\n")

    # Setup plotting
    setup_plotting_style()

    # Create flow diagram
    fig, ax = plot_attention_flow_arrows(
        attention_weights,
        tokens,
        threshold=threshold,
        title="Single-Head Attention Flow (threshold=0.15)",
        figsize=(14, 6)
    )

    # Save figure
    save_figure(fig, "attention_flow.png", subdir="phase3", dpi=300)

    print("\n✓ Attention flow diagram complete!")
    print("  Interpretation:")
    print("  - Arrows point from key (source) to query (target)")
    print("  - Thicker arrows = stronger attention")
    print("  - Curved arrows avoid overlap")
    print("  - Missing arrows = attention weight below threshold\n")


def visualize_multihead_flow():
    """
    Create attention flow diagrams for each head in multi-head attention.

    Shows how different heads have different flow patterns.
    """
    print_section("Visualizing Multi-Head Attention Flow")

    # Load Phase 2 data (multi-head attention)
    data_path = DATA_PROCESSED_DIR / "phase2_multihead_attention.pt"

    if not data_path.exists():
        print(f"Error: Multi-head attention data not found at {data_path}")
        print("Please run phase2_multi_head/multi_head_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attention_weights = data['attention_weights'].squeeze(0).detach().cpu().numpy()  # (n_heads, seq_len, seq_len)
    tokens = data['tokens']
    n_heads = data['n_heads']

    print(f"Loaded multi-head attention weights: shape {attention_weights.shape}")
    print(f"Number of heads: {n_heads}")
    print(f"Tokens: {tokens}\n")

    threshold = 0.15

    # Create flow diagram for each head
    import matplotlib.pyplot as plt

    # Create subplots for all heads
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for head_idx in range(n_heads):
        head_attn = attention_weights[head_idx]  # (seq_len, seq_len)

        # Count significant connections for this head
        n_significant = (head_attn > threshold).sum()
        total = head_attn.size

        print(f"Head {head_idx + 1}:")
        print(f"  Significant connections: {n_significant} / {total}")
        print(f"  Density: {n_significant / total * 100:.1f}%\n")

        # Plot on subplot
        ax = axes[head_idx]

        # Use simplified version since we're making subplots
        seq_len = len(tokens)
        token_positions = np.arange(seq_len)
        y_level = 0.5

        # Draw tokens
        for i, (pos, token) in enumerate(zip(token_positions, tokens)):
            circle = plt.Circle((pos, y_level), 0.12, color='lightblue',
                              ec='black', zorder=3)
            ax.add_patch(circle)
            ax.text(pos, y_level, token, ha='center', va='center',
                   fontsize=10, fontweight='bold', zorder=4)

        # Draw arrows for significant attention
        from matplotlib.patches import FancyArrowPatch

        for query_idx in range(seq_len):
            for key_idx in range(seq_len):
                weight = head_attn[query_idx, key_idx]

                if weight > threshold and query_idx != key_idx:
                    x_start = token_positions[key_idx]
                    x_end = token_positions[query_idx]
                    y_offset = 0.25 if key_idx < query_idx else -0.25

                    arrow = FancyArrowPatch(
                        (x_start, y_level), (x_end, y_level),
                        arrowstyle='->', mutation_scale=15,
                        connectionstyle=f"arc3,rad={y_offset}",
                        color='red', alpha=weight, linewidth=weight * 2.5,
                        zorder=2
                    )
                    ax.add_patch(arrow)

        # Styling
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.set_ylim(-0.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Head {head_idx + 1}', fontsize=13, fontweight='bold')

    fig.suptitle('Multi-Head Attention Flow Diagrams (threshold=0.15)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save figure
    save_figure(fig, "multihead_flow.png", subdir="phase3", dpi=300)

    print("\n✓ Multi-head flow diagrams complete!")
    print("  Compare heads to see different flow patterns:")
    print("  - Which heads are sparse (few arrows)?")
    print("  - Which heads are dense (many arrows)?")
    print("  - Do different heads show different structures?\n")


def create_all_flow_visualizations():
    """Create all attention flow visualizations."""
    print_section("Phase 3: Attention Flow Visualizations", char="=")

    # Create visualizations
    visualize_single_head_flow()
    visualize_multihead_flow()

    print_section("Flow Visualizations Complete!", char="=")
    print("Output files:")
    print("  - visualizations/plots/03_attention/phase3/attention_flow.png")
    print("  - visualizations/plots/03_attention/phase3/multihead_flow.png")
    print()
    print("Key insights:")
    print("  1. Flow diagrams reveal attention structure at a glance")
    print("  2. Different heads have different flow patterns")
    print("  3. Some heads are focused (few arrows), others diffuse (many arrows)")
    print()
    print("Next step:")
    print("  Run attention_3d.py for 3D attention landscapes")
    print()


if __name__ == "__main__":
    create_all_flow_visualizations()
