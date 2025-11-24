"""
Phase 2 Visualizations: Multi-Head Self-Attention

Creates visualizations to understand multi-head attention:
1. Multi-head grid - shows all heads side-by-side
2. Attention entropy by head - reveals head specialization

These visualizations reveal how different heads learn different attention patterns.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import math

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    setup_plotting_style,
    save_figure,
    print_section,
    DATA_PROCESSED_DIR
)
from shared.visualization_utils import (
    plot_multihead_grid,
    plot_entropy_bars
)
from shared.attention_utils import compute_attention_entropy


def visualize_multihead_grid():
    """
    Create multi-head attention grid visualization.

    Shows all attention heads side-by-side, revealing:
    - How attention patterns differ across heads
    - Which heads are focused vs diffuse
    - Whether heads have specialized to different patterns
    """
    print_section("Visualizing Multi-Head Attention Grid")

    # Load saved attention weights
    data_path = DATA_PROCESSED_DIR / "phase2_multihead_attention.pt"

    if not data_path.exists():
        print(f"Error: Attention weights not found at {data_path}")
        print("Please run multi_head_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attention_weights = data['attention_weights'].squeeze(0).numpy()  # (n_heads, seq_len, seq_len)
    tokens = data['tokens']
    n_heads = data['n_heads']

    print(f"Loaded multi-head attention weights: shape {attention_weights.shape}")
    print(f"Number of heads: {n_heads}")
    print(f"Tokens: {tokens}\n")

    # Setup plotting
    setup_plotting_style()

    # Create multi-head grid
    fig, axes = plot_multihead_grid(
        attention_weights,
        tokens,
        n_heads,
        title="Multi-Head Self-Attention Patterns",
        figsize=(18, 5)
    )

    # Save figure
    save_figure(fig, "multi_head_grid.png", subdir="phase2", dpi=300)

    print("\n✓ Multi-head grid visualization complete!")
    print("  Key observations to look for:")
    print("  - Head diversity: Do heads show different patterns?")
    print("  - Specialization: Does any head focus on specific relationships?")
    print("  - Self-attention: Which heads have strong diagonals?")
    print("  - Sparsity: Which heads are focused vs uniform?\n")


def visualize_attention_entropy():
    """
    Create attention entropy bar chart.

    Entropy measures how "spread out" attention is:
    - Low entropy = focused/peaked attention (few tokens get most weight)
    - High entropy = uniform/diffuse attention (many tokens get similar weight)

    Different heads often have very different entropy levels,
    indicating specialization.
    """
    print_section("Visualizing Attention Entropy by Head")

    # Load saved attention weights
    data_path = DATA_PROCESSED_DIR / "phase2_multihead_attention.pt"

    if not data_path.exists():
        print(f"Error: Attention weights not found at {data_path}")
        print("Please run multi_head_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attention_weights = data['attention_weights']  # (1, n_heads, seq_len, seq_len)
    tokens = data['tokens']
    n_heads = data['n_heads']

    seq_len = len(tokens)

    print(f"Computing entropy for {n_heads} heads...\n")

    # Compute entropy for each head
    entropies = []
    max_entropy = math.log(seq_len)  # Maximum possible entropy (uniform distribution)

    for head_idx in range(n_heads):
        # Get attention for this head
        head_attn = attention_weights[0, head_idx, :, :]  # (seq_len, seq_len)

        # Compute entropy for each query position
        head_entropy = compute_attention_entropy(head_attn.unsqueeze(0)).squeeze()  # (seq_len,)

        # Average over query positions
        mean_entropy = head_entropy.mean().item()
        entropies.append(mean_entropy)

        # Characterize
        normalized = mean_entropy / max_entropy
        if normalized < 0.3:
            pattern = "FOCUSED"
        elif normalized > 0.7:
            pattern = "DIFFUSE"
        else:
            pattern = "MODERATE"

        print(f"Head {head_idx + 1}:")
        print(f"  Entropy: {mean_entropy:.3f} / {max_entropy:.3f}")
        print(f"  Normalized: {normalized:.3f}")
        print(f"  Pattern: {pattern}")
        print()

    entropies = np.array(entropies)

    # Setup plotting
    setup_plotting_style()

    # Create entropy bar chart
    fig, ax = plot_entropy_bars(
        entropies,
        n_heads,
        title="Attention Entropy by Head",
        figsize=(10, 6)
    )

    # Add horizontal line for maximum entropy
    ax.axhline(y=max_entropy, color='red', linestyle='--', linewidth=2,
               label=f'Max entropy (uniform): {max_entropy:.2f}')
    ax.legend()

    # Save figure
    save_figure(fig, "attention_entropy.png", subdir="phase2")

    print("\n✓ Entropy visualization complete!")
    print("  Interpretation:")
    print("  - Low entropy heads: Specialized, focused attention")
    print("  - High entropy heads: General-purpose, broad attention")
    print("  - Diversity in entropy = heads have specialized!\n")


def analyze_head_patterns():
    """
    Additional analysis: characterize what each head is doing.
    """
    print_section("Analyzing Head Patterns")

    # Load saved attention weights
    data_path = DATA_PROCESSED_DIR / "phase2_multihead_attention.pt"

    if not data_path.exists():
        print(f"Error: Attention weights not found at {data_path}")
        return

    # Load data
    data = torch.load(data_path)
    attention_weights = data['attention_weights'].squeeze(0).numpy()  # (n_heads, seq_len, seq_len)
    tokens = data['tokens']
    n_heads = data['n_heads']

    print("Pattern analysis for each head:\n")

    for head_idx in range(n_heads):
        head_attn = attention_weights[head_idx]  # (seq_len, seq_len)

        print(f"Head {head_idx + 1}:")

        # Check for diagonal pattern (self-attention)
        diagonal_strength = np.mean(np.diag(head_attn))
        print(f"  Self-attention strength: {diagonal_strength:.3f}")

        # Check for previous-token pattern
        if head_attn.shape[0] > 1:
            prev_token_strength = np.mean(np.diag(head_attn, k=-1))
            print(f"  Previous-token attention: {prev_token_strength:.3f}")

        # Check for first-token pattern (attention to position 0)
        first_token_strength = np.mean(head_attn[:, 0])
        print(f"  First-token attention: {first_token_strength:.3f}")

        # Check for uniformity (variance in attention)
        attention_var = np.var(head_attn)
        print(f"  Attention variance: {attention_var:.4f}")

        # Most attended token pair
        max_idx = np.unravel_index(np.argmax(head_attn), head_attn.shape)
        max_weight = head_attn[max_idx]
        print(f"  Strongest attention: '{tokens[max_idx[0]]}' → '{tokens[max_idx[1]]}' ({max_weight:.3f})")

        print()

    print("=" * 70)
    print("\nCommon patterns to look for:")
    print("  - High diagonal: Self-attention (tokens attend to themselves)")
    print("  - High previous-token: Sequential dependencies")
    print("  - High first-token: Global context gathering")
    print("  - Low variance: Uniform attention (head is 'confused')")
    print("  - High variance + low entropy: Focused, specialized head")
    print()


def create_all_visualizations():
    """Create all Phase 2 visualizations."""
    print_section("Phase 2: Multi-Head Attention Visualizations", char="=")

    # Create visualizations
    visualize_multihead_grid()
    visualize_attention_entropy()
    analyze_head_patterns()

    print_section("All Phase 2 Visualizations Complete!", char="=")
    print("Output files:")
    print("  - visualizations/plots/03_attention/phase2/multi_head_grid.png")
    print("  - visualizations/plots/03_attention/phase2/attention_entropy.png")
    print()
    print("Key insights:")
    print("  1. Different heads show different attention patterns")
    print("  2. Entropy varies across heads (specialization!)")
    print("  3. Multi-head is richer than single-head")
    print()
    print("Next steps:")
    print("  1. Compare the multi-head grid to Phase 1's single-head heatmap")
    print("  2. Examine which heads are focused vs diffuse")
    print("  3. Move to Phase 3: python phase3_visualization/attention_flow.py")
    print()


if __name__ == "__main__":
    create_all_visualizations()
