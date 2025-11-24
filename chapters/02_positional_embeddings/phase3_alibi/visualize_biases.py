"""
Visualize ALiBi Biases

Creates visualizations for ALiBi:
- Bias heatmaps for different heads
- Slope comparisons across heads
- Attention patterns with ALiBi
- Distance vs bias plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    setup_plotting_style,
    save_figure,
    DATA_PROCESSED_DIR,
    VIZ_PLOTS_DIR,
    print_section
)
from shared.attention import get_alibi_bias, get_alibi_slopes


def plot_bias_heatmaps(seq_len: int = 12, n_heads: int = 8,
                       filename: str = "alibi_bias_heatmaps.png"):
    """
    Plot ALiBi bias heatmaps for multiple heads.

    Args:
        seq_len: Sequence length
        n_heads: Number of heads
        filename: Output filename
    """
    print("Creating ALiBi bias heatmaps...")

    bias = get_alibi_bias(seq_len, n_heads)
    slopes = get_alibi_slopes(n_heads)

    # Plot first 4 heads
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, h in enumerate([0, 2, 4, 7]):
        ax = axes[idx]

        im = ax.imshow(bias[h].detach().numpy(), cmap='RdBu_r', aspect='auto')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Head {h} (slope = {slopes[h].item():.4f})')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Bias Value')

        # Add grid
        ax.grid(alpha=0.2, color='black')

    plt.tight_layout()
    save_figure(fig, filename, subdir='phase3')
    plt.close()


def plot_slope_comparison(n_heads: int = 8, seq_len: int = 20,
                         filename: str = "alibi_slope_comparison.png"):
    """
    Plot bias curves for different head slopes.

    Args:
        n_heads: Number of heads
        seq_len: Sequence length
        filename: Output filename
    """
    print("Creating slope comparison plot...")

    slopes = get_alibi_slopes(n_heads)

    fig, ax = plt.subplots(figsize=(12, 7))

    # For each head, plot bias as function of distance
    distances = np.arange(seq_len)
    colors = plt.cm.viridis(np.linspace(0, 1, n_heads))

    for h in range(n_heads):
        m = slopes[h].item()
        biases = -m * distances
        ax.plot(distances, biases, marker='o', linewidth=2,
               color=colors[h], label=f'Head {h} (m={m:.4f})')

    ax.set_xlabel('Distance |query_pos - key_pos|')
    ax.set_ylabel('Bias Value')
    ax.set_title('ALiBi: Different Slopes for Different Heads\n'
                 '(Enables multi-scale attention: local vs global)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    save_figure(fig, filename, subdir='phase3')
    plt.close()


def plot_attention_patterns(seq_len: int = 12, n_heads: int = 8,
                           filename: str = "alibi_attention_patterns.png"):
    """
    Visualize attention patterns with ALiBi.

    Args:
        seq_len: Sequence length
        n_heads: Number of heads
        filename: Output filename
    """
    print("Creating attention pattern visualization...")

    # Create dummy attention scores (uniform before ALiBi)
    torch.manual_seed(42)
    raw_scores = torch.randn(1, n_heads, seq_len, seq_len) * 0.5  # Small variance

    # Get ALiBi bias
    bias = get_alibi_bias(seq_len, n_heads)

    # Add bias
    scores_with_alibi = raw_scores + bias.unsqueeze(0)

    # Apply softmax
    attn_no_bias = torch.softmax(raw_scores, dim=-1)[0]
    attn_with_bias = torch.softmax(scores_with_alibi, dim=-1)[0]

    # Plot comparison for selected heads
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    heads_to_plot = [0, 2, 4, 7]

    for idx, h in enumerate(heads_to_plot):
        # Without ALiBi
        ax1 = axes[0, idx]
        im1 = ax1.imshow(attn_no_bias[h].detach().numpy(), cmap='viridis', vmin=0, vmax=0.3)
        ax1.set_title(f'Head {h}: Without ALiBi')
        ax1.set_xlabel('Key')
        ax1.set_ylabel('Query')
        plt.colorbar(im1, ax=ax1)

        # With ALiBi
        ax2 = axes[1, idx]
        slopes = get_alibi_slopes(n_heads)
        im2 = ax2.imshow(attn_with_bias[h].detach().numpy(), cmap='viridis', vmin=0, vmax=0.3)
        ax2.set_title(f'Head {h}: With ALiBi\n(slope={slopes[h].item():.4f})')
        ax2.set_xlabel('Key')
        ax2.set_ylabel('Query')
        plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    save_figure(fig, filename, subdir='phase3')
    plt.close()


def plot_3d_attention_landscape(seq_len: int = 12, head: int = 0,
                                filename: str = "alibi_3d_landscape.png"):
    """
    Create 3D visualization of attention landscape with ALiBi.

    Args:
        seq_len: Sequence length
        head: Which head to visualize
        filename: Output filename
    """
    print("Creating 3D attention landscape...")

    # Get bias for one head
    bias = get_alibi_bias(seq_len, 8)
    slopes = get_alibi_slopes(8)

    # Create dummy scores
    torch.manual_seed(42)
    raw_scores = torch.randn(seq_len, seq_len) * 0.3

    # Add ALiBi bias
    scores_with_bias = raw_scores + bias[head]

    # Apply softmax
    attention = torch.softmax(scores_with_bias, dim=-1)

    # Create meshgrid
    X, Y = np.meshgrid(np.arange(seq_len), np.arange(seq_len))

    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, attention.detach().numpy(), cmap='viridis',
                          alpha=0.8, edgecolor='none')

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_zlabel('Attention Weight')
    ax.set_title(f'ALiBi Attention Landscape\n'
                f'Head {head} (slope = {slopes[head].item():.4f})')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5)

    # View angle
    ax.view_init(elev=20, azim=45)

    save_figure(fig, filename, subdir='phase3')
    plt.close()


def plot_distance_decay(n_heads: int = 8, max_distance: int = 50,
                       filename: str = "alibi_distance_decay.png"):
    """
    Plot how bias decays with distance for different heads.

    Args:
        n_heads: Number of heads
        max_distance: Maximum distance to plot
        filename: Output filename
    """
    print("Creating distance decay plot...")

    slopes = get_alibi_slopes(n_heads)
    distances = np.arange(max_distance)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale
    colors = plt.cm.viridis(np.linspace(0, 1, n_heads))
    for h in range(n_heads):
        m = slopes[h].item()
        biases = -m * distances
        ax1.plot(distances, biases, linewidth=2, color=colors[h],
                label=f'Head {h}')

    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Bias Value')
    ax1.set_title('ALiBi Bias vs Distance (Linear Scale)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5)

    # Log scale (absolute value)
    for h in range(n_heads):
        m = slopes[h].item()
        biases = np.abs(-m * distances)
        # Avoid log(0)
        biases[0] = 1e-10
        ax2.plot(distances, biases, linewidth=2, color=colors[h],
                label=f'Head {h}')

    ax2.set_xlabel('Distance')
    ax2.set_ylabel('|Bias Value|')
    ax2.set_title('ALiBi Bias vs Distance (Log Scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, filename, subdir='phase3')
    plt.close()


def plot_multi_scale_attention(seq_len: int = 20, n_heads: int = 8,
                               filename: str = "alibi_multi_scale.png"):
    """
    Demonstrate multi-scale attention with different slopes.

    Args:
        seq_len: Sequence length
        n_heads: Number of heads
        filename: Output filename
    """
    print("Creating multi-scale attention visualization...")

    slopes = get_alibi_slopes(n_heads)
    query_pos = seq_len // 2  # Middle position

    fig, ax = plt.subplots(figsize=(12, 7))

    # For each head, show attention distribution from middle position
    key_positions = np.arange(seq_len)
    colors = plt.cm.viridis(np.linspace(0, 1, n_heads))

    for h in range(n_heads):
        # Compute biases
        distances = np.abs(key_positions - query_pos)
        biases = -slopes[h].item() * distances

        # Softmax (simplified, uniform raw scores)
        raw_scores = np.zeros(seq_len)
        scores = raw_scores + biases
        attention = np.exp(scores) / np.sum(np.exp(scores))

        ax.plot(key_positions, attention, marker='o', linewidth=2,
               color=colors[h], label=f'Head {h} (slope={slopes[h].item():.3f})',
               markersize=6)

    # Mark query position
    ax.axvline(x=query_pos, color='red', linestyle='--', linewidth=2,
              alpha=0.5, label=f'Query position ({query_pos})')

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'Multi-Scale Attention with ALiBi\n'
                f'(Query at position {query_pos}, different heads focus at different scales)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, filename, subdir='phase3')
    plt.close()


def visualize_all():
    """Generate all ALiBi visualizations."""
    print_section("Generating Phase 3 (ALiBi) Visualizations")

    setup_plotting_style()

    # Check if biases are available
    bias_path = DATA_PROCESSED_DIR / "alibi_biases.pt"
    if not bias_path.exists():
        print(f"WARNING: {bias_path} not found. Run alibi.py first.")
        print("Generating visualizations with default parameters...\n")

    # Create visualizations
    plot_bias_heatmaps()
    plot_slope_comparison()
    plot_attention_patterns()
    plot_3d_attention_landscape()
    plot_distance_decay()
    plot_multi_scale_attention()

    print("\n" + "=" * 80)
    print("All ALiBi visualizations complete!")
    print(f"Check {VIZ_PLOTS_DIR / 'phase3'} for plots")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize ALiBi Biases")
    args = parser.parse_args()

    visualize_all()


if __name__ == "__main__":
    main()
