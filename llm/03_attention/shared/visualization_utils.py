"""
Visualization Utilities for Attention

Helper functions for creating attention visualizations:
- Heatmaps with token labels
- Multi-head grids
- Flow diagrams
- 3D surfaces
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Optional, List, Tuple
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    title: str = "Attention Weights",
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (8, 7)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot attention weights as a heatmap with token labels.

    Args:
        attention_weights: Attention matrix (seq_len, seq_len)
        tokens: List of token strings for labels
        title: Plot title
        ax: Existing axes to plot on (if None, creates new figure)
        cmap: Colormap name
        figsize: Figure size (only used if ax is None)

    Returns:
        fig, ax: matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create heatmap
    im = ax.imshow(attention_weights, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)

    # Labels
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)

    # Add grid for readability
    ax.set_xticks(np.arange(len(tokens)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(tokens)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)

    # Annotate cells with values
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    return fig, ax


def plot_multihead_grid(
    attention_weights: np.ndarray,
    tokens: List[str],
    n_heads: int,
    title: str = "Multi-Head Attention",
    figsize: Tuple[int, int] = (16, 4),
    cmap: str = "YlOrRd"
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot multiple attention heads in a grid layout.

    Args:
        attention_weights: Attention tensor (n_heads, seq_len, seq_len)
        tokens: List of token strings
        n_heads: Number of attention heads
        title: Overall plot title
        figsize: Figure size
        cmap: Colormap name

    Returns:
        fig, axes: matplotlib figure and array of axes
    """
    # Create grid of subplots
    fig, axes = plt.subplots(1, n_heads, figsize=figsize)

    # Ensure axes is iterable (in case n_heads=1)
    if n_heads == 1:
        axes = [axes]

    # Plot each head
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        attn = attention_weights[head_idx]

        # Create heatmap
        im = ax.imshow(attn, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Set labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(tokens, fontsize=9)

        ax.set_title(f'Head {head_idx + 1}', fontsize=11, fontweight='bold')

        # Only show y-label on first subplot
        if head_idx == 0:
            ax.set_ylabel('Query', fontsize=10)

        # Show x-label on all
        ax.set_xlabel('Key', fontsize=10)

        # Add grid
        ax.set_xticks(np.arange(len(tokens)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(tokens)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

    # Add main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig, axes


def plot_qkv_flow(
    tokens: List[str],
    query_idx: int,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scores: np.ndarray,
    attention_weights: np.ndarray,
    output: np.ndarray,
    figsize: Tuple[int, int] = (14, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the Query-Key-Value flow for a single query token.

    Shows the complete attention mechanism:
    1. Input -> Q, K, V projections
    2. Q·K scores
    3. Softmax -> attention weights
    4. Weighted sum of V -> output

    Args:
        tokens: List of tokens
        query_idx: Index of query token to visualize
        Q: Query vectors (seq_len, d_k)
        K: Key vectors (seq_len, d_k)
        V: Value vectors (seq_len, d_v)
        scores: Attention scores before softmax (seq_len,)
        attention_weights: Attention weights after softmax (seq_len,)
        output: Final output vector (d_v,)
        figsize: Figure size

    Returns:
        fig, ax: matplotlib figure and axes
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Attention Mechanism for Query Token: "{tokens[query_idx]}" (position {query_idx})',
                 fontsize=14, fontweight='bold')

    # 1. Query vector
    ax = axes[0, 0]
    q_vec = Q[query_idx]
    ax.bar(range(len(q_vec)), q_vec, color='skyblue', alpha=0.7)
    ax.set_title('1. Query Vector (Q)', fontweight='bold')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # 2. Attention scores (Q·K)
    ax = axes[0, 1]
    colors = ['red' if i == query_idx else 'steelblue' for i in range(len(tokens))]
    ax.bar(range(len(scores)), scores, color=colors, alpha=0.7)
    ax.set_title('2. Attention Scores (Q·K)', fontweight='bold')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)

    # 3. Attention weights (after softmax)
    ax = axes[0, 2]
    ax.bar(range(len(attention_weights)), attention_weights, color=colors, alpha=0.7)
    ax.set_title('3. Attention Weights (softmax)', fontweight='bold')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel('Weight')
    ax.set_ylim([0, 1])
    ax.axhline(y=1.0/len(tokens), color='gray', linestyle='--', label='Uniform')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Value vectors (heatmap)
    ax = axes[1, 0]
    im = ax.imshow(V.T, cmap='coolwarm', aspect='auto')
    ax.set_title('4. Value Vectors (V)', fontweight='bold')
    ax.set_xlabel('Token Position')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax, label='Value')

    # 5. Weighted values (V weighted by attention)
    ax = axes[1, 1]
    weighted_V = V * attention_weights[:, np.newaxis]  # Broadcasting
    im = ax.imshow(weighted_V.T, cmap='coolwarm', aspect='auto')
    ax.set_title('5. Weighted Values (attn × V)', fontweight='bold')
    ax.set_xlabel('Token Position')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax, label='Weighted Value')

    # 6. Output vector (sum of weighted values)
    ax = axes[1, 2]
    ax.bar(range(len(output)), output, color='darkgreen', alpha=0.7)
    ax.set_title('6. Output Vector (Σ weighted V)', fontweight='bold')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_attention_flow_arrows(
    attention_weights: np.ndarray,
    tokens: List[str],
    threshold: float = 0.1,
    title: str = "Attention Flow",
    figsize: Tuple[int, int] = (14, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot attention as arrows between tokens.

    Shows information flow: arrow from key to query with thickness = attention weight.

    Args:
        attention_weights: Attention matrix (seq_len, seq_len)
        tokens: List of tokens
        threshold: Minimum weight to draw arrow (filters weak connections)
        title: Plot title
        figsize: Figure size

    Returns:
        fig, ax: matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    seq_len = len(tokens)

    # Position tokens horizontally
    token_positions = np.arange(seq_len)
    y_level = 0.5

    # Draw tokens as circles
    for i, (pos, token) in enumerate(zip(token_positions, tokens)):
        circle = plt.Circle((pos, y_level), 0.15, color='lightblue', ec='black', zorder=3)
        ax.add_patch(circle)
        ax.text(pos, y_level, token, ha='center', va='center',
                fontsize=11, fontweight='bold', zorder=4)

    # Draw attention arrows
    for query_idx in range(seq_len):
        for key_idx in range(seq_len):
            weight = attention_weights[query_idx, key_idx]

            # Only draw if above threshold and not self-attention
            if weight > threshold and query_idx != key_idx:
                # Arrow from key to query
                x_start = token_positions[key_idx]
                x_end = token_positions[query_idx]

                # Curved arrow (arc above or below)
                y_offset = 0.3 if key_idx < query_idx else -0.3

                arrow = FancyArrowPatch(
                    (x_start, y_level), (x_end, y_level),
                    arrowstyle='->', mutation_scale=20,
                    connectionstyle=f"arc3,rad={y_offset}",
                    color='red', alpha=weight, linewidth=weight * 3,
                    zorder=2
                )
                ax.add_patch(arrow)

                # Add weight label
                mid_x = (x_start + x_end) / 2
                mid_y = y_level + y_offset * 0.5
                ax.text(mid_x, mid_y, f'{weight:.2f}',
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Set axis limits and labels
    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig, ax


def plot_3d_attention_surface(
    attention_weights: np.ndarray,
    tokens: List[str],
    title: str = "3D Attention Landscape",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis'
) -> Tuple[plt.Figure, Axes3D]:
    """
    Plot attention weights as a 3D surface.

    Args:
        attention_weights: Attention matrix (seq_len, seq_len)
        tokens: List of tokens
        title: Plot title
        figsize: Figure size
        cmap: Colormap name

    Returns:
        fig, ax: matplotlib figure and 3D axes
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    seq_len = len(tokens)

    # Create meshgrid
    X, Y = np.meshgrid(range(seq_len), range(seq_len))

    # Plot surface
    surf = ax.plot_surface(X, Y, attention_weights, cmap=cmap,
                           alpha=0.8, edgecolor='none')

    # Set labels
    ax.set_xlabel('Key Position', fontsize=11, labelpad=10)
    ax.set_ylabel('Query Position', fontsize=11, labelpad=10)
    ax.set_zlabel('Attention Weight', fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tokens)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    return fig, ax


def create_rotating_3d_animation(
    attention_weights: np.ndarray,
    tokens: List[str],
    output_path: str,
    title: str = "3D Attention Landscape",
    n_frames: int = 60,
    fps: int = 20,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Create a rotating 3D animation of attention weights.

    Args:
        attention_weights: Attention matrix (seq_len, seq_len)
        tokens: List of tokens
        output_path: Path to save GIF
        title: Plot title
        n_frames: Number of frames in rotation
        fps: Frames per second
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    seq_len = len(tokens)

    # Create meshgrid
    X, Y = np.meshgrid(range(seq_len), range(seq_len))

    # Plot surface
    surf = ax.plot_surface(X, Y, attention_weights, cmap='viridis',
                           alpha=0.8, edgecolor='none')

    # Set labels
    ax.set_xlabel('Key Position', fontsize=11, labelpad=10)
    ax.set_ylabel('Query Position', fontsize=11, labelpad=10)
    ax.set_zlabel('Attention Weight', fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, fontsize=8)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tokens, fontsize=8)

    # Animation function
    def animate(frame):
        ax.view_init(elev=30, azim=frame * 360 / n_frames)
        return [surf]

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=1000/fps, blit=False)

    # Save
    anim.save(output_path, writer='pillow', fps=fps)
    print(f"Saved animation to: {output_path}")

    plt.close(fig)


def plot_entropy_bars(
    entropies: np.ndarray,
    n_heads: int,
    title: str = "Attention Entropy by Head",
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot bar chart of attention entropy for each head.

    Args:
        entropies: Mean entropy per head (n_heads,)
        n_heads: Number of heads
        title: Plot title
        figsize: Figure size

    Returns:
        fig, ax: matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    head_labels = [f'Head {i+1}' for i in range(n_heads)]
    colors = plt.cm.viridis(np.linspace(0, 1, n_heads))

    bars = ax.bar(head_labels, entropies, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, entropy in zip(bars, entropies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{entropy:.2f}',
                ha='center', va='bottom', fontweight='bold')

    # Labels
    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Average Entropy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add reference line for maximum entropy (uniform distribution)
    # Max entropy = log(seq_len) for uniform distribution
    # We'll add this if we know seq_len, but skip for now

    plt.tight_layout()
    return fig, ax
