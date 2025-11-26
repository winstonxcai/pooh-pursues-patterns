"""
Visualize RoPE Rotations

Creates visualizations showing:
- 2D rotation of vectors at different positions
- 3D projection of rotated embeddings
- Relative position encoding in attention scores
- Animated rotations
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    setup_plotting_style,
    save_figure,
    DATA_PROCESSED_DIR,
    VIZ_PLOTS_DIR,
    VIZ_ANIMATIONS_DIR,
    print_section
)
from shared.attention import precompute_rope_frequencies, apply_rotary_embedding


def plot_2d_rotation(max_positions: int = 6, filename: str = "rope_2d_rotation.png"):
    """
    Visualize 2D rotation for different positions.

    Shows how a base vector [1, 0] is rotated at different positions.
    """
    print("Creating 2D rotation visualization...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create base vector in 2D
    base_vector = np.array([1.0, 0.0])

    # Compute rotation angles for different positions
    # Using RoPE frequency for first dimension pair
    d_k = 16  # Dimension per head
    freqs = precompute_rope_frequencies(max_positions, d_k, device=torch.device('cpu'))

    # Extract angles for first dimension pair
    angles = freqs[:, 0].detach().numpy()

    # Plot rotated vectors
    colors = plt.cm.viridis(np.linspace(0, 1, max_positions))

    for pos in range(max_positions):
        angle = angles[pos]

        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a],
                              [sin_a, cos_a]])

        # Rotate vector
        rotated = rot_matrix @ base_vector

        # Plot
        ax.arrow(0, 0, rotated[0], rotated[1],
                head_width=0.05, head_length=0.08,
                fc=colors[pos], ec=colors[pos],
                linewidth=2, alpha=0.7,
                label=f'Pos {pos} ({angle*180/np.pi:.1f}°)')

    # Styling
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.set_title('RoPE: 2D Rotation at Different Positions\n(First dimension pair)')
    ax.legend(loc='upper right', fontsize=8)

    # Add circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.3)
    ax.add_patch(circle)

    save_figure(fig, filename, subdir='phase2')
    plt.close()


def create_rotation_animation(max_positions: int = 12, filename: str = "rope_2d_animation.gif"):
    """
    Create animated visualization of RoPE rotation.

    Shows vector rotating as position increases.
    """
    print("Creating 2D rotation animation...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Compute frequencies
    d_k = 16
    freqs = precompute_rope_frequencies(max_positions, d_k, device=torch.device('cpu'))
    angles = freqs[:, 0].detach().numpy()

    base_vector = np.array([1.0, 0.0])

    def update(frame):
        ax.clear()

        # Plot previous positions with fading
        colors = plt.cm.viridis(np.linspace(0, 1, max_positions))

        for pos in range(frame + 1):
            angle = angles[pos]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a],
                                  [sin_a, cos_a]])
            rotated = rot_matrix @ base_vector

            alpha = 0.3 + 0.7 * (pos / frame) if frame > 0 else 1.0
            linewidth = 1 + 2 * (pos / frame) if frame > 0 else 3

            ax.arrow(0, 0, rotated[0], rotated[1],
                    head_width=0.05, head_length=0.08,
                    fc=colors[pos], ec=colors[pos],
                    linewidth=linewidth, alpha=alpha)

        # Styling
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_title(f'RoPE Rotation Animation\nPosition: {frame} ({angles[frame]*180/np.pi:.1f}°)')

        # Add circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.3)
        ax.add_patch(circle)

        return ax,

    # Animate
    anim = FuncAnimation(fig, update, frames=max_positions, interval=200, blit=False)

    # Save
    save_path = VIZ_ANIMATIONS_DIR / 'phase2' / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=5))
    print(f"Saved animation to: {save_path}")
    plt.close()


def plot_frequency_bands(d_k: int = 16, filename: str = "rope_frequency_bands.png"):
    """
    Plot different rotation frequencies across dimension pairs.

    Shows how different dimensions rotate at different rates.
    """
    print("Creating frequency bands visualization...")

    max_pos = 20
    freqs = precompute_rope_frequencies(max_pos, d_k, device=torch.device('cpu'))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot first 4 dimension pairs
    for dim_pair in range(min(4, d_k // 2)):
        ax = axes[dim_pair]

        angles = freqs[:, dim_pair].detach().numpy()
        positions = np.arange(max_pos)

        # Plot angle vs position
        ax.plot(positions, angles, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Position')
        ax.set_ylabel('Rotation Angle (radians)')
        ax.set_title(f'Dimension Pair {dim_pair}')
        ax.grid(alpha=0.3)

        # Add wavelength info
        if dim_pair < len(angles) - 1:
            freq = angles[1] - angles[0]  # Frequency (angle increment per position)
            wavelength = 2 * np.pi / freq if freq > 0 else float('inf')
            ax.text(0.05, 0.95, f'Wavelength: {wavelength:.1f} positions',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_figure(fig, filename, subdir='phase2')
    plt.close()


def plot_relative_position_matrix(seq_len: int = 10, filename: str = "rope_relative_position.png"):
    """
    Demonstrate that RoPE encodes relative position.

    Shows that attention scores depend on (query_pos - key_pos).
    """
    print("Creating relative position matrix...")

    d_k = 16
    device = torch.device('cpu')

    # Create random Q and K
    torch.manual_seed(42)
    Q = torch.randn(1, 1, seq_len, d_k, device=device)
    K = torch.randn(1, 1, seq_len, d_k, device=device)

    # Compute frequencies
    freqs = precompute_rope_frequencies(seq_len, d_k, device=device)

    # Apply RoPE
    Q_rotated = apply_rotary_embedding(Q, freqs)
    K_rotated = apply_rotary_embedding(K, freqs)

    # Compute attention scores
    scores = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1)) / (d_k ** 0.5)
    scores = scores[0, 0].detach().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(scores, cmap='RdBu_r', aspect='auto')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('RoPE Attention Scores\n(Depends on relative position q_pos - k_pos)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Score (before softmax)')

    # Add diagonal lines to highlight relative position patterns
    for offset in [-3, -1, 0, 1, 3]:
        if offset != 0:
            diag_positions = []
            for i in range(seq_len):
                j = i - offset
                if 0 <= j < seq_len:
                    diag_positions.append((j, i))
            if diag_positions:
                js, is_ = zip(*diag_positions)
                ax.plot(js, is_, 'gray', alpha=0.3, linewidth=1)

    save_figure(fig, filename, subdir='phase2')
    plt.close()


def plot_3d_rope_embeddings(max_pos: int = 20, filename: str = "rope_3d_projection.png"):
    """
    Project RoPE-rotated embeddings to 3D.

    Shows the geometric structure of rotated embeddings.
    """
    print("Creating 3D projection of RoPE embeddings...")

    d_k = 64
    device = torch.device('cpu')

    # Create base embedding (same for all positions)
    torch.manual_seed(42)
    base_embedding = torch.randn(1, 1, 1, d_k, device=device)

    # Replicate for all positions
    embeddings = base_embedding.repeat(1, 1, max_pos, 1)

    # Compute frequencies and apply RoPE
    freqs = precompute_rope_frequencies(max_pos, d_k, device=device)
    rotated_embeddings = apply_rotary_embedding(embeddings, freqs)

    # Extract and reshape for PCA
    rotated_embeddings = rotated_embeddings[0, 0].detach().numpy()  # (max_pos, d_k)

    # PCA to 3D
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(rotated_embeddings)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    positions = np.arange(max_pos)
    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=positions,
        cmap='viridis',
        s=100,
        alpha=0.6
    )

    # Connect with line
    ax.plot(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        'gray',
        alpha=0.3,
        linewidth=1
    )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax.set_title('RoPE: Rotated Embeddings in 3D (PCA)')

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Position')

    save_figure(fig, filename, subdir='phase2')
    plt.close()


def visualize_all():
    """Generate all RoPE visualizations."""
    print_section("Generating Phase 2 (RoPE) Visualizations")

    setup_plotting_style()

    # Check if frequencies are available
    freq_path = DATA_PROCESSED_DIR / "rope_frequencies.pt"
    if not freq_path.exists():
        print(f"WARNING: {freq_path} not found. Run rope.py first.")
        print("Generating visualizations with default parameters...\n")

    # Create visualizations
    plot_2d_rotation()
    create_rotation_animation()
    plot_frequency_bands()
    plot_relative_position_matrix()
    plot_3d_rope_embeddings()

    print("\n" + "=" * 80)
    print("All RoPE visualizations complete!")
    print(f"Check {VIZ_PLOTS_DIR / 'phase2'} for plots")
    print(f"Check {VIZ_ANIMATIONS_DIR / 'phase2'} for animations")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize RoPE Rotations")
    args = parser.parse_args()

    visualize_all()


if __name__ == "__main__":
    main()
