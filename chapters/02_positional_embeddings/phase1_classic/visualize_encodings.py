"""
Visualize Positional Encodings

Creates visualizations for sinusoidal and learned positional embeddings:
- Heatmaps of encoding values
- 3D projections using PCA
- Similarity matrices
- Frequency spectrum (sinusoidal only)
- Animated 3D rotations
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

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


def plot_encoding_heatmap(encodings: np.ndarray, title: str, filename: str):
    """
    Plot heatmap of positional encodings.

    Args:
        encodings: Positional encodings (seq_len, d_model)
        title: Plot title
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(encodings[:20, :], cmap='RdBu_r', aspect='auto')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Encoding Value')

    # Add grid
    ax.set_xticks(np.arange(0, encodings.shape[1], 4))
    ax.set_yticks(np.arange(0, 20))
    ax.grid(alpha=0.3)

    save_figure(fig, filename, subdir='phase1')
    plt.close()


def plot_3d_projection(encodings: np.ndarray, title: str, filename: str, max_pos: int = 20):
    """
    Project encodings to 3D using PCA and visualize.

    Args:
        encodings: Positional encodings (seq_len, d_model)
        title: Plot title
        filename: Output filename
        max_pos: Maximum position to visualize
    """
    # PCA to 3D
    pca = PCA(n_components=3)
    encodings_3d = pca.fit_transform(encodings[:max_pos])

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    positions = np.arange(max_pos)
    scatter = ax.scatter(
        encodings_3d[:, 0],
        encodings_3d[:, 1],
        encodings_3d[:, 2],
        c=positions,
        cmap='viridis',
        s=100,
        alpha=0.6
    )

    # Plot line connecting positions
    ax.plot(
        encodings_3d[:, 0],
        encodings_3d[:, 1],
        encodings_3d[:, 2],
        'gray',
        alpha=0.3,
        linewidth=1
    )

    # Labels
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Position')

    save_figure(fig, filename, subdir='phase1')
    plt.close()


def create_3d_animation(encodings: np.ndarray, title: str, filename: str, max_pos: int = 20):
    """
    Create animated 3D visualization rotating around the projection.

    Args:
        encodings: Positional encodings (seq_len, d_model)
        title: Plot title
        filename: Output filename (will be .gif)
        max_pos: Maximum position to visualize
    """
    # PCA to 3D
    pca = PCA(n_components=3)
    encodings_3d = pca.fit_transform(encodings[:max_pos])

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    positions = np.arange(max_pos)

    def update(frame):
        ax.clear()

        # Plot points
        scatter = ax.scatter(
            encodings_3d[:, 0],
            encodings_3d[:, 1],
            encodings_3d[:, 2],
            c=positions,
            cmap='viridis',
            s=100,
            alpha=0.6
        )

        # Plot line
        ax.plot(
            encodings_3d[:, 0],
            encodings_3d[:, 1],
            encodings_3d[:, 2],
            'gray',
            alpha=0.3,
            linewidth=1
        )

        # Set view angle
        ax.view_init(elev=20, azim=frame)

        # Labels
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.set_title(f'{title} (angle: {frame}°)')

        return scatter,

    # Animate
    anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

    # Save
    save_path = VIZ_ANIMATIONS_DIR / 'phase1' / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=20))
    print(f"Saved animation to: {save_path}")
    plt.close()


def plot_similarity_matrix(encodings: np.ndarray, title: str, filename: str, max_pos: int = 20):
    """
    Plot pairwise cosine similarity between position encodings.

    Args:
        encodings: Positional encodings (seq_len, d_model)
        title: Plot title
        filename: Output filename
        max_pos: Maximum position to visualize
    """
    # Compute cosine similarities
    similarities = cosine_similarity(encodings[:max_pos])

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(similarities, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity')

    # Add grid
    ax.set_xticks(np.arange(0, max_pos, 2))
    ax.set_yticks(np.arange(0, max_pos, 2))
    ax.grid(alpha=0.3)

    save_figure(fig, filename, subdir='phase1')
    plt.close()


def plot_frequency_spectrum(d_model: int, filename: str):
    """
    Plot frequency spectrum of sinusoidal encoding.

    Args:
        d_model: Model dimension
        filename: Output filename
    """
    # Compute wavelengths for each dimension
    dims = np.arange(0, d_model, 2)
    div_term = np.exp(dims * -(np.log(10000.0) / d_model))
    wavelengths = 2 * np.pi / div_term

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dims, wavelengths, marker='o', linewidth=2)
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Wavelength (positions)')
    ax.set_title('Sinusoidal Encoding Frequency Spectrum')
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    save_figure(fig, filename, subdir='phase1')
    plt.close()


def visualize_all():
    """Generate all visualizations for Phase 1."""
    print_section("Generating Phase 1 Visualizations")

    setup_plotting_style()

    # Load sinusoidal encodings
    print("Loading sinusoidal encodings...")
    sin_path = DATA_PROCESSED_DIR / "sinusoidal_encodings.pt"
    if not sin_path.exists():
        print(f"ERROR: {sin_path} not found. Run sinusoidal.py first.")
        return

    sin_data = torch.load(sin_path)
    sin_encodings = sin_data['encodings'].detach().numpy()
    d_model = sin_data['d_model']

    print(f"  Shape: {sin_encodings.shape}")
    print(f"  d_model: {d_model}\n")

    # Sinusoidal visualizations
    print("Creating sinusoidal heatmap...")
    plot_encoding_heatmap(
        sin_encodings,
        "Sinusoidal Positional Encodings",
        "sinusoidal_heatmap.png"
    )

    print("Creating sinusoidal 3D projection...")
    plot_3d_projection(
        sin_encodings,
        "Sinusoidal Encodings in 3D (PCA)",
        "sinusoidal_3d.png"
    )

    print("Creating sinusoidal 3D animation...")
    create_3d_animation(
        sin_encodings,
        "Sinusoidal Encodings in 3D (PCA)",
        "sinusoidal_3d_rotation.gif"
    )

    print("Creating sinusoidal similarity matrix...")
    plot_similarity_matrix(
        sin_encodings,
        "Sinusoidal Encoding Similarities",
        "sinusoidal_similarity.png"
    )

    print("Creating frequency spectrum...")
    plot_frequency_spectrum(d_model, "sinusoidal_frequency_spectrum.png")

    # Load learned encodings if available
    learned_path = DATA_PROCESSED_DIR / "learned_embeddings.pt"
    if learned_path.exists():
        print("\nLoading learned encodings...")
        learned_data = torch.load(learned_path)
        learned_encodings = learned_data['encodings'].detach().numpy()

        print(f"  Shape: {learned_encodings.shape}\n")

        # Learned visualizations
        print("Creating learned heatmap...")
        plot_encoding_heatmap(
            learned_encodings,
            "Learned Positional Embeddings",
            "learned_heatmap.png"
        )

        print("Creating learned 3D projection...")
        plot_3d_projection(
            learned_encodings,
            "Learned Embeddings in 3D (PCA)",
            "learned_3d.png"
        )

        print("Creating learned 3D animation...")
        create_3d_animation(
            learned_encodings,
            "Learned Embeddings in 3D (PCA)",
            "learned_3d_rotation.gif"
        )

        print("Creating learned similarity matrix...")
        plot_similarity_matrix(
            learned_encodings,
            "Learned Embedding Similarities",
            "learned_similarity.png"
        )
    else:
        print(f"\nWARNING: {learned_path} not found. Skipping learned embeddings.")
        print("Run learned.py first to train learned embeddings.")

    print("\n" + "=" * 80)
    print("All visualizations complete!")
    print(f"Check {VIZ_PLOTS_DIR / 'phase1'} for plots")
    print(f"Check {VIZ_ANIMATIONS_DIR / 'phase1'} for animations")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize Positional Encodings")
    args = parser.parse_args()

    visualize_all()


if __name__ == "__main__":
    main()
