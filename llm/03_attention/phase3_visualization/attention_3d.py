"""
Phase 3: 3D Attention Landscapes

Creates 3D surface plots and rotating animations of attention weights.

The 3D visualization provides an intuitive geometric view:
- X-axis: Key positions
- Y-axis: Query positions
- Z-axis: Attention weight
- Height of surface = strength of attention

Rotating animations show the structure from all angles.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (DATA_PROCESSED_DIR, VIZ_ANIMATIONS_DIR,
                          print_section, save_figure, setup_plotting_style)
from shared.visualization_utils import (create_rotating_3d_animation,
                                        plot_3d_attention_surface)


def visualize_3d_surface():
    """
    Create static 3D surface plot of attention weights.

    The surface plot shows attention as a landscape:
    - Peaks = high attention
    - Valleys = low attention
    - Shape reveals attention structure
    """
    print_section("Visualizing 3D Attention Surface")

    # Load Phase 1 data (single-head attention)
    data_path = DATA_PROCESSED_DIR / "phase1_attention_weights.pt"

    if not data_path.exists():
        print(f"Error: Attention data not found at {data_path}")
        print("Please run phase1_single_head/self_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attention_weights = data['attention_weights'].squeeze(0).detach().cpu().numpy()  # (seq_len, seq_len)
    tokens = data['tokens']

    print(f"Loaded attention weights: shape {attention_weights.shape}")
    print(f"Tokens: {tokens}\n")

    # Analyze surface characteristics
    print("Surface characteristics:")
    print(f"  Min height: {attention_weights.min():.3f}")
    print(f"  Max height: {attention_weights.max():.3f}")
    print(f"  Mean height: {attention_weights.mean():.3f}")
    print(f"  Std deviation: {attention_weights.std():.3f}\n")

    # Find peaks (highest attention)
    max_idx = np.unravel_index(np.argmax(attention_weights), attention_weights.shape)
    print(f"Highest peak:")
    print(f"  Position: '{tokens[max_idx[0]]}' → '{tokens[max_idx[1]]}'")
    print(f"  Weight: {attention_weights[max_idx]:.3f}\n")

    # Setup plotting
    setup_plotting_style()

    # Create 3D surface plot
    fig, ax = plot_3d_attention_surface(
        attention_weights,
        tokens,
        title="3D Attention Landscape (Single-Head)",
        figsize=(12, 9),
        cmap='viridis'
    )

    # Save figure
    save_figure(fig, "3d_attention_surface.png", subdir="phase3", dpi=300)

    print("\n✓ 3D surface visualization complete!")
    print("  The surface shows attention as a landscape:")
    print("  - High peaks = strong attention")
    print("  - Flat regions = weak/uniform attention")
    print("  - Diagonal ridge = self-attention pattern\n")


def create_rotating_animation():
    """
    Create rotating 3D animation of attention surface.

    A 360-degree rotation shows the structure from all angles,
    making patterns easier to understand.
    """
    print_section("Creating Rotating 3D Animation")

    # Load Phase 1 data
    data_path = DATA_PROCESSED_DIR / "phase1_attention_weights.pt"

    if not data_path.exists():
        print(f"Error: Attention data not found at {data_path}")
        print("Please run phase1_single_head/self_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attention_weights = data['attention_weights'].squeeze(0).detach().cpu().numpy()
    tokens = data['tokens']

    print(f"Creating 360° rotation animation...")
    print(f"  Frames: 60")
    print(f"  FPS: 20")
    print(f"  Duration: 3 seconds\n")

    # Create animation directory if needed
    anim_dir = VIZ_ANIMATIONS_DIR / "phase3"
    anim_dir.mkdir(parents=True, exist_ok=True)

    # Create animation
    output_path = anim_dir / "3d_attention_rotation.gif"

    print(f"Rendering animation to: {output_path}")
    print("(This may take 30-60 seconds...)\n")

    create_rotating_3d_animation(
        attention_weights,
        tokens,
        output_path=str(output_path),
        title="3D Attention Landscape (Rotating)",
        n_frames=60,
        fps=20,
        figsize=(12, 9)
    )

    print("\n✓ Animation complete!")
    print(f"  Saved to: {output_path}")
    print("  Open the GIF to see the rotating landscape!\n")


def visualize_multihead_3d():
    """
    Create 3D surface plots for each head in multi-head attention.

    Shows how different heads create different attention landscapes.
    """
    print_section("Visualizing Multi-Head 3D Surfaces")

    # Load Phase 2 data
    data_path = DATA_PROCESSED_DIR / "phase2_multihead_attention.pt"

    if not data_path.exists():
        print(f"Error: Multi-head data not found at {data_path}")
        print("Please run phase2_multi_head/multi_head_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attention_weights = data['attention_weights'].squeeze(0).detach().cpu().numpy()  # (n_heads, seq_len, seq_len)
    tokens = data['tokens']
    n_heads = data['n_heads']

    print(f"Creating 3D surfaces for {n_heads} heads...\n")

    # Create figure with subplots
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 12))

    seq_len = len(tokens)
    X, Y = np.meshgrid(range(seq_len), range(seq_len))

    for head_idx in range(n_heads):
        head_attn = attention_weights[head_idx]

        # Create subplot
        ax = fig.add_subplot(2, 2, head_idx + 1, projection='3d')

        # Plot surface
        surf = ax.plot_surface(X, Y, head_attn, cmap='viridis',
                              alpha=0.8, edgecolor='none')

        # Labels
        ax.set_xlabel('Key', fontsize=10)
        ax.set_ylabel('Query', fontsize=10)
        ax.set_zlabel('Weight', fontsize=10)
        ax.set_title(f'Head {head_idx + 1}', fontsize=12, fontweight='bold')

        # Set tick labels (only for first few to avoid clutter)
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, fontsize=7, rotation=45)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(tokens, fontsize=7)

        # Set viewing angle
        ax.view_init(elev=25, azim=45)

        # Stats
        print(f"Head {head_idx + 1}:")
        print(f"  Max weight: {head_attn.max():.3f}")
        print(f"  Min weight: {head_attn.min():.3f}")
        print(f"  Mean weight: {head_attn.mean():.3f}")
        print()

    fig.suptitle('Multi-Head 3D Attention Landscapes',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save
    save_figure(fig, "multihead_3d_surfaces.png", subdir="phase3", dpi=300)

    print("\n✓ Multi-head 3D surfaces complete!")
    print("  Compare head landscapes:")
    print("  - Which heads have sharp peaks?")
    print("  - Which heads are flat/uniform?")
    print("  - Do different heads have different structures?\n")


def create_all_3d_visualizations():
    """Create all 3D attention visualizations."""
    print_section("Phase 3: 3D Attention Visualizations", char="=")

    # Create visualizations
    visualize_3d_surface()
    create_rotating_animation()
    visualize_multihead_3d()

    print_section("All 3D Visualizations Complete!", char="=")
    print("Output files:")
    print("  - visualizations/plots/03_attention/phase3/3d_attention_surface.png")
    print("  - visualizations/plots/03_attention/phase3/multihead_3d_surfaces.png")
    print("  - visualizations/animations/03_attention/phase3/3d_attention_rotation.gif")
    print()
    print("Key insights:")
    print("  1. 3D view provides intuitive geometric understanding")
    print("  2. Attention structure visible as landscape topology")
    print("  3. Rotation reveals patterns from all angles")
    print()
    print("Next steps:")
    print("  1. View the rotating GIF animation")
    print("  2. Compare landscapes across different heads")
    print("  3. Move to Phase 4: python phase4_causal/causal_attention.py")
    print()


if __name__ == "__main__":
    create_all_3d_visualizations()
