"""
Phase 4: Compare Untrained vs Trained Causal Attention

Visualizes how causal attention patterns evolve from random (untrained) to
specialized (trained) under the constraint of only attending to past positions.

Key comparison with Phase 2 (bidirectional):
- Both have same architecture and data
- Phase 2: Full attention (can see all positions)
- Phase 4: Causal attention (can only see past)

Shows how causality constraint shapes head specialization differently.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    print_section,
    setup_plotting_style,
    DATA_PROCESSED_DIR,
    VIZ_PLOTS_DIR
)


def load_causal_attention_data():
    """Load both untrained and trained causal attention weights."""
    # Untrained causal attention
    untrained_path = DATA_PROCESSED_DIR / "phase4_causal_comparison.pt"
    # Trained causal attention
    trained_path = DATA_PROCESSED_DIR / "phase4_trained_causal_attention.pt"
    
    if not untrained_path.exists():
        print(f"Error: Untrained causal attention not found at {untrained_path}")
        print("Please run: uv run python phase4_causal/causal_attention.py")
        return None, None
    
    if not trained_path.exists():
        print(f"Error: Trained causal attention not found at {trained_path}")
        print("Please run: uv run python phase4_causal/train_causal.py")
        return None, None
    
    untrained_data = torch.load(untrained_path)
    trained_data = torch.load(trained_path)
    
    return untrained_data, trained_data


def compute_causal_head_metrics(attention_weights):
    """
    Compute metrics for each causal attention head.
    Only considers lower triangle (past + self) since upper triangle is masked.
    """
    n_heads, seq_len, _ = attention_weights.shape
    
    metrics = []
    for head_idx in range(n_heads):
        head_attn = attention_weights[head_idx]
        
        # Entropy per query (only over valid past positions)
        entropies = []
        for i in range(seq_len):
            # Only consider positions 0 to i (past + self)
            valid_dist = head_attn[i, :i+1]
            if len(valid_dist) > 1:
                ent = entropy(valid_dist + 1e-10)
            else:
                ent = 0.0  # First position can only attend to itself
            entropies.append(ent)
        
        # Focus score (average max weight over valid positions)
        max_weights = []
        for i in range(seq_len):
            max_weights.append(np.max(head_attn[i, :i+1]))
        focus_score = np.mean(max_weights)
        
        metrics.append({
            'entropy': np.array(entropies),
            'mean_entropy': np.mean(entropies),
            'focus_score': focus_score,
            'max_attention': np.max(head_attn),
            'min_attention': np.min(head_attn[head_attn > 0])  # Min non-zero
        })
    
    return metrics


def plot_causal_multihead_comparison(untrained_attn, trained_attn, tokens, n_heads, output_path):
    """Plot all causal heads side-by-side: untrained vs trained."""
    fig, axes = plt.subplots(2, n_heads, figsize=(5*n_heads, 10))
    
    for head_idx in range(n_heads):
        # Untrained (top row)
        ax_untrained = axes[0, head_idx]
        sns.heatmap(
            untrained_attn[head_idx],
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=tokens,
            yticklabels=tokens,
            vmin=0,
            vmax=1,
            ax=ax_untrained,
            square=True,
            cbar=head_idx == n_heads - 1
        )
        ax_untrained.set_title(f'Head {head_idx + 1}\n(Untrained Causal)', 
                               fontsize=12, fontweight='bold')
        if head_idx == 0:
            ax_untrained.set_ylabel('Query Position', fontsize=11)
        
        # Trained (bottom row)
        ax_trained = axes[1, head_idx]
        sns.heatmap(
            trained_attn[head_idx],
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=tokens,
            yticklabels=tokens,
            vmin=0,
            vmax=1,
            ax=ax_trained,
            square=True,
            cbar=head_idx == n_heads - 1
        )
        ax_trained.set_title(f'Head {head_idx + 1}\n(Trained Causal)', 
                            fontsize=12, fontweight='bold')
        ax_trained.set_xlabel('Key Position', fontsize=11)
        if head_idx == 0:
            ax_trained.set_ylabel('Query Position', fontsize=11)
    
    plt.suptitle('Causal Attention Evolution: Untrained → Trained\n'
                 '(Upper triangle = 0, only past attention allowed)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved causal comparison to: {output_path}")
    plt.close()


def plot_causal_entropy_comparison(untrained_metrics, trained_metrics, n_heads, output_path):
    """Plot entropy and focus comparison for causal attention."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(n_heads)
    width = 0.35
    
    untrained_entropies = [m['mean_entropy'] for m in untrained_metrics]
    trained_entropies = [m['mean_entropy'] for m in trained_metrics]
    
    ax1.bar(x - width/2, untrained_entropies, width, 
            label='Untrained', color='lightcoral', alpha=0.8)
    ax1.bar(x + width/2, trained_entropies, width, 
            label='Trained', color='steelblue', alpha=0.8)
    
    ax1.set_xlabel('Head', fontsize=12)
    ax1.set_ylabel('Mean Entropy (nats)', fontsize=12)
    ax1.set_title('Causal Attention Entropy by Head', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Head {i+1}' for i in range(n_heads)])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Focus score
    untrained_focus = [m['focus_score'] for m in untrained_metrics]
    trained_focus = [m['focus_score'] for m in trained_metrics]
    
    ax2.bar(x - width/2, untrained_focus, width, 
            label='Untrained', color='lightcoral', alpha=0.8)
    ax2.bar(x + width/2, trained_focus, width, 
            label='Trained', color='steelblue', alpha=0.8)
    
    ax2.set_xlabel('Head', fontsize=12)
    ax2.set_ylabel('Focus Score', fontsize=12)
    ax2.set_title('Causal Attention Focus by Head', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Head {i+1}' for i in range(n_heads)])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Causal Attention: Training Increases Focus (Past-Only Constraint)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved causal entropy comparison to: {output_path}")
    plt.close()


def plot_bidirectional_vs_causal_trained(output_path):
    """
    Compare trained bidirectional (Phase 2) vs trained causal (Phase 4) attention.
    Shows impact of causal constraint on learned patterns.
    """
    # Load Phase 2 trained (bidirectional)
    bidir_path = DATA_PROCESSED_DIR / "phase2_trained_attention.pt"
    # Load Phase 4 trained (causal)
    causal_path = DATA_PROCESSED_DIR / "phase4_trained_causal_attention.pt"
    
    if not bidir_path.exists() or not causal_path.exists():
        print("Skipping bidirectional vs causal comparison (missing data)")
        return
    
    bidir_data = torch.load(bidir_path)
    causal_data = torch.load(causal_path)
    
    bidir_attn = bidir_data['attention_weights'].squeeze(0).detach().cpu().numpy()
    causal_attn = causal_data['attention_weights'].squeeze(0).detach().cpu().numpy()
    tokens = bidir_data['tokens']
    n_heads = bidir_data['n_heads']
    
    fig, axes = plt.subplots(2, n_heads, figsize=(5*n_heads, 10))
    
    for head_idx in range(n_heads):
        # Bidirectional (top row)
        ax_bidir = axes[0, head_idx]
        sns.heatmap(
            bidir_attn[head_idx],
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=tokens,
            yticklabels=tokens,
            vmin=0,
            vmax=1,
            ax=ax_bidir,
            square=True,
            cbar=head_idx == n_heads - 1
        )
        ax_bidir.set_title(f'Head {head_idx + 1}\n(Bidirectional)', 
                          fontsize=12, fontweight='bold')
        if head_idx == 0:
            ax_bidir.set_ylabel('Query Position', fontsize=11)
        
        # Causal (bottom row)
        ax_causal = axes[1, head_idx]
        sns.heatmap(
            causal_attn[head_idx],
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=tokens,
            yticklabels=tokens,
            vmin=0,
            vmax=1,
            ax=ax_causal,
            square=True,
            cbar=head_idx == n_heads - 1
        )
        ax_causal.set_title(f'Head {head_idx + 1}\n(Causal)', 
                           fontsize=12, fontweight='bold')
        ax_causal.set_xlabel('Key Position', fontsize=11)
        if head_idx == 0:
            ax_causal.set_ylabel('Query Position', fontsize=11)
    
    plt.suptitle('Impact of Causal Masking on Learned Patterns\n'
                 'Bidirectional (can see all) vs Causal (past only)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved bidirectional vs causal comparison to: {output_path}")
    plt.close()


def print_detailed_causal_analysis(untrained_attn, trained_attn, tokens,
                                   untrained_metrics, trained_metrics, n_heads):
    """Print detailed analysis of causal attention changes."""
    print_section("Causal Multi-Head Attention Analysis")
    
    for head_idx in range(n_heads):
        print(f"\nHEAD {head_idx + 1} (CAUSAL):")
        print("-" * 70)
        
        u_metrics = untrained_metrics[head_idx]
        t_metrics = trained_metrics[head_idx]
        
        print(f"  Untrained:")
        print(f"    Mean entropy: {u_metrics['mean_entropy']:.3f}")
        print(f"    Focus score: {u_metrics['focus_score']:.3f}")
        
        print(f"  Trained:")
        print(f"    Mean entropy: {t_metrics['mean_entropy']:.3f}")
        print(f"    Focus score: {t_metrics['focus_score']:.3f}")
        
        entropy_change = ((t_metrics['mean_entropy'] - u_metrics['mean_entropy']) 
                         / u_metrics['mean_entropy'] * 100)
        focus_change = ((t_metrics['focus_score'] - u_metrics['focus_score']) 
                       / u_metrics['focus_score'] * 100)
        
        print(f"  Changes:")
        print(f"    Entropy: {entropy_change:+.1f}%")
        print(f"    Focus: {focus_change:+.1f}%")
        
        # Show strongest causal attention patterns
        print(f"  Strongest trained patterns (past-only):")
        head_attn = trained_attn[head_idx]
        for i, query_token in enumerate(tokens):
            # Only look at past positions
            past_attn = head_attn[i, :i+1]
            max_idx = past_attn.argmax()
            max_weight = past_attn[max_idx]
            if max_weight > 0.3:  # Only show strong connections
                key_token = tokens[max_idx]
                marker = " [SELF]" if max_idx == i else " [PAST]"
                print(f"    '{query_token}' → '{key_token}': {max_weight:.3f}{marker}")


def create_all_visualizations():
    """Create all causal attention comparison visualizations."""
    print_section("Comparing Untrained vs Trained Causal Attention")
    
    # Load data
    untrained_data, trained_data = load_causal_attention_data()
    if untrained_data is None or trained_data is None:
        return
    
    # Extract causal attention weights
    untrained_attn = untrained_data['attention_causal'].squeeze(0).detach().cpu().numpy()
    trained_attn = trained_data['attention_weights'].squeeze(0).detach().cpu().numpy()
    tokens = untrained_data['tokens']
    n_heads = untrained_data['tokens']  # Should be in data
    
    # Handle n_heads extraction
    if isinstance(n_heads, list):
        n_heads = untrained_attn.shape[0]
    
    print(f"Loaded causal attention weights:")
    print(f"  Untrained shape: {untrained_attn.shape}")
    print(f"  Trained shape: {trained_attn.shape}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Tokens: {tokens}\n")
    
    # Verify causal property
    print("Verifying causal property (upper triangle should be 0):")
    for head_idx in range(n_heads):
        untrained_upper = np.triu(untrained_attn[head_idx], k=1).sum()
        trained_upper = np.triu(trained_attn[head_idx], k=1).sum()
        print(f"  Head {head_idx+1}: Untrained={untrained_upper:.6f}, Trained={trained_upper:.6f}")
    print()
    
    # Compute metrics
    untrained_metrics = compute_causal_head_metrics(untrained_attn)
    trained_metrics = compute_causal_head_metrics(trained_attn)
    
    # Print analysis
    print_detailed_causal_analysis(untrained_attn, trained_attn, tokens,
                                   untrained_metrics, trained_metrics, n_heads)
    
    # Setup plotting
    setup_plotting_style()
    output_dir = VIZ_PLOTS_DIR / "phase4"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print_section("Generating Causal Comparison Visualizations")
    
    plot_causal_multihead_comparison(
        untrained_attn, trained_attn, tokens, n_heads,
        output_dir / "causal_multihead_comparison.png"
    )
    
    plot_causal_entropy_comparison(
        untrained_metrics, trained_metrics, n_heads,
        output_dir / "causal_entropy_comparison.png"
    )
    
    plot_bidirectional_vs_causal_trained(
        output_dir / "bidirectional_vs_causal_trained.png"
    )
    
    print_section("Causal Comparison Complete")
    print(f"All visualizations saved to: {output_dir}")
    print("\nKey Takeaways:")
    print("  1. Causal constraint forces all heads to learn from past only")
    print("  2. Upper triangle always zero (no future information)")
    print("  3. Heads still specialize, but patterns differ from bidirectional")
    print("  4. Compare with Phase 2 to see impact of causality constraint")
    print("  5. Causal attention enables autoregressive generation (GPT-style)")


if __name__ == "__main__":
    create_all_visualizations()

