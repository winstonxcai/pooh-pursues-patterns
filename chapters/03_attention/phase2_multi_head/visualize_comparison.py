"""
Phase 2: Compare Untrained vs Trained Multi-Head Attention

Visualizes how each attention head evolves from random (untrained) to
specialized (trained) patterns. Shows head diversity and specialization.

Key Insights:
1. Untrained: All heads are diffuse and random
2. Trained: Each head develops unique, focused patterns
3. Head specialization: Different heads learn different relationships
4. Diversity increases: Heads become less correlated
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import entropy

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (DATA_PROCESSED_DIR, VIZ_PLOTS_DIR, print_section,
                          setup_plotting_style)


def load_attention_data():
    """Load both untrained and trained multi-head attention weights."""
    untrained_path = DATA_PROCESSED_DIR / "phase2_multihead_attention.pt"
    trained_path = DATA_PROCESSED_DIR / "phase2_trained_attention.pt"
    
    if not untrained_path.exists():
        print(f"Error: Untrained attention not found at {untrained_path}")
        print("Please run: uv run python phase2_multi_head/multi_head_attention.py")
        return None, None
    
    if not trained_path.exists():
        print(f"Error: Trained attention not found at {trained_path}")
        print("Please run: uv run python phase2_multi_head/train.py")
        return None, None
    
    untrained_data = torch.load(untrained_path)
    trained_data = torch.load(trained_path)
    
    return untrained_data, trained_data


def compute_head_metrics(attention_weights):
    """
    Compute metrics for each attention head.
    
    Args:
        attention_weights: (n_heads, seq_len, seq_len)
        
    Returns:
        dict with metrics per head
    """
    n_heads, seq_len, _ = attention_weights.shape
    
    metrics = []
    for head_idx in range(n_heads):
        head_attn = attention_weights[head_idx]
        
        # Entropy per query
        entropies = []
        for i in range(seq_len):
            dist = head_attn[i]
            ent = entropy(dist + 1e-10)
            entropies.append(ent)
        
        # Focus score (average max weight)
        max_weights = np.max(head_attn, axis=1)
        focus_score = np.mean(max_weights)
        
        metrics.append({
            'entropy': np.array(entropies),
            'mean_entropy': np.mean(entropies),
            'focus_score': focus_score,
            'max_attention': np.max(head_attn),
            'min_attention': np.min(head_attn)
        })
    
    return metrics


def plot_multihead_comparison(untrained_attn, trained_attn, tokens, n_heads, output_path):
    """Plot all heads side-by-side: untrained vs trained."""
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
        ax_untrained.set_title(f'Head {head_idx + 1}\n(Untrained)', 
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
        ax_trained.set_title(f'Head {head_idx + 1}\n(Trained)', 
                            fontsize=12, fontweight='bold')
        ax_trained.set_xlabel('Key Position', fontsize=11)
        if head_idx == 0:
            ax_trained.set_ylabel('Query Position', fontsize=11)
    
    plt.suptitle('Multi-Head Attention: Untrained → Trained\nEach Head Learns Different Patterns', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved multi-head comparison to: {output_path}")
    plt.close()


def plot_entropy_by_head(untrained_metrics, trained_metrics, n_heads, output_path):
    """Plot entropy comparison for each head."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean entropy per head
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
    ax1.set_title('Attention Entropy by Head', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Head {i+1}' for i in range(n_heads)])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Focus score per head
    untrained_focus = [m['focus_score'] for m in untrained_metrics]
    trained_focus = [m['focus_score'] for m in trained_metrics]
    
    ax2.bar(x - width/2, untrained_focus, width, 
            label='Untrained', color='lightcoral', alpha=0.8)
    ax2.bar(x + width/2, trained_focus, width, 
            label='Trained', color='steelblue', alpha=0.8)
    
    ax2.set_xlabel('Head', fontsize=12)
    ax2.set_ylabel('Focus Score (max attention)', fontsize=12)
    ax2.set_title('Attention Focus by Head', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Head {i+1}' for i in range(n_heads)])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Head-Specific Metrics: Training Increases Focus', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved entropy comparison to: {output_path}")
    plt.close()


def plot_head_diversity(untrained_attn, trained_attn, n_heads, output_path):
    """Plot head diversity (correlation between heads)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Compute correlation matrices
    def compute_correlation_matrix(attention_weights):
        n_heads = attention_weights.shape[0]
        corr_matrix = np.zeros((n_heads, n_heads))
        for i in range(n_heads):
            for j in range(n_heads):
                attn_i = attention_weights[i].flatten()
                attn_j = attention_weights[j].flatten()
                corr_matrix[i, j] = np.corrcoef(attn_i, attn_j)[0, 1]
        return corr_matrix
    
    untrained_corr = compute_correlation_matrix(untrained_attn)
    trained_corr = compute_correlation_matrix(trained_attn)
    
    # Untrained correlation
    sns.heatmap(untrained_corr, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, center=0, square=True, ax=ax1,
                xticklabels=[f'H{i+1}' for i in range(n_heads)],
                yticklabels=[f'H{i+1}' for i in range(n_heads)])
    ax1.set_title('Untrained Head Correlation\n(High = Similar Patterns)', 
                  fontsize=12, fontweight='bold')
    
    # Trained correlation
    sns.heatmap(trained_corr, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, center=0, square=True, ax=ax2,
                xticklabels=[f'H{i+1}' for i in range(n_heads)],
                yticklabels=[f'H{i+1}' for i in range(n_heads)])
    ax2.set_title('Trained Head Correlation\n(Low = Diverse Patterns)', 
                  fontsize=12, fontweight='bold')
    
    # Compute average off-diagonal correlation
    def avg_off_diagonal(corr_matrix):
        n = corr_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return np.mean(np.abs(corr_matrix[mask]))
    
    untrained_diversity = 1 - avg_off_diagonal(untrained_corr)
    trained_diversity = 1 - avg_off_diagonal(trained_corr)
    
    plt.suptitle(f'Head Diversity: Training Increases Specialization\n'
                 f'Diversity Score - Untrained: {untrained_diversity:.2f}, '
                 f'Trained: {trained_diversity:.2f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved head diversity to: {output_path}")
    plt.close()


def print_detailed_analysis(untrained_attn, trained_attn, tokens, 
                           untrained_metrics, trained_metrics, n_heads):
    """Print detailed analysis of changes."""
    print_section("Multi-Head Attention Analysis")
    
    for head_idx in range(n_heads):
        print(f"\nHEAD {head_idx + 1}:")
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
        
        # Show strongest attention patterns
        print(f"  Strongest trained attention:")
        head_attn = trained_attn[head_idx]
        for i, query_token in enumerate(tokens):
            max_idx = head_attn[i].argmax()
            max_weight = head_attn[i, max_idx]
            if max_weight > 0.3:  # Only show strong connections
                key_token = tokens[max_idx]
                marker = " [SELF]" if max_idx == i else ""
                print(f"    '{query_token}' → '{key_token}': {max_weight:.3f}{marker}")


def create_all_visualizations():
    """Create all comparison visualizations."""
    print_section("Comparing Untrained vs Trained Multi-Head Attention")
    
    # Load data
    untrained_data, trained_data = load_attention_data()
    if untrained_data is None or trained_data is None:
        return
    
    # Extract attention weights
    untrained_attn = untrained_data['attention_weights'].squeeze(0).detach().cpu().numpy()
    trained_attn = trained_data['attention_weights'].squeeze(0).detach().cpu().numpy()
    tokens = untrained_data['tokens']
    n_heads = untrained_data['n_heads']
    
    print(f"Loaded attention weights:")
    print(f"  Untrained shape: {untrained_attn.shape}")
    print(f"  Trained shape: {trained_attn.shape}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Tokens: {tokens}\n")
    
    # Compute metrics
    untrained_metrics = compute_head_metrics(untrained_attn)
    trained_metrics = compute_head_metrics(trained_attn)
    
    # Print analysis
    print_detailed_analysis(untrained_attn, trained_attn, tokens,
                           untrained_metrics, trained_metrics, n_heads)
    
    # Setup plotting
    setup_plotting_style()
    output_dir = VIZ_PLOTS_DIR / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print_section("Generating Comparison Visualizations")
    
    plot_multihead_comparison(
        untrained_attn, trained_attn, tokens, n_heads,
        output_dir / "multihead_comparison.png"
    )
    
    plot_entropy_by_head(
        untrained_metrics, trained_metrics, n_heads,
        output_dir / "entropy_by_head.png"
    )
    
    plot_head_diversity(
        untrained_attn, trained_attn, n_heads,
        output_dir / "head_diversity.png"
    )
    
    print_section("Comparison Complete")
    print(f"All visualizations saved to: {output_dir}")
    print("\nKey Takeaways:")
    print("  1. Each head develops unique attention patterns")
    print("  2. Head diversity increases (less correlation)")
    print("  3. Some heads become very focused, others remain diffuse")
    print("  4. Different heads specialize in different relationships")
    print("  5. Multi-head attention enables parallel pattern learning")


if __name__ == "__main__":
    create_all_visualizations()

