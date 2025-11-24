"""
Phase 1: Compare Untrained vs Trained Attention Patterns

Visualizes how attention patterns evolve from random (untrained) to 
focused and meaningful (trained) after learning next-token prediction.

Key Insights to Demonstrate:
1. Untrained attention is diffuse and uniform
2. Trained attention becomes focused and structured
3. Attention entropy decreases (more focused)
4. Specific linguistic/sequential patterns emerge
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
    """Load both untrained and trained attention weights."""
    untrained_path = DATA_PROCESSED_DIR / "phase1_attention_weights.pt"
    trained_path = DATA_PROCESSED_DIR / "phase1_trained_attention.pt"
    
    if not untrained_path.exists():
        print(f"Error: Untrained attention not found at {untrained_path}")
        print("Please run: uv run python phase1_single_head/self_attention.py")
        return None, None
    
    if not trained_path.exists():
        print(f"Error: Trained attention not found at {trained_path}")
        print("Please run: uv run python phase1_single_head/train.py")
        return None, None
    
    untrained_data = torch.load(untrained_path)
    trained_data = torch.load(trained_path)
    
    return untrained_data, trained_data


def compute_attention_metrics(attention_weights):
    """
    Compute metrics to quantify attention behavior.
    
    Returns:
        entropy_per_query: Entropy of each query's attention distribution
        sparsity: Percentage of attention weight in top-1 token
        focus_score: Average of max attention weight per query
    """
    # attention_weights: (seq_len, seq_len)
    seq_len = attention_weights.shape[0]
    
    # Entropy for each query (how spread out is the attention?)
    entropy_per_query = []
    for i in range(seq_len):
        dist = attention_weights[i]
        # Add small epsilon to avoid log(0)
        ent = entropy(dist + 1e-10)
        entropy_per_query.append(ent)
    
    # Sparsity: average percentage of weight in top-1 token
    max_weights = np.max(attention_weights, axis=1)
    sparsity = np.mean(max_weights)
    
    # Focus score: average max attention
    focus_score = np.mean(max_weights)
    
    return {
        'entropy_per_query': np.array(entropy_per_query),
        'mean_entropy': np.mean(entropy_per_query),
        'sparsity': sparsity,
        'focus_score': focus_score
    }


def plot_side_by_side_heatmaps(untrained_attn, trained_attn, tokens, output_path):
    """Plot untrained and trained attention heatmaps side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Untrained heatmap
    sns.heatmap(
        untrained_attn,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=tokens,
        yticklabels=tokens,
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=1,
        ax=ax1,
        square=True
    )
    ax1.set_title('Untrained Attention\n(Random, Diffuse)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Key Position', fontsize=12)
    ax1.set_ylabel('Query Position', fontsize=12)
    
    # Trained heatmap
    sns.heatmap(
        trained_attn,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=tokens,
        yticklabels=tokens,
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=1,
        ax=ax2,
        square=True
    )
    ax2.set_title('Trained Attention\n(Focused, Structured)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Key Position', fontsize=12)
    ax2.set_ylabel('Query Position', fontsize=12)
    
    plt.suptitle('Attention Evolution: Untrained → Trained', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison heatmap to: {output_path}")
    plt.close()


def plot_entropy_comparison(untrained_metrics, trained_metrics, tokens, output_path):
    """Plot entropy comparison between untrained and trained."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart: entropy per query
    x = np.arange(len(tokens))
    width = 0.35
    
    ax1.bar(x - width/2, untrained_metrics['entropy_per_query'], width, 
            label='Untrained', color='lightcoral', alpha=0.8)
    ax1.bar(x + width/2, trained_metrics['entropy_per_query'], width, 
            label='Trained', color='steelblue', alpha=0.8)
    
    ax1.set_xlabel('Query Position', fontsize=12)
    ax1.set_ylabel('Entropy (nats)', fontsize=12)
    ax1.set_title('Attention Entropy per Query', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Summary metrics
    metrics_comparison = {
        'Mean\nEntropy': [untrained_metrics['mean_entropy'], trained_metrics['mean_entropy']],
        'Focus\nScore': [untrained_metrics['focus_score'], trained_metrics['focus_score']],
        'Sparsity': [untrained_metrics['sparsity'], trained_metrics['sparsity']]
    }
    
    x_pos = np.arange(len(metrics_comparison))
    untrained_vals = [v[0] for v in metrics_comparison.values()]
    trained_vals = [v[1] for v in metrics_comparison.values()]
    
    ax2.bar(x_pos - width/2, untrained_vals, width, 
            label='Untrained', color='lightcoral', alpha=0.8)
    ax2.bar(x_pos + width/2, trained_vals, width, 
            label='Trained', color='steelblue', alpha=0.8)
    
    ax2.set_xlabel('Metric', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Summary Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics_comparison.keys())
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Attention Becomes More Focused After Training', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved entropy comparison to: {output_path}")
    plt.close()


def plot_attention_distribution(untrained_attn, trained_attn, tokens, output_path):
    """Plot attention weight distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    seq_len = len(tokens)
    for i in range(min(6, seq_len)):
        ax = axes[i]
        
        x = np.arange(seq_len)
        width = 0.35
        
        ax.bar(x - width/2, untrained_attn[i], width, 
               label='Untrained', color='lightcoral', alpha=0.8)
        ax.bar(x + width/2, trained_attn[i], width, 
               label='Trained', color='steelblue', alpha=0.8)
        
        ax.set_title(f'Query: "{tokens[i]}" (pos {i})', fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Attention Weight')
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(min(6, seq_len), 6):
        axes[i].axis('off')
    
    plt.suptitle('Attention Weight Distributions: Untrained vs Trained', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution comparison to: {output_path}")
    plt.close()


def print_attention_analysis(untrained_attn, trained_attn, tokens, 
                             untrained_metrics, trained_metrics):
    """Print detailed analysis of attention changes."""
    print_section("Attention Pattern Analysis")
    
    print("UNTRAINED ATTENTION:")
    print(f"  Mean entropy: {untrained_metrics['mean_entropy']:.3f} nats")
    print(f"  Focus score (avg max weight): {untrained_metrics['focus_score']:.3f}")
    print(f"  Sparsity (top-1 weight): {untrained_metrics['sparsity']:.3f}")
    print(f"  → Interpretation: Diffuse, near-uniform attention\n")
    
    print("TRAINED ATTENTION:")
    print(f"  Mean entropy: {trained_metrics['mean_entropy']:.3f} nats")
    print(f"  Focus score (avg max weight): {trained_metrics['focus_score']:.3f}")
    print(f"  Sparsity (top-1 weight): {trained_metrics['sparsity']:.3f}")
    print(f"  → Interpretation: More focused, structured attention\n")
    
    print("CHANGES:")
    entropy_change = ((trained_metrics['mean_entropy'] - untrained_metrics['mean_entropy']) 
                      / untrained_metrics['mean_entropy'] * 100)
    focus_change = ((trained_metrics['focus_score'] - untrained_metrics['focus_score']) 
                    / untrained_metrics['focus_score'] * 100)
    
    print(f"  Entropy change: {entropy_change:+.1f}%")
    print(f"  Focus change: {focus_change:+.1f}%")
    
    if entropy_change < 0:
        print(f"  ✓ Entropy decreased → attention became more focused")
    if focus_change > 0:
        print(f"  ✓ Focus increased → stronger attention to key tokens")
    
    print("\nPER-QUERY ANALYSIS:")
    for i, token in enumerate(tokens):
        untrained_max = np.max(untrained_attn[i])
        trained_max = np.max(trained_attn[i])
        untrained_max_pos = np.argmax(untrained_attn[i])
        trained_max_pos = np.argmax(trained_attn[i])
        
        print(f"  Query '{token}' (pos {i}):")
        print(f"    Untrained: max weight {untrained_max:.3f} at '{tokens[untrained_max_pos]}'")
        print(f"    Trained:   max weight {trained_max:.3f} at '{tokens[trained_max_pos]}'")
        
        if trained_max > untrained_max * 1.2:  # 20% increase
            print(f"    → Became more focused!")


def create_all_visualizations():
    """Create all comparison visualizations."""
    print_section("Comparing Untrained vs Trained Attention")
    
    # Load data
    untrained_data, trained_data = load_attention_data()
    if untrained_data is None or trained_data is None:
        return
    
    # Extract attention weights
    untrained_attn = untrained_data['attention_weights'].squeeze(0).detach().cpu().numpy()
    trained_attn = trained_data['attention_weights'].squeeze(0).detach().cpu().numpy()
    tokens = untrained_data['tokens']
    
    print(f"Loaded attention weights:")
    print(f"  Untrained shape: {untrained_attn.shape}")
    print(f"  Trained shape: {trained_attn.shape}")
    print(f"  Tokens: {tokens}\n")
    
    # Compute metrics
    untrained_metrics = compute_attention_metrics(untrained_attn)
    trained_metrics = compute_attention_metrics(trained_attn)
    
    # Print analysis
    print_attention_analysis(untrained_attn, trained_attn, tokens, 
                            untrained_metrics, trained_metrics)
    
    # Setup plotting
    setup_plotting_style()
    output_dir = VIZ_PLOTS_DIR / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print_section("Generating Comparison Visualizations")
    
    plot_side_by_side_heatmaps(
        untrained_attn, trained_attn, tokens,
        output_dir / "attention_comparison.png"
    )
    
    plot_entropy_comparison(
        untrained_metrics, trained_metrics, tokens,
        output_dir / "entropy_comparison.png"
    )
    
    plot_attention_distribution(
        untrained_attn, trained_attn, tokens,
        output_dir / "distribution_comparison.png"
    )
    
    print_section("Comparison Complete")
    print(f"All visualizations saved to: {output_dir}")
    print("\nKey Takeaways:")
    print("  1. Untrained attention is diffuse and near-uniform")
    print("  2. Trained attention becomes focused on specific tokens")
    print("  3. Entropy decreases (more structured)")
    print("  4. Focus score increases (stronger max attention)")
    print("  5. Patterns emerge that reflect learned structure")


if __name__ == "__main__":
    create_all_visualizations()

