"""
Phase 1 Visualizations: Single-Head Self-Attention

Creates visualizations to understand single-head attention:
1. Attention weight heatmap - shows which tokens attend to which
2. QKV flow diagram - visualizes the complete attention mechanism

These visualizations make the abstract math concrete and interpretable.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    get_device,
    set_seed,
    load_toy_sequence,
    tokens_to_ids,
    setup_plotting_style,
    save_figure,
    print_section,
    DATA_PROCESSED_DIR
)
from shared.visualization_utils import (
    plot_attention_heatmap,
    plot_qkv_flow
)
from shared.attention_utils import scaled_dot_product_attention


def visualize_attention_heatmap():
    """
    Create attention weight heatmap visualization.

    The heatmap shows the attention weight matrix:
    - Rows = query positions (which token is attending)
    - Columns = key positions (which token is being attended to)
    - Value = attention weight (0 to 1, brighter = stronger)
    """
    print_section("Visualizing Attention Heatmap")

    # Load saved attention weights
    data_path = DATA_PROCESSED_DIR / "phase1_attention_weights.pt"

    if not data_path.exists():
        print(f"Error: Attention weights not found at {data_path}")
        print("Please run self_attention.py first!")
        return

    # Load data
    data = torch.load(data_path)
    attention_weights = data['attention_weights'].squeeze(0).detach().cpu().numpy()  # (seq_len, seq_len)
    tokens = data['tokens']

    print(f"Loaded attention weights: shape {attention_weights.shape}")
    print(f"Tokens: {tokens}\n")

    # Setup plotting
    setup_plotting_style()

    # Create heatmap
    fig, ax = plot_attention_heatmap(
        attention_weights,
        tokens,
        title="Single-Head Self-Attention Weights",
        figsize=(9, 8)
    )

    # Save figure
    save_figure(fig, "attention_heatmap.png", subdir="phase1")

    print("\n✓ Attention heatmap visualization complete!")
    print("  Key observations to look for:")
    print("  - Diagonal: How much does each token attend to itself?")
    print("  - Strong weights: Which token pairs have high attention?")
    print("  - Patterns: Are there linguistic relationships visible?\n")


def visualize_qkv_flow():
    """
    Create Query-Key-Value flow diagram for a single query token.

    This visualization breaks down the attention mechanism step-by-step:
    1. Query vector for selected token
    2. Attention scores (Q·K) with all keys
    3. Attention weights (after softmax)
    4. Value vectors from all tokens
    5. Weighted values (attention × values)
    6. Final output (sum of weighted values)
    """
    print_section("Visualizing QKV Flow")

    set_seed(42)
    device = get_device()

    # Configuration
    d_model = 64
    d_k = 64
    vocab_size = 20

    # Load toy sequence
    tokens, vocab = load_toy_sequence()
    token_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)
    seq_len = len(tokens)

    # Select query token to visualize (let's use "sat" at position 2)
    query_idx = 2
    query_token = tokens[query_idx]
    print(f"Visualizing attention mechanism for query token: '{query_token}' (position {query_idx})\n")

    # Create embeddings
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    x = embedding(token_ids)  # (1, seq_len, d_model)

    # Create projections (same as in SelfAttention)
    W_q = nn.Linear(d_model, d_k, bias=False).to(device)
    W_k = nn.Linear(d_model, d_k, bias=False).to(device)
    W_v = nn.Linear(d_model, d_k, bias=False).to(device)

    # Project to Q, K, V
    Q = W_q(x).squeeze(0)  # (seq_len, d_k)
    K = W_k(x).squeeze(0)  # (seq_len, d_k)
    V = W_v(x).squeeze(0)  # (seq_len, d_k)

    # Get vectors for our query token
    q_vec = Q[query_idx]  # (d_k,)

    # Compute attention scores for this query
    scores = torch.matmul(q_vec, K.T)  # (seq_len,)
    scores_scaled = scores / np.sqrt(d_k)

    # Apply softmax to get attention weights
    attention_weights = torch.softmax(scores_scaled, dim=-1)  # (seq_len,)

    # Compute output as weighted sum of values
    output = torch.matmul(attention_weights.unsqueeze(0), V).squeeze(0)  # (d_k,)

    # Convert to numpy for plotting
    Q_np = Q.detach().cpu().numpy()
    K_np = K.detach().cpu().numpy()
    V_np = V.detach().cpu().numpy()
    scores_np = scores_scaled.detach().cpu().numpy()
    attention_np = attention_weights.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()

    print(f"Query vector shape: {q_vec.shape}")
    print(f"Attention scores shape: {scores.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Output vector shape: {output.shape}\n")

    # Show top-3 attended tokens
    top_indices = np.argsort(attention_np)[::-1][:3]
    print("Top-3 attended tokens:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. '{tokens[idx]}' (position {idx}): weight = {attention_np[idx]:.3f}")
    print()

    # Setup plotting
    setup_plotting_style()

    # Create QKV flow diagram
    fig, axes = plot_qkv_flow(
        tokens=tokens,
        query_idx=query_idx,
        Q=Q_np,
        K=K_np,
        V=V_np,
        scores=scores_np,
        attention_weights=attention_np,
        output=output_np,
        figsize=(16, 10)
    )

    # Save figure
    save_figure(fig, "qkv_flow.png", subdir="phase1")

    print("\n✓ QKV flow diagram complete!")
    print("  This diagram shows the complete attention mechanism:")
    print("  1. Query vector (what token '{query_token}' is looking for)")
    print("  2. Attention scores (similarity with all keys)")
    print("  3. Attention weights (normalized scores)")
    print("  4. Value vectors (information from all tokens)")
    print("  5. Weighted values (values scaled by attention)")
    print("  6. Output vector (final representation for '{query_token}')\n")


def create_all_visualizations():
    """Create all Phase 1 visualizations."""
    print_section("Phase 1: Single-Head Attention Visualizations", char="=")

    # Create visualizations
    visualize_attention_heatmap()
    visualize_qkv_flow()

    print_section("All Phase 1 Visualizations Complete!", char="=")
    print("Output files:")
    print("  - visualizations/plots/03_attention/phase1/attention_heatmap.png")
    print("  - visualizations/plots/03_attention/phase1/qkv_flow.png")
    print()
    print("Next steps:")
    print("  1. Examine the attention heatmap - which tokens attend to each other?")
    print("  2. Study the QKV flow - understand the mechanism step-by-step")
    print("  3. Move to Phase 2: python phase2_multi_head/multi_head_attention.py")
    print()


if __name__ == "__main__":
    create_all_visualizations()
