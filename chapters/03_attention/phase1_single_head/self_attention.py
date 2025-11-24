"""
Phase 1: Single-Head Self-Attention Implementation

Implements scaled dot-product self-attention from scratch.

This is the core attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017):
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

Key concepts demonstrated:
- Query, Key, Value projections
- Scaled dot-product attention
- Importance of scaling by √d_k
- Self-attention (Q, K, V from same source)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import math

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    get_device,
    set_seed,
    load_toy_sequence,
    tokens_to_ids,
    print_section,
    print_tensor_info,
    DATA_PROCESSED_DIR
)
from shared.attention_utils import (
    scaled_dot_product_attention,
    verify_attention_weights
)


class SelfAttention(nn.Module):
    """
    Single-head self-attention module.

    Implements the fundamental attention mechanism with learned projections
    for Query, Key, and Value.

    In self-attention, Q, K, and V all come from the same input sequence,
    allowing each token to attend to all other tokens (including itself).
    """

    def __init__(self, d_model: int, d_k: int = None, dropout: float = 0.0):
        """
        Initialize self-attention module.

        Args:
            d_model: Model/embedding dimension
            d_k: Dimension for queries and keys (default: same as d_model)
            dropout: Dropout probability (default: 0.0)
        """
        super().__init__()

        # If d_k not specified, use d_model
        if d_k is None:
            d_k = d_model

        self.d_model = d_model
        self.d_k = d_k
        self.dropout = dropout

        # Linear projections for Q, K, V
        # These are the learned weight matrices that transform input to queries/keys/values
        self.W_q = nn.Linear(d_model, d_k, bias=False)  # Query projection
        self.W_k = nn.Linear(d_model, d_k, bias=False)  # Key projection
        self.W_v = nn.Linear(d_model, d_k, bias=False)  # Value projection

        # Optional: output projection (not in minimal formulation, but common in practice)
        self.W_o = nn.Linear(d_k, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ):
        """
        Forward pass of self-attention.

        Args:
            x: Input embeddings (batch, seq_len, d_model)
            mask: Optional attention mask (seq_len, seq_len) or compatible shape
                  1 = allowed, 0 = blocked
            return_attention: Whether to return attention weights

        Returns:
            output: Attention output (batch, seq_len, d_model)
            attention_weights: (optional) Attention weights (batch, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape

        # Step 1: Project input to Q, K, V
        # Each token gets its own query, key, and value vector
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_k)

        # Step 2: Compute scaled dot-product attention
        # This is where the magic happens!
        context, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        # context: (batch, seq_len, d_k)
        # attention_weights: (batch, seq_len, seq_len)

        # Step 3: Apply output projection
        output = self.W_o(context)  # (batch, seq_len, d_model)

        if return_attention:
            return output, attention_weights
        return output


def demonstrate_scaling_importance():
    """
    Demonstrate why scaling by √d_k is critical for attention.

    Without scaling, large d_k causes dot products to grow large,
    pushing softmax into saturation regions with tiny gradients.
    """
    print_section("Demonstrating Importance of Scaling")

    set_seed(42)
    device = get_device()

    # Test with different dimensions
    dimensions = [8, 32, 128]
    seq_len = 6

    print("Comparing attention with and without scaling for different d_k:\n")

    for d_k in dimensions:
        print(f"d_k = {d_k}")
        print("-" * 40)

        # Create random Q and K
        Q = torch.randn(1, seq_len, d_k, device=device)
        K = torch.randn(1, seq_len, d_k, device=device)

        # Compute scores without scaling
        scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))

        # Compute scores with scaling
        scores_scaled = scores_unscaled / math.sqrt(d_k)

        print(f"  Unscaled scores: min={scores_unscaled.min():.2f}, "
              f"max={scores_unscaled.max():.2f}, "
              f"std={scores_unscaled.std():.2f}")
        print(f"  Scaled scores:   min={scores_scaled.min():.2f}, "
              f"max={scores_scaled.max():.2f}, "
              f"std={scores_scaled.std():.2f}")

        # Apply softmax
        attn_unscaled = torch.softmax(scores_unscaled, dim=-1)
        attn_scaled = torch.softmax(scores_scaled, dim=-1)

        # Compute entropy (measure of uniformity)
        # Low entropy = peaked/focused, High entropy = uniform/diffuse
        def entropy(p):
            p = torch.clamp(p, min=1e-9)
            return -(p * torch.log(p)).sum(dim=-1).mean()

        ent_unscaled = entropy(attn_unscaled)
        ent_scaled = entropy(attn_scaled)

        print(f"  Attention entropy (unscaled): {ent_unscaled:.4f}")
        print(f"  Attention entropy (scaled):   {ent_scaled:.4f}")

        # Gradient magnitude (simulated)
        # In saturation, gradients are tiny
        grad_unscaled = attn_unscaled * (1 - attn_unscaled)  # Softmax gradient
        grad_scaled = attn_scaled * (1 - attn_scaled)

        print(f"  Gradient magnitude (unscaled): {grad_unscaled.mean():.6f}")
        print(f"  Gradient magnitude (scaled):   {grad_scaled.mean():.6f}")
        print()

    print("Observation: Without scaling, large d_k causes:")
    print("  1. Large score magnitudes → softmax saturation")
    print("  2. Low entropy → overly peaked attention")
    print("  3. Tiny gradients → training difficulties")
    print("\nScaling by √d_k keeps scores in a reasonable range! ✓\n")


def run_single_head_demo():
    """Demonstrate single-head self-attention on toy sequence."""
    print_section("Single-Head Self-Attention Demo")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    # Configuration
    d_model = 64  # Embedding dimension
    d_k = 64      # Query/Key dimension (same as d_model for single-head)
    vocab_size = 20

    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  d_k: {d_k}")
    print(f"  vocab_size: {vocab_size}\n")

    # Load toy sequence: "the cat sat on the mat"
    tokens, vocab = load_toy_sequence()
    print(f"Toy sequence: {' '.join(tokens)}")
    print(f"Tokens: {tokens}")
    print(f"Vocabulary: {vocab}\n")

    # Convert tokens to IDs
    token_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)  # (1, seq_len)
    seq_len = len(tokens)

    print(f"Token IDs: {token_ids.squeeze().tolist()}\n")

    # Create token embeddings (random for demo)
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    x = embedding(token_ids)  # (1, seq_len, d_model)

    print_tensor_info("Input embeddings", x)
    print()

    # Create self-attention module
    self_attn = SelfAttention(d_model, d_k).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in self_attn.parameters() if p.requires_grad)
    print(f"Self-attention parameters: {n_params:,}\n")

    # Forward pass
    print("Running forward pass...\n")
    output, attention_weights = self_attn(x, return_attention=True)

    # Display results
    print_tensor_info("Output", output)
    print()
    print_tensor_info("Attention weights", attention_weights)
    print()

    # Verify attention weights are valid probabilities
    is_valid = verify_attention_weights(attention_weights)
    if is_valid:
        print("✓ Attention weights are valid (non-negative, sum to 1)\n")
    else:
        print("✗ Attention weights are INVALID!\n")

    # Show attention weights for each query
    attn_np = attention_weights.squeeze(0).detach().cpu().numpy()
    print("Attention patterns:")
    print("=" * 70)
    print(f"{'Query':<10} -> {'Keys (attention weights)':>50}")
    print("=" * 70)

    for i, query_token in enumerate(tokens):
        weights = attn_np[i]
        # Create string showing which tokens are attended to
        attn_str = " | ".join([f"{tok}:{weights[j]:.2f}" for j, tok in enumerate(tokens)])
        print(f"{query_token:<10} -> {attn_str}")

    print("=" * 70)
    print()

    # Identify strongest attention for each query
    print("Strongest attention for each query:")
    print("-" * 50)
    for i, query_token in enumerate(tokens):
        weights = attn_np[i]
        max_idx = weights.argmax()
        max_weight = weights[max_idx]
        max_token = tokens[max_idx]
        print(f"  '{query_token}' attends most to '{max_token}' ({max_weight:.3f})")
    print()

    # Save attention weights for visualization
    save_path = DATA_PROCESSED_DIR / "phase1_attention_weights.pt"
    torch.save({
        'attention_weights': attention_weights.cpu(),
        'tokens': tokens,
        'output': output.cpu()
    }, save_path)
    print(f"Saved attention weights to: {save_path}\n")

    print("✅ Single-head self-attention demo complete!")
    print("   Run visualize.py to see attention heatmaps and QKV flow diagrams.\n")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_scaling_importance()
    run_single_head_demo()
