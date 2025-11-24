"""
Phase 2: Multi-Head Self-Attention Implementation

Extends single-head attention to multiple parallel attention heads.

Key innovation: Instead of one attention mechanism, run H heads in parallel:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

Benefits:
- Multiple representation subspaces (different heads learn different patterns)
- Richer, more expressive attention
- Heads specialize during training (syntactic, semantic, positional patterns)

From "Attention Is All You Need" (Vaswani et al., 2017)
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.attention_utils import (compute_attention_entropy,
                                    scaled_dot_product_attention,
                                    verify_attention_weights)
from shared.utils import (DATA_PROCESSED_DIR, get_device, load_toy_sequence,
                          print_section, print_tensor_info, set_seed,
                          tokens_to_ids)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.

    Runs H parallel attention mechanisms ("heads"), each operating on a
    subset of the embedding dimensions (d_model / H).

    The key insight: Different heads can learn to attend to different aspects
    of the input (syntax, semantics, position, etc.)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model/embedding dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.dropout = dropout

        # Linear projections for all heads combined
        # Instead of creating separate W_q^i for each head, we create one large
        # matrix and split it into heads later (more efficient)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection to combine all heads
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ):
        """
        Forward pass of multi-head attention.

        Args:
            x: Input embeddings (batch, seq_len, d_model)
            mask: Optional attention mask (seq_len, seq_len) or compatible
            return_attention: Whether to return attention weights

        Returns:
            output: Attention output (batch, seq_len, d_model)
            attention_weights: (optional) Per-head weights (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape

        # Step 1: Linear projections for Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Split into multiple heads
        # Reshape from (batch, seq_len, d_model) to (batch, seq_len, n_heads, d_k)
        # Then transpose to (batch, n_heads, seq_len, d_k) for parallel processing
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Now: (batch, n_heads, seq_len, d_k)

        # Step 3: Apply scaled dot-product attention for each head in parallel
        # Q: (batch, n_heads, seq_len, d_k)
        # K: (batch, n_heads, seq_len, d_k)
        # V: (batch, n_heads, seq_len, d_k)
        context, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        # context: (batch, n_heads, seq_len, d_k)
        # attention_weights: (batch, n_heads, seq_len, seq_len)

        # Step 4: Concatenate heads
        # Transpose back to (batch, seq_len, n_heads, d_k)
        # Then reshape to (batch, seq_len, n_heads * d_k) = (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, d_model)

        # Step 5: Apply output projection
        output = self.W_o(context)  # (batch, seq_len, d_model)

        if return_attention:
            return output, attention_weights
        return output


def analyze_head_specialization(attention_weights: torch.Tensor, tokens: list):
    """
    Analyze what patterns each head has learned.

    Measures:
    - Entropy: Low = focused attention, High = diffuse attention
    - Primary attended tokens: Which tokens get most attention per head

    Args:
        attention_weights: (batch, n_heads, seq_len, seq_len)
        tokens: List of token strings
    """
    print("Analyzing head specialization:")
    print("=" * 70)

    batch_size, n_heads, seq_len, _ = attention_weights.shape

    # Remove batch dimension (assuming batch_size=1)
    attn = attention_weights.squeeze(0)  # (n_heads, seq_len, seq_len)

    for head_idx in range(n_heads):
        head_attn = attn[head_idx]  # (seq_len, seq_len)

        # Compute entropy for this head
        entropy = compute_attention_entropy(head_attn.unsqueeze(0)).squeeze()  # (seq_len,)
        mean_entropy = entropy.mean().item()

        # Get max entropy possible (uniform distribution)
        max_entropy = math.log(seq_len)
        normalized_entropy = mean_entropy / max_entropy

        print(f"\nHead {head_idx + 1}:")
        print(f"  Average entropy: {mean_entropy:.3f} (max: {max_entropy:.3f})")
        print(f"  Normalized entropy: {normalized_entropy:.3f} (1.0 = uniform)")

        # Characterize attention pattern
        if normalized_entropy < 0.3:
            pattern = "FOCUSED (attends to few tokens)"
        elif normalized_entropy > 0.7:
            pattern = "DIFFUSE (attends broadly)"
        else:
            pattern = "MODERATE (balanced attention)"

        print(f"  Pattern: {pattern}")

        # Find most attended tokens for each query
        head_attn_np = head_attn.detach().cpu().numpy()
        print(f"  Primary attention targets:")
        for i, query_token in enumerate(tokens):
            max_idx = head_attn_np[i].argmax()
            max_weight = head_attn_np[i, max_idx]
            max_token = tokens[max_idx]
            if max_idx == i:
                print(f"    '{query_token}' → '{max_token}' ({max_weight:.2f}) [SELF]")
            else:
                print(f"    '{query_token}' → '{max_token}' ({max_weight:.2f})")

    print("\n" + "=" * 70)


def run_multihead_demo():
    """Demonstrate multi-head self-attention on toy sequence."""
    print_section("Multi-Head Self-Attention Demo")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    # Configuration
    d_model = 64   # Embedding dimension
    n_heads = 4    # Number of attention heads
    d_k = d_model // n_heads  # 16 dimensions per head
    vocab_size = 20

    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_k (per head): {d_k}")
    print(f"  vocab_size: {vocab_size}\n")

    # Load toy sequence: "the cat sat on the mat"
    tokens, vocab = load_toy_sequence()
    print(f"Toy sequence: {' '.join(tokens)}")
    print(f"Tokens: {tokens}\n")

    # Convert tokens to IDs
    token_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)  # (1, seq_len)
    seq_len = len(tokens)

    # Create token embeddings
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    x = embedding(token_ids)  # (1, seq_len, d_model)

    print_tensor_info("Input embeddings", x)
    print()

    # Create multi-head attention module
    multihead_attn = MultiHeadAttention(d_model, n_heads).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in multihead_attn.parameters() if p.requires_grad)
    print(f"Multi-head attention parameters: {n_params:,}")
    print(f"  (For comparison: single-head would have {4 * d_model * d_model:,} params)\n")

    # Forward pass
    print("Running forward pass...\n")
    output, attention_weights = multihead_attn(x, return_attention=True)

    # Display results
    print_tensor_info("Output", output)
    print()
    print_tensor_info("Attention weights (all heads)", attention_weights)
    print()

    # Verify attention weights for each head
    print("Verifying attention weights for each head:")
    for head_idx in range(n_heads):
        head_attn = attention_weights[:, head_idx, :, :]  # (batch, seq_len, seq_len)
        is_valid = verify_attention_weights(head_attn)
        status = "✓" if is_valid else "✗"
        print(f"  Head {head_idx + 1}: {status}")
    print()

    # Analyze what each head learned
    analyze_head_specialization(attention_weights, tokens)

    # Compare heads
    print("\nHead diversity analysis:")
    print("-" * 70)
    attn_np = attention_weights.squeeze(0).detach().cpu().numpy()  # (n_heads, seq_len, seq_len)

    # Compute pairwise correlation between heads
    print("Correlation between heads (1.0 = identical patterns):")
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            # Flatten attention matrices and compute correlation
            attn_i = attn_np[i].flatten()
            attn_j = attn_np[j].flatten()
            corr = np.corrcoef(attn_i, attn_j)[0, 1]
            print(f"  Head {i+1} ↔ Head {j+1}: {corr:.3f}")
    print()

    # Save attention weights for visualization
    save_path = DATA_PROCESSED_DIR / "phase2_multihead_attention.pt"
    torch.save({
        'attention_weights': attention_weights.cpu(),
        'tokens': tokens,
        'n_heads': n_heads,
        'output': output.cpu()
    }, save_path)
    print(f"Saved multi-head attention weights to: {save_path}\n")

    print("✅ Multi-head self-attention demo complete!")
    print("   Run visualize.py to see multi-head grid and entropy analysis.\n")


if __name__ == "__main__":
    import numpy as np
    run_multihead_demo()
