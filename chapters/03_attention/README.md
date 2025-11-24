# Self-Attention & Multi-Head Attention: A Learning Project

A hands-on implementation project to understand the core mechanism that makes transformers work — attention.

---

## Overview

Attention is the defining innovation of transformer architectures. It allows each token in a sequence to dynamically gather information from all other tokens, weighted by relevance.

This project builds attention from the ground up, progressing through four phases:
1. Implement single-head scaled dot-product attention from scratch
2. Extend to multi-head attention with parallel processing
3. Visualize attention patterns and interpret what the model learns
4. Apply causal masking for autoregressive generation

By the end, you will understand not just how attention works, but why it works — and what happens when you modify its structure.

---

## The Core Problem

**How can a model determine which parts of a sequence are relevant to each other?**

Traditional RNNs process sequences left-to-right, maintaining a hidden state. But this creates bottlenecks:
- Information from early tokens must pass through many steps to reach later tokens
- The model has a fixed-size hidden state to compress everything
- Parallel processing is impossible (sequential by nature)

**Self-attention solves this by:**
- Allowing every token to directly attend to every other token
- Dynamically computing relevance weights (attention scores)
- Processing all positions in parallel
- Creating token representations that incorporate context from the entire sequence

The fundamental operation:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q (Query): "What am I looking for?"
- K (Key): "What do I contain?"
- V (Value): "What information do I provide?"

---

## Phase 1: Scaled Dot-Product Self-Attention

### The Mechanism

**Self-attention** means queries, keys, and values all come from the same sequence. Each token attends to all tokens (including itself).

**The Algorithm (for a single token):**

1. **Compute Queries, Keys, Values**
   - Project input through learned weight matrices
   - Q = X W_Q, K = X W_K, V = X W_V
   - Each token gets its own q, k, v vectors

2. **Calculate Attention Scores**
   - For query token i, compute dot product with all keys
   - score_ij = q_i · k_j
   - Higher score = more relevant

3. **Scale**
   - Divide by √d_k (dimension of key vectors)
   - Prevents softmax saturation for large dimensions
   - Maintains stable gradients

4. **Normalize with Softmax**
   - Convert scores to probability distribution
   - attention_weights_i = softmax(scores_i / √d_k)
   - Weights sum to 1 across all keys

5. **Compute Weighted Sum**
   - output_i = Σ (attention_weight_ij × v_j)
   - Token i's output is weighted combination of all value vectors
   - High-weight tokens contribute more

**Matrix Form (for entire sequence):**

```
Scores = Q K^T                    # (seq_len, seq_len)
Attention = softmax(Scores / √d_k) # (seq_len, seq_len)
Output = Attention V              # (seq_len, d_model)
```

### Why Scaling Matters

**Problem without scaling:**

For large d_k (e.g., 64, 128), dot products grow large in magnitude:
- q · k ∝ √d_k (assuming unit variance inputs)
- Large values push softmax into saturation regions
- Gradients become tiny → training fails

**Solution: Scale by √d_k**

- Normalizes dot products to unit variance
- Keeps softmax in sensitive gradient region
- Empirically critical for deep models

**Experiment to demonstrate:**
- Compute attention with d_k = 8, 32, 128
- With and without scaling
- Visualize gradient magnitudes
- Show training stability difference

### Self-Attention vs Other Mechanisms

| Mechanism | Connections | Parallel | Max Path Length | Parameters |
|-----------|-------------|----------|-----------------|------------|
| **RNN** | Sequential | No | O(n) | O(d²) |
| **CNN** | Local window | Yes | O(log n) (stacked) | O(k·d²) |
| **Self-Attention** | All-to-all | Yes | O(1) | O(3d²) |

Self-attention connects every pair of positions directly (O(1) path), enabling:
- Long-range dependencies without sequential bottlenecks
- Parallel computation across all positions
- Dynamic, input-dependent weighting

### Implementation Details

**Input:**
- Sequence of tokens: X ∈ R^(seq_len × d_model)
- Each token is a d_model-dimensional vector (from embeddings + position)

**Projection Matrices:**
- W_Q, W_K, W_V ∈ R^(d_model × d_k)
- Typically d_k = d_model (for single-head)
- Learned during training

**Attention Mask (optional for Phase 1, required for Phase 4):**
- Binary matrix indicating valid attention connections
- Used for causal masking (future positions forbidden)
- Applied before softmax: scores = scores + mask

**Output:**
- Same shape as input: R^(seq_len × d_model)
- Each token's representation now incorporates context

### Toy Example

**Sequence:** "the cat sat on the mat"

**For token "sat" (position 2):**

1. Compute q_2 (query for "sat")
2. Compute dot products with all keys:
   - q_2 · k_0 (key for "the") → score: 0.8
   - q_2 · k_1 (key for "cat") → score: 3.2 ← high!
   - q_2 · k_2 (key for "sat") → score: 1.5
   - q_2 · k_3 (key for "on") → score: 0.5
   - q_2 · k_4 (key for "the") → score: 0.7
   - q_2 · k_5 (key for "mat") → score: 2.1 ← medium-high

3. Scale and softmax → attention weights:
   - "the": 0.05, "cat": 0.45, "sat": 0.15, "on": 0.05, "the": 0.10, "mat": 0.20

4. Output: 0.45×v_cat + 0.20×v_mat + 0.15×v_sat + ...
   - "sat" representation now heavily incorporates "cat" (subject) and "mat" (object)

### Visualizations

**1. Attention Weight Heatmap**
- Rows: query positions (0-5)
- Columns: key positions (0-5)
- Color: attention weight after softmax (0 to 1)
- Each row sums to 1
- Shows which tokens attend to which — the core output of the mechanism

**2. Query-Key-Value Flow Diagram**
- For single token (e.g., "sat")
- Show vector transformations: input → Q/K/V → scores → weights → output
- Annotate with dimensions at each step
- Illustrates the complete mechanism from input to output

---

## Phase 2: Multi-Head Attention

### The Motivation

**Single-head attention has a limitation:** It creates one representation subspace. The model learns one way to compute relevance.

**Problem:** Different types of relationships matter:
- Syntactic dependencies (subject-verb agreement)
- Semantic similarity (synonyms, related concepts)
- Positional relationships (adjacent tokens, distant tokens)
- Task-specific patterns

**Solution: Multiple attention heads in parallel**

Each head learns to focus on different aspects of the input.

### The Mechanism

**Multi-head attention:**

1. **Split into H heads**
   - Divide d_model into H equal parts
   - Each head operates on d_k = d_model / H dimensions

2. **Parallel attention**
   - Each head has its own W_Q^h, W_K^h, W_V^h matrices
   - Independently compute attention: head_h = Attention(Q^h, K^h, V^h)
   - H different attention patterns computed simultaneously

3. **Concatenate**
   - Combine all head outputs: concat(head_1, ..., head_H)
   - Result: (seq_len, d_model) — same shape as input

4. **Final projection**
   - Linear transformation: output = concat(heads) W_O
   - W_O ∈ R^(d_model × d_model)
   - Allows heads to interact and combine information

**Full formula:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O

where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
```

### Why Multiple Heads?

**Ensemble of attention patterns:**
- Head 1 might focus on adjacent tokens (local syntax)
- Head 2 might focus on the first token (global context)
- Head 3 might focus on same-POS tokens (grammatical patterns)
- Head 4 might attend uniformly (backup/regularization)

**Richer representations:**
- Single attention is one-dimensional measure of relevance
- Multi-head creates H-dimensional relevance space
- More expressive, more flexible

**Empirical evidence:**
- Transformers with 8-16 heads consistently outperform single-head
- Different heads learn interpretable patterns (subject-verb, coreference, etc.)
- Ablation studies: removing any head hurts performance

### Implementation Details

**Standard Configuration:**
- Number of heads: H = 8 (common) or 12-16 (larger models)
- d_model = 512 (original Transformer)
- d_k = d_v = d_model / H = 64

**Parameter Count:**

For each head:
- W_Q^h: d_model × d_k
- W_K^h: d_model × d_k  
- W_V^h: d_model × d_k

Total: 3 × H × d_model × d_k = 3 × d_model²

Output projection:
- W_O: d_model × d_model

**Grand total:** 4 × d_model² parameters

**Computational Cost:**

For sequence length L:
- Attention scores: O(L² × d_model) — dominant cost
- Linear projections: O(L × d_model²)
- Overall: O(L² × d_model + L × d_model²)
- For long sequences, the L² term dominates!

**Memory Layout:**

Efficient implementation uses 4D tensors:
- Q, K, V: (batch, n_heads, seq_len, d_k)
- Attention: (batch, n_heads, seq_len, seq_len)
- Output: (batch, seq_len, n_heads × d_k) → reshape → (batch, seq_len, d_model)

### Head Specialization

**Typical patterns observed in trained models:**

1. **Positional heads:** Attend primarily to specific relative positions
   - Previous token: "the [CAT] sat" → "cat" attends to "the"
   - Next token: autoregressive prediction

2. **Syntactic heads:** Follow grammatical structure
   - Subject-verb: "the cat [SAT]" → "sat" attends to "cat"
   - Determiner-noun: "[THE] cat sat" → "the" attends to "cat"

3. **Semantic heads:** Group related concepts
   - "cat" and "mat" both attend to each other (related objects)

4. **Broad context heads:** Attend uniformly or to sentence boundaries
   - Capture global sentence meaning

5. **Self-attention heads:** Strong diagonal pattern
   - Tokens attend primarily to themselves
   - Acts as residual connection

**Important:** These patterns emerge during training — they are not hardcoded!

### Visualizations

**1. Multi-Head Attention Grid**
- Grid layout: H rows (one per head)
- Each cell: attention heatmap for that head
- For toy sequence "the cat sat on the mat"
- Shows diversity of attention patterns across all heads
- Most important visualization for understanding multi-head behavior

**2. Attention Entropy by Head**
- Bar chart: entropy of attention distribution for each head
- Low entropy = focused (peaky) attention on few tokens
- High entropy = diffuse (uniform) attention across many tokens
- Reveals which heads are specialized vs general-purpose

---

## Phase 3: Attention Visualization & Interpretation

### Why Visualization Matters

Attention weights are the most interpretable part of transformers:
- They directly show which tokens influence which others
- Patterns often align with linguistic/semantic relationships
- Visualizations can reveal what the model learned — or failed to learn

This phase focuses on creating rich, informative visualizations.

### Visualization Techniques

This phase focuses on creating rich, informative visualizations beyond basic heatmaps. We implement two advanced visualization methods that provide unique insights into attention patterns.

### Interpreting Attention Patterns

**Common patterns and their meanings:**

**1. Diagonal Pattern (Self-Attention)**
- Tokens attend strongly to themselves
- Indicates model preserves token identity
- Acts like a residual connection

**2. Previous Token Attention**
- Strong band just below diagonal
- Model attending to immediate context
- Common in autoregressive models

**3. Broad/Uniform Attention**
- All attention weights roughly equal
- Model unsure or aggregating global context
- Sometimes indicates untrained/undertrained head

**4. Sparse/Focused Attention**
- One or two tokens get majority of weight
- Model strongly confident in relevance
- Often seen in well-trained models

**5. Attention to [CLS] or [SEP] tokens**
- Special tokens gather/distribute information
- Acts as information hub
- Common in BERT-style models

**6. Positional Patterns**
- Attention to specific relative positions (+1, -1, +2, etc.)
- Indicates learned syntactic patterns
- Example: verbs attending to subjects 3-4 positions back

### Toy Example Analysis

**Sequence:** "the cat sat on the mat"

**Expected patterns in a trained model:**

**Head 1 (Subject-Verb):**
- "sat" attends strongly to "cat" (subject)
- Shows grammatical relationship

**Head 2 (Verb-Object):**
- "sat" attends to "mat" (object via "on")
- Shows semantic relationship

**Head 3 (Determiner-Noun):**
- "the" (position 0) attends to "cat"
- "the" (position 4) attends to "mat"

**Head 4 (Sequential):**
- Each token attends primarily to previous token
- Captures sequential flow

### Visualizations

**1. Attention Flow Diagram**
- Tokens as nodes arranged horizontally
- Arrows from query to key positions
- Arrow thickness = attention weight
- Only show edges above threshold (e.g., > 0.15)
- Clearly shows information flow between tokens

**2. 3D Attention Surface**
- X-axis: query position, Y-axis: key position, Z-axis: attention weight
- Creates 3D "landscape" of attention patterns
- Animated rotation (360°) shows structure from all angles
- Provides intuitive geometric understanding of attention

---

## Phase 4: Causal Masking & Autoregressive Attention

### The Problem

**Standard self-attention allows tokens to attend to ALL positions — including future tokens.**

This is fine for encoding (BERT-style), but breaks autoregressive generation (GPT-style):

**Example:**
- Sequence: "the cat sat on"
- Predicting next token: "the"
- If position 3 ("on") can see position 4 ("the"), it's cheating!
- Model must predict future from past only

**Solution: Causal masking** — prevent attention to future positions.

### Causal Mask Mechanism

**Mask Definition:**

Lower triangular matrix (with diagonal = 1):

```
Position:  0    1    2    3    4    5
  0      [1    0    0    0    0    0]  ← pos 0 can only see pos 0
  1      [1    1    0    0    0    0]  ← pos 1 sees pos 0-1
  2      [1    1    1    0    0    0]  ← pos 2 sees pos 0-2
  3      [1    1    1    1    0    0]
  4      [1    1    1    1    1    0]
  5      [1    1    1    1    1    1]  ← pos 5 sees all (past + self)
```

**Implementation:**

Before softmax, set future positions to -∞:

```
scores = Q K^T / √d_k
scores = scores + mask  # where mask[i,j] = 0 if j ≤ i else -∞
attention_weights = softmax(scores)  # -∞ → 0 after softmax
```

**Result:** Token at position i can only attend to positions 0, 1, ..., i.

### Why -∞ Instead of Zero?

**Common mistake:** Mask by setting scores to 0

**Problem:** softmax(0) is not zero!

```
softmax([2.0, 0.0, 3.0]) = [0.24, 0.03, 0.73]
```

The zero score still contributes (0.03).

**Correct approach:** Set to -∞

```
softmax([2.0, -∞, 3.0]) = [0.27, 0.00, 0.73]
```

After softmax, -∞ becomes exactly 0 — no information leaks from future.

**In practice:** Use large negative number like -1e9 (close enough to -∞).

### Autoregressive Generation

**Setup:**

1. Start with prompt: "the cat"
2. Predict next token: "sat"
3. Append to sequence: "the cat sat"
4. Predict next: "on"
5. Repeat until [EOS] or max length

**Key requirement:** At each step, model only sees past tokens.

Causal masking ensures this during both training and inference.

**Training:**
- Process entire sequence in parallel
- Apply causal mask so each position only sees past
- Compute loss at all positions simultaneously
- Efficient!

**Inference:**
- Generate one token at a time
- Only need to mask future (since future doesn't exist yet)
- Can use KV caching (Chapter 6) to speed up

### Visualization of Causality

**Causal Mask Heatmap:**
- Lower triangle = 1 (allowed)
- Upper triangle = 0 (blocked)
- Clear visual of information flow

**Causal Attention Pattern:**
- Attention heatmap with causal mask applied
- Upper right triangle is all zeros
- Shows how attention is forced to be backward-looking

**Comparison: Bidirectional vs Causal:**
- Side-by-side heatmaps
- Same sequence, same model
- Left: full attention (BERT-style)
- Right: causal attention (GPT-style)
- Shows dramatic difference

### Implementation Details

**Efficient Mask Generation (PyTorch):**

```python
import torch

def create_causal_mask(seq_len):
    # Upper triangular matrix of -∞
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

**Applying Mask (PyTorch):**

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
scores = scores + causal_mask  # Broadcasting applies to batch/heads
attention_weights = torch.softmax(scores, dim=-1)
```

**Attention with Optional Masking (PyTorch):**

```python
import torch
import math

def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask  # or: scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

### Types of Masking

**1. Causal Mask (Autoregressive)**
- Lower triangular
- For GPT-style generation
- This is Phase 4 focus

**2. Padding Mask**
- Hide padding tokens (in batched sequences)
- Mask[i, j] = 0 if token j is padding
- Prevents model from attending to meaningless pad tokens

**3. Combined Mask**
- Causal AND padding
- Common in practice for variable-length sequences

**4. Custom Attention Masks**
- Arbitrary patterns (e.g., local attention, blocked attention)
- Explored in later chapters (Chapter 7)

### Experiments

**1. Verify Causal Property**
- Generate with causal mask
- Manually shuffle future tokens (should not affect generation)
- Confirms model truly doesn't use future information

**2. Break Causality**
- Train model WITH causal mask
- Test WITHOUT causal mask → model should be confused
- Or vice versa → model cheats during training

**3. Next-Token Prediction**
- Toy task: predict next token in sequence
- Compare: with vs without causal mask
- With mask: forces model to learn patterns
- Without mask: model can cheat by looking ahead

**4. Attention Pattern Analysis**
- How does causal mask change attention patterns?
- Do heads still specialize under causal constraint?
- Are patterns more local/focused?

### Visualizations

**1. Causal Mask Matrix**
- Binary heatmap showing mask structure
- Lower triangle = white (attention allowed)
- Upper triangle = black (attention blocked)
- Clearly illustrates the triangular causality constraint

**2. Bidirectional vs Causal Comparison**
- Same input sequence processed two ways
- Left: full bidirectional attention
- Right: causal masked attention (upper triangle zeroed)
- Side-by-side comparison shows what information is lost with causality

---

## Comparative Analysis

### Single-Head vs Multi-Head

| Aspect | Single-Head | Multi-Head (H=8) |
|--------|-------------|------------------|
| **Parameters** | 3 × d² | 4 × d² |
| **Attention Patterns** | 1 | H (diverse) |
| **Expressiveness** | Limited | Rich |
| **Interpretability** | Simple | Complex but revealing |
| **Performance** | Baseline | Significantly better |
| **Computation** | 1× | ~H× (parallelizable) |

Multi-head is the clear winner for almost all tasks.

### Attention Mechanisms Comparison

| Property | Self-Attention | Cross-Attention | Masked Self-Attention |
|----------|----------------|-----------------|----------------------|
| **Q Source** | Same sequence | Different sequence | Same sequence |
| **K/V Source** | Same sequence | Different sequence | Same sequence |
| **Mask** | Optional | Padding only | Causal + padding |
| **Use Case** | Encoder | Encoder-Decoder | Decoder (generation) |
| **Information Flow** | Bidirectional | Cross-modal | Causal (unidirectional) |

This chapter focuses on self-attention and masked self-attention. Cross-attention appears in encoder-decoder models.

### Attention vs Other Mechanisms

| Mechanism | All-to-All | Learnable | Parallel | Max Path | Content-Based |
|-----------|-----------|-----------|----------|----------|---------------|
| **Attention** | ✓ | ✓ | ✓ | O(1) | ✓ |
| **RNN/LSTM** | ✗ | ✓ | ✗ | O(n) | ✗ |
| **CNN** | ✗ (local) | ✓ | ✓ | O(log n) | ✗ |
| **Fully-Connected** | ✓ | ✓ | ✓ | O(1) | ✗ |

Attention uniquely combines all-to-all connections, content-based routing, and parallel computation.

---

## Learning Outcomes

By completing this project, you will:
- Implement scaled dot-product attention from first principles
- Understand the role of Query, Key, and Value projections
- Extend single-head to multi-head attention
- Visualize attention patterns and interpret what models learn
- Apply causal masking for autoregressive generation
- Recognize common attention patterns (self-attention, syntactic, semantic)
- Understand computational complexity (O(L² d_model))
- Appreciate why attention revolutionized NLP

**Key Insights:**
- Attention is a differentiable key-value lookup mechanism
- Scaling (dividing by √d_k) is critical for training stability
- Multiple heads learn diverse, complementary patterns
- Attention weights are interpretable and often linguistically meaningful
- Causal masking enables parallel training of autoregressive models
- The O(L²) complexity is attention's main limitation (addressed in Chapter 7)

---

## Project Structure

```
03_attention/
├── phase1_single_head/
│   ├── self_attention.py             # Single-head self-attention implementation
│   └── visualize.py                  # Heatmap + QKV flow diagram
├── phase2_multi_head/
│   ├── multi_head_attention.py       # Full multi-head implementation
│   └── visualize.py                  # Multi-head grid + entropy analysis
├── phase3_visualization/
│   ├── attention_flow.py             # Node-link flow diagrams
│   └── attention_3d.py               # 3D attention landscapes
├── phase4_causal/
│   ├── causal_attention.py           # Masked self-attention with causal mask
│   └── visualize.py                  # Mask structure + bidirectional comparison
├── shared/
│   ├── attention_utils.py            # Common utilities
│   ├── toy_model.py                  # Minimal transformer for testing
│   └── visualization_utils.py        # Plotting helpers
└── README.md

# Data and outputs stored in root-level folders:
# - ../../data/raw/03_attention/
#   - toy_sequences.txt                    # Test sequences
# - ../../data/processed/03_attention/
#   - attention_weights.pt                 # Saved attention patterns
#   - multi_head_patterns.pt               # Head-specific patterns
# - ../../visualizations/plots/03_attention/
#   - phase1_attention_heatmap.png
#   - phase1_qkv_flow.png
#   - phase2_multi_head_grid.png
#   - phase2_attention_entropy.png
#   - phase3_attention_flow.png
#   - phase3_3d_attention.gif              # Rotating 3D landscape
#   - phase4_causal_mask.png
#   - phase4_bidirectional_vs_causal.png
#   - summary_all_phases.png               # Complete comparison
# - ../../models/checkpoints/03_attention/
#   - single_head_model.pt                 # Trained single-head
#   - multi_head_model.pt                  # Trained multi-head
#   - causal_model.pt                      # Trained with causal mask
```

---

## Getting Started

### Implementation Framework

**All implementations in this chapter use PyTorch.**

We build attention mechanisms from scratch using PyTorch tensors and operations. This provides:
- Clear, readable code that matches the mathematical notation
- GPU acceleration for experiments
- Native support for batching and automatic differentiation
- Easy integration with the rest of the project

### Recommended Approach

**Start with Phase 1:**
1. Implement scaled dot-product attention for a single pair of vectors
2. Extend to full sequence (matrix form)
3. Verify scaling is necessary
4. Visualize attention for toy sequence

**Progress to Phase 2:**
1. Split into multiple heads
2. Implement parallel attention
3. Concatenate and project
4. Analyze head specialization
5. Compare to single-head performance

**Then Phase 3:**
1. Create rich visualizations
2. Explore interactive tools
3. Analyze patterns statistically
4. Interpret what the model learned

**Finally Phase 4:**
1. Implement causal masking
2. Apply to generation task
3. Verify causality property
4. Compare to bidirectional attention

### Toy Sequence

All phases use: **"the cat sat on the mat"**

- 6 tokens (perfect for visualization)
- Repeated word "the" (tests positional understanding)
- Clear semantic relationships (subject-verb-object)
- Simple enough to interpret attention patterns

### Model Parameters

**For learning and visualization:**
- Embedding dimension: d_model = 64
- Number of heads: H = 4 (Phase 2+)
- Head dimension: d_k = d_v = 16 (d_model / H)
- Sequence length: 6 (toy sequence)
- Vocabulary: ~100 words

**For experiments:**
- Increase to d_model = 128, H = 8 for more realistic setting
- Test on longer sequences (up to 32 tokens)

---

## Mathematical Foundations

### Scaled Dot-Product Attention

**Input:**
- Q ∈ R^(L × d_k): queries
- K ∈ R^(L × d_k): keys  
- V ∈ R^(L × d_v): values

**Algorithm:**

```
1. Compute attention scores:
   S = Q K^T                          # (L, L)

2. Scale:
   S_scaled = S / √d_k                # Prevent saturation

3. Optional masking:
   S_masked = S_scaled + Mask         # Add -∞ where needed

4. Normalize:
   A = softmax(S_masked)              # (L, L), rows sum to 1

5. Aggregate:
   Output = A V                       # (L, d_v)
```

**Properties:**
- Permutation equivariant (without position encoding)
- O(L² d_k) time complexity
- O(L²) space for attention matrix
- Differentiable end-to-end

### Multi-Head Attention

**Formal definition:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W^O

where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

and:
- W_i^Q ∈ R^(d_model × d_k)
- W_i^K ∈ R^(d_model × d_k)
- W_i^V ∈ R^(d_model × d_v)
- W^O ∈ R^(H d_v × d_model)
```

**Typical configuration:**
- d_k = d_v = d_model / H
- Each head operates on subset of dimensions
- Output projection combines information

### Why Attention Works

**Information Routing:**
- Each token "asks" questions (query)
- Other tokens "advertise" content (key)
- Relevance = query-key similarity
- Information transferred = weighted values

**Content-Based Addressing:**
- Unlike fixed CNNs or sequential RNNs
- Connections determined by content, not position
- Dynamically adapts to input

**Gradient Flow:**
- Direct paths between all positions
- No vanishing gradient through long sequences
- Each token's gradient flows through attention weights

---

## References

**Original Papers:**

1. **Attention Mechanism:** Vaswani et al. (2017). "Attention Is All You Need"
2. **Analysis:** Michel et al. (2019). "Are Sixteen Heads Really Better than One?"
3. **Interpretability:** Clark et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention"
4. **Visualization:** Vig (2019). "A Multiscale Visualization of Attention in the Transformer Model"

**Key Implementations:**

- GPT (Causal Multi-Head): OpenAI
- BERT (Bidirectional Multi-Head): Google
- T5 (Encoder-Decoder): Google
- LLaMA (Causal + RoPE): Meta AI

**Interpretability Tools:**

- BertViz (Jesse Vig)
- Transformers Interpret
- Attention Flow visualization (tensor2tensor)

---

## Experiments to Try

### 1. Attention Head Pruning
- Train multi-head model
- Remove heads one at a time
- Measure performance degradation
- Find minimum number of heads needed

### 2. Attention Distance Analysis
- Measure average attention distance per layer
- Does attention become more local or global in deeper layers?
- How does causal masking affect this?

### 3. Attention Entropy Evolution
- Track attention entropy during training
- Does it decrease (more focused) or increase (more diffuse)?
- Per-head analysis

### 4. Custom Masking Patterns
- Implement local attention (only attend to ±k positions)
- Implement strided attention (every n-th position)
- Compare to full attention

### 5. Attention Rollout
- Multiply attention matrices across layers
- Trace information flow from input to output
- Visualize end-to-end attention paths

---

## Quick Start Guide

### Prerequisites

```bash
# From project root, ensure dependencies are installed
uv sync
```

This installs PyTorch and all other required dependencies (matplotlib, numpy, etc.).

### Option 1: Run Everything (Recommended for First Time)

```bash
cd chapters/03_attention
./run_all.sh
```

This runs all phases sequentially (~10-15 minutes total).

### Option 2: Run Individual Phases

#### Test First

```bash
uv run python test_all.py
```

#### Phase 1: Single-Head Attention

```bash
# Implementation and visualizations
uv run python phase1_single_head/self_attention.py
uv run python phase1_single_head/visualize.py
```

**Outputs:**
- `visualizations/plots/03_attention/phase1/attention_heatmap.png`
- `visualizations/plots/03_attention/phase1/qkv_flow.png`

#### Phase 2: Multi-Head Attention

```bash
# Implementation and visualizations
uv run python phase2_multi_head/multi_head_attention.py
uv run python phase2_multi_head/visualize.py
```

**Outputs:**
- `visualizations/plots/03_attention/phase2/multi_head_grid.png`
- `visualizations/plots/03_attention/phase2/attention_entropy.png`

#### Phase 3: Visualization & Interpretation

```bash
# Advanced visualization techniques
uv run python phase3_visualization/attention_flow.py
uv run python phase3_visualization/attention_3d.py
```

**Outputs:**
- `visualizations/plots/03_attention/phase3/attention_flow.png`
- `visualizations/animations/03_attention/phase3/3d_attention.gif`

#### Phase 4: Causal Masking

```bash
# Implementation and visualizations
uv run python phase4_causal/causal_attention.py
uv run python phase4_causal/visualize.py
```

**Outputs:**
- `visualizations/plots/03_attention/phase4/causal_mask.png`
- `visualizations/plots/03_attention/phase4/bidirectional_vs_causal.png`

---

## Key Visualizations to Check

### Phase 1
- `phase1/attention_heatmap.png` - Attention weight matrix
- `phase1/qkv_flow.png` - Query-Key-Value mechanism diagram

### Phase 2
- `phase2/multi_head_grid.png` - All heads displayed together
- `phase2/attention_entropy.png` - Head specialization via entropy

### Phase 3
- `phase3/attention_flow.png` - Node-link flow diagram
- `phase3/3d_attention.gif` - Rotating 3D attention landscape

### Phase 4
- `phase4/causal_mask.png` - Triangular mask structure
- `phase4/bidirectional_vs_causal.png` - Side-by-side comparison

---

## Expected Runtime

| Phase | Script | Time (MPS) | Time (CPU) |
|-------|--------|------------|------------|
| Test | `test_all.py` | ~20 sec | ~40 sec |
| Phase 1 | Implementation | ~5 sec | ~10 sec |
| Phase 1 | Visualizations | ~10 sec | ~15 sec |
| Phase 2 | Implementation | ~10 sec | ~20 sec |
| Phase 2 | Visualizations | ~15 sec | ~25 sec |
| Phase 3 | Flow diagrams | ~10 sec | ~20 sec |
| Phase 3 | 3D landscapes | ~15 sec | ~30 sec |
| Phase 4 | Implementation | ~10 sec | ~15 sec |
| Phase 4 | Visualizations | ~10 sec | ~20 sec |
| **Total** | **All phases** | **~5-7 min** | **~10-15 min** |

*Times are approximate. No training required for this chapter — all models are small and run in inference mode.*

---

## Troubleshooting

### Import Errors

Make sure you're running from the correct directory:

```bash
# From project root
cd chapters/03_attention
uv run python <script>.py
```

### Out of Memory

Unlikely for this chapter (models are tiny), but if it happens:
- Reduce sequence length
- Reduce number of heads or embedding dimension

### Visualization Issues

Make sure matplotlib backend is properly configured:

```python
# Add to top of script if visualizations don't appear
import matplotlib
matplotlib.use('Agg')  # For saving without display
```

### Numerical Instability

If you see NaN in attention weights:
- Check scaling is applied (divide by √d_k using `math.sqrt(d_k)`)
- Verify softmax is on correct dimension (`torch.softmax(scores, dim=-1)`)
- Use float32, not float16 (for these toy experiments: `torch.float32`)

---

**Attention is the heart of transformers. Master it, and everything else falls into place!**

