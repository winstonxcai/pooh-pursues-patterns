# Positional Embeddings: A Learning Project

A hands-on implementation project to understand how transformers encode position information from scratch.

---

## Overview

Transformers process sequences as sets—they have no built-in concept of order. Without positional information, "the cat sat on the mat" and "the mat sat on the cat" would be indistinguishable.

This project builds intuition for how position is injected into transformer models, progressing through four phases:
1. Implement and compare classic sinusoidal and learned positional embeddings
2. Build RoPE (Rotary Position Embeddings) from scratch
3. Implement ALiBi (Attention with Linear Biases)
4. Demonstrate the catastrophic failure of attention without position

Each phase includes 3D visualizations showing how position shapes the representation space.

---

## The Core Problem

**Transformers are permutation-invariant:** Self-attention computes relationships between all tokens simultaneously using dot products. Without position encoding:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

This operation treats input as an unordered set. The tokens "cat" at position 0 and position 5 would be identical.

**Solution:** Inject positional information either by:
- Adding position vectors to token embeddings (sinusoidal, learned, RoPE)
- Adding position-dependent biases to attention scores (ALiBi)

---

## Phase 1: Classic Positional Embeddings

### Sinusoidal Positional Embeddings (Vaswani et al., 2017)

**The Original Transformer Approach**

Uses fixed sine and cosine functions at different frequencies to encode position:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` = position in sequence (0, 1, 2, ...)
- `i` = dimension index (0 to d_model/2)
- `d_model` = embedding dimension

**Key Properties:**
- Deterministic and parameter-free
- Each position gets a unique encoding
- Linear relationships: PE(pos+k) can be expressed as linear function of PE(pos)
- Generalizes to any sequence length (extrapolation)

**Implementation Details:**
- Create position encoding matrix: (max_seq_len, d_model)
- Add to token embeddings: `x = token_embedding + pos_embedding`
- No learnable parameters

**Why these specific frequencies?**
- Low frequencies (outer dimensions): encode global position
- High frequencies (inner dimensions): encode fine-grained position
- Creates a unique "fingerprint" for each position

### Learned Positional Embeddings

**Trainable Alternative**

Instead of fixed sinusoids, learn position embeddings as parameters:

```python
self.pos_embedding = nn.Embedding(max_seq_len, d_model)
```

**Key Properties:**
- Learnable parameters: (max_seq_len, d_model)
- Optimized for specific tasks/datasets
- Cannot extrapolate beyond max_seq_len
- Used in BERT, GPT-2, GPT-3

**Implementation Details:**
- Create embedding table for positions 0 to max_seq_len-1
- Look up position embeddings during forward pass
- Add to token embeddings like sinusoidal
- Trained end-to-end with model

**Sinusoidal vs Learned: Trade-offs**

| Aspect | Sinusoidal | Learned |
|--------|-----------|---------|
| Parameters | 0 | max_seq_len × d_model |
| Extrapolation | Yes (to any length) | No (limited to max_seq_len) |
| Task-specific | No | Yes (optimized per task) |
| Performance | Comparable | Slightly better on some tasks |
| Memory | Negligible | O(max_seq_len × d_model) |

### Toy Example Sequence

For all experiments, we'll use this simple test sequence:

```
"the cat sat on the mat"
```

Tokenized (word-level for simplicity): `["the", "cat", "sat", "on", "the", "mat"]`

This 6-token sequence is perfect for visualization and understanding position effects.

### Visualizations

**1. Position Encoding Heatmap**
- **Rows:** positions (0-5 for our toy sequence)
- **Columns:** embedding dimensions (0 to d_model)
- **Color:** encoding value
- Shows the wave-like pattern of sinusoidal encodings
- Compare with learned embeddings (random-looking initially, structured after training)

**2. 3D Position Space Projection**
- Project position encodings to 3D using PCA
- **X, Y, Z:** first 3 principal components
- **Points:** positions 0-20 (extended beyond our 6-token sequence)
- **Animation:** Rotate 360° to show 3D structure
- Sinusoidal: smooth spiral/helix pattern
- Learned: scattered points (before training) → structured (after training)

**3. Position Similarity Matrix**
- Pairwise cosine similarity between position encodings
- Heatmap showing which positions are "similar"
- Sinusoidal: diagonal pattern (nearby positions are similar)
- Learned: task-dependent structure

**4. Frequency Spectrum**
- For sinusoidal only
- Show the different frequency components across dimensions
- Low freq → high freq progression
- Helps visualize the "unique fingerprint" property

**5. Combined Token + Position Embeddings**
- Take our toy sequence: "the cat sat on the mat"
- Project (token_emb + pos_emb) to 3D
- Show how position "shifts" identical tokens ("the" at pos 0 vs pos 4)
- Side-by-side comparison: sinusoidal vs learned

---

## Phase 2: RoPE (Rotary Position Embeddings)

**Su et al., 2021 — Used in LLaMA, PaLM, GPT-NeoX**

### The Key Idea

Instead of adding position to embeddings, RoPE rotates query and key vectors based on their positions. This makes relative position information explicit in the dot product.

**Mathematical Foundation:**

For a 2D vector, rotation by angle θ:
```
[x']   [cos(θ)  -sin(θ)] [x]
[y'] = [sin(θ)   cos(θ)] [y]
```

RoPE extends this to high dimensions by applying rotations to pairs of dimensions.

**For position m, dimension pair (2i, 2i+1):**
```
θ_i = m / 10000^(2i/d)

[q_{2i}']     [cos(m·θ_i)  -sin(m·θ_i)] [q_{2i}]
[q_{2i+1}'] = [sin(m·θ_i)   cos(m·θ_i)] [q_{2i+1}]
```

Apply the same rotation to keys based on their position.

### Why Rotation Instead of Addition?

**Relative Position Information:**

When computing attention scores (dot product):
```
q_m^T k_n = (R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
```

The dot product depends only on relative position (n-m), not absolute positions!

**Benefits:**
- Encodes relative position naturally
- Works well for long sequences
- Interpolation: can extend to longer sequences than trained on
- No added parameters
- Better extrapolation than sinusoidal

### Implementation Details

**1. Precompute Rotation Matrices**
```python
def precompute_freqs(dim, max_seq_len, base=10000):
    # Compute θ_i for each dimension pair
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # Compute m·θ_i for each position m
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # (seq_len, dim/2)
    return freqs
```

**2. Apply Rotation**
```python
def apply_rotary_emb(x, freqs):
    # x: (batch, seq_len, n_heads, head_dim)
    # freqs: (seq_len, head_dim/2)

    # Reshape to pairs
    x_complex = torch.view_as_complex(x.reshape(..., -1, 2))
    # Apply rotation (complex multiplication)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    x_rotated = x_complex * freqs_complex
    # Reshape back
    return torch.view_as_real(x_rotated).flatten(-2)
```

**3. Use in Attention**
```python
# Apply RoPE to queries and keys
Q_rotated = apply_rotary_emb(Q, freqs)
K_rotated = apply_rotary_emb(K, freqs)

# Standard attention with rotated Q, K
scores = Q_rotated @ K_rotated.transpose(-2, -1)
```

### Visualizations

**1. Rotation Visualization (2D)**
- Take a 2D query vector [1, 0]
- Show how it rotates for positions 0, 1, 2, 3, 4, 5
- Animate the rotation in 2D space
- Different dimensions rotate at different speeds

**2. RoPE in 3D Space**
- Project rotated embeddings to 3D
- Show spiral/helical structure
- Compare to sinusoidal (different geometry)
- Animate rotation over positions

**3. Relative Position Matrix**
- Compute q_m^T k_n for all position pairs
- Heatmap showing attention scores depend on (m-n)
- Should see diagonal patterns
- Demonstrates relative position encoding

**4. Frequency Bands**
- Visualize different rotation rates across dimensions
- Low dimensions: slow rotation (coarse position)
- High dimensions: fast rotation (fine position)
- Similar to sinusoidal frequency decomposition

**5. Interpolation Test**
- Train on sequences of length 6
- Test on sequences of length 12
- Show that RoPE generalizes well
- Compare with learned embeddings (fails)

---

## Phase 3: ALiBi (Attention with Linear Biases)

**Press et al., 2021 — Used in BLOOM, MPT**

### The Radical Idea

**Don't use positional embeddings at all.** Instead, add position-dependent biases directly to attention scores.

**Attention without ALiBi:**
```
scores = QK^T / √d_k
attention = softmax(scores)
```

**Attention with ALiBi:**
```
scores = QK^T / √d_k - m · |i - j|
attention = softmax(scores)
```

Where:
- `i, j` = query and key positions
- `|i - j|` = distance between positions
- `m` = head-specific slope (penalty strength)

### Why This Works

**Intuition:**
- Penalize attention to distant tokens
- Nearby tokens have higher attention scores
- No parameters to learn
- Extremely simple

**Head-specific slopes:**
- Different attention heads get different slopes: m = 2^(-8/n) for head n
- Some heads focus on local context (steep slope)
- Other heads consider broader context (gentle slope)

**Advantages:**
- Zero positional parameters
- Excellent extrapolation (tested up to 20x training length!)
- Memory efficient
- Faster training (no position embeddings to learn)

### Implementation Details

**1. Compute Bias Matrix**
```python
def get_alibi_slopes(n_heads):
    # Head-specific slopes
    slopes = 2 ** (-8 / n_heads * torch.arange(1, n_heads + 1))
    return slopes

def get_alibi_bias(seq_len, n_heads):
    # Distance matrix |i - j|
    positions = torch.arange(seq_len).unsqueeze(0)
    distances = torch.abs(positions - positions.T)  # (seq_len, seq_len)

    # Apply slopes
    slopes = get_alibi_slopes(n_heads).view(-1, 1, 1)  # (n_heads, 1, 1)
    bias = -slopes * distances.unsqueeze(0)  # (n_heads, seq_len, seq_len)
    return bias
```

**2. Apply in Attention**
```python
# Compute standard attention scores
scores = Q @ K.T / sqrt(d_k)  # (batch, n_heads, seq_len, seq_len)

# Add ALiBi bias
bias = get_alibi_bias(seq_len, n_heads)
scores = scores + bias

# Standard softmax and output
attention = softmax(scores)
output = attention @ V
```

**No changes to embeddings:**
```python
x = token_embeddings  # No positional embeddings added!
```

### Visualizations

**1. ALiBi Bias Heatmap**
- Show bias matrix for one head
- **Rows/Cols:** positions
- **Color:** bias value (negative)
- Diagonal = 0 (no penalty for same position)
- Values decrease linearly with distance
- Show multiple heads with different slopes

**2. Attention Pattern Comparison**
- 3×3 grid: rows = methods (no position, sinusoidal+position, ALiBi)
- Columns: different query positions
- Show attention distribution
- ALiBi: smooth decay with distance
- No position: uniform/chaotic
- Sinusoidal: task-dependent

**3. Slope Comparison**
- For 8 attention heads
- Plot bias curves for each head
- X-axis: distance |i-j|
- Y-axis: bias value
- Different heads = different slopes
- Steeper = more local focus

**4. Extrapolation Test**
- Train on 6-token sequences
- Test on 12, 24, 48 tokens
- Measure perplexity or accuracy
- Compare: ALiBi vs sinusoidal vs learned vs RoPE
- ALiBi should excel

**5. 3D Attention Landscape**
- X-axis: query position
- Y-axis: key position
- Z-axis: attention weight (after softmax)
- For one head with ALiBi
- Should see ridge along diagonal
- Show how different slopes change the landscape

---

## Phase 4: Ablation Study — No Position

### The Experiment

**Remove all positional information** and observe what happens.

**Setup:**
```python
# NO positional embeddings
x = token_embeddings  # Just token embeddings

# NO ALiBi biases
scores = Q @ K.T / sqrt(d_k)  # Standard attention, no bias

# Process through transformer
output = transformer(x)
```

### What Breaks?

**1. Position-dependent tasks fail:**
- Sequence ordering
- Next-token prediction
- Positional reasoning

**2. Attention becomes position-agnostic:**
- Attention weights independent of position
- "the" at position 0 gets same attention as "the" at position 4
- Model can't distinguish "cat sat on mat" from "mat sat on cat"

**3. Permutation invariance:**
- Shuffling input tokens produces identical representations
- No notion of "before" or "after"

### Test Cases

**Simple Sequence Prediction:**
```
Input: "A B C D E"
Task: Predict next token

With position: F (correct)
Without position: Random/uniform (fails)
```

**Position-based QA:**
```
Input: "John gave Mary a book. She thanked him."
Question: "Who thanked whom?"

With position: "She" (Mary) thanked "him" (John) — position resolves pronouns
Without position: Cannot distinguish — both "she" and "him" have same representation
```

**Word Order:**
```
Sentence 1: "The dog bit the man"
Sentence 2: "The man bit the dog"

With position: Different meanings
Without position: Identical representations!
```

### Visualizations

**1. Attention Heatmap Comparison**
- Grid: 2 rows (with position, without position) × 3 columns (different methods)
- For toy sequence "the cat sat on the mat"
- **Without position:** Attention is chaotic/uniform, no structure
- **With position:** Clear patterns emerge

**2. Embedding Space Collapse**
- t-SNE of token representations
- **With position:** Multiple "the" tokens at different positions are separated
- **Without position:** All "the" tokens collapse to same point
- Position breaks degeneracy

**3. Prediction Accuracy**
- Bar chart: next-token prediction accuracy
- Categories: No position, Sinusoidal, Learned, RoPE, ALiBi
- No position: ~random baseline
- Others: significant improvement

**4. Permutation Test**
- Generate random permutations of "the cat sat on the mat"
- Compute output representations
- **Without position:** All permutations → identical output
- **With position:** Different outputs for different orders
- Show variance across permutations

**5. Distance vs Attention**
- Scatter plot: position distance |i-j| vs attention weight
- **Without position:** No correlation (flat)
- **With position (Sinusoidal/RoPE):** Weak structure
- **With ALiBi:** Strong negative correlation (by design)

---

## Comparative Analysis

### Summary Table

| Method | Parameters | Extrapolation | Relative Position | Memory | Complexity |
|--------|-----------|---------------|-------------------|--------|-----------|
| **None** | 0 | N/A | No | 0 | None |
| **Sinusoidal** | 0 | Excellent | Weak | 0 | Low |
| **Learned** | L×d | Poor | No | O(L×d) | Low |
| **RoPE** | 0 | Good | **Strong** | 0 | Medium |
| **ALiBi** | 0 | **Excellent** | Strong | 0 | Low |

Where L = max sequence length, d = embedding dimension

### When to Use Each

**Sinusoidal:**
- Simple baseline
- Research/prototyping
- When extrapolation to any length needed
- Parameter efficiency critical

**Learned:**
- Fixed-length sequences (e.g., BERT classification)
- When model can memorize position patterns
- Historical models (GPT-2/3, BERT)

**RoPE:**
- Long sequences
- When relative position matters most
- State-of-the-art models (LLaMA, Mistral)
- Good balance of all properties

**ALiBi:**
- Extreme length generalization needed
- Training short, deploying long
- Memory-constrained environments
- Very long contexts (e.g., BLOOM, MPT)

**None:**
- Set-based tasks (if genuinely position-invariant)
- Never for sequence modeling!

---

## Learning Outcomes

By completing this project, you will:
- Understand why transformers need positional information
- Implement four different positional encoding methods from scratch
- Visualize how position shapes the embedding space in 3D
- Compare absolute vs relative positional encoding
- See the catastrophic failure when position is removed
- Understand modern choices (RoPE in LLaMA, ALiBi in BLOOM)
- Gain intuition for extrapolation and interpolation properties

**Key Insights:**
- Transformers are inherently position-blind
- Position can be added (embeddings) or implicit (rotation/bias)
- Relative position > absolute position for many tasks
- Simpler methods (ALiBi) can outperform complex ones
- Visualization in 3D reveals geometric structure

---

## Project Structure

```
02_positional_embeddings/
├── phase1_classic/
│   ├── sinusoidal.py              # Sinusoidal implementation
│   ├── learned.py                 # Learned embeddings implementation
│   ├── visualize_encodings.py     # Heatmaps, 3D projections
│   └── compare_methods.py         # Side-by-side comparison
├── phase2_rope/
│   ├── rope.py                    # RoPE implementation
│   ├── visualize_rotation.py      # Rotation animations, 3D
│   └── interpolation_test.py      # Extrapolation experiments
├── phase3_alibi/
│   ├── alibi.py                   # ALiBi implementation
│   ├── visualize_biases.py        # Bias heatmaps, slopes
│   └── extrapolation_test.py      # Long sequence tests
├── phase4_ablation/
│   ├── no_position.py             # Transformer without position
│   ├── visualize_failure.py       # Show degradation
│   └── permutation_test.py        # Invariance demonstration
├── shared/
│   ├── attention.py               # Shared attention implementation
│   ├── toy_model.py               # Small transformer for testing
│   └── utils.py                   # Common utilities
└── README.md

# Data and outputs stored in root-level folders:
# - ../../data/raw/02_positional_embeddings/
#   - toy_sequences.txt                  # Test sequences
# - ../../data/processed/02_positional_embeddings/
#   - position_encodings.pt              # Saved encodings
# - ../../visualizations/plots/02_positional_embeddings/
#   - phase1_sinusoidal_heatmap.png
#   - phase1_learned_heatmap.png
#   - phase1_3d_comparison.gif           # Animated 3D rotation
#   - phase2_rope_rotation.gif           # RoPE rotation animation
#   - phase2_rope_3d.png
#   - phase3_alibi_bias_heatmap.png
#   - phase3_slope_comparison.png
#   - phase4_attention_comparison.png
#   - phase4_permutation_variance.png
#   - summary_comparison.png             # All methods side-by-side
# - ../../models/checkpoints/02_positional_embeddings/
#   - learned_pos_embeddings.pt          # Trained learned embeddings
```

---

## Getting Started

### Recommended Approach

**Start with Phase 1:**
1. Implement sinusoidal encoding
2. Visualize the wave patterns
3. Implement learned embeddings
4. Compare both on toy sequence

**Progress to Phase 2:**
1. Understand rotation mathematics
2. Implement RoPE
3. Visualize rotations in 2D/3D
4. Test extrapolation

**Then Phase 3:**
1. Implement ALiBi (simplest!)
2. Visualize bias matrices
3. Test extreme extrapolation
4. Compare with RoPE

**Finally Phase 4:**
1. Remove all position info
2. Watch everything break
3. Appreciate why position matters

### Toy Sequence

All phases use: **"the cat sat on the mat"**

- 6 tokens (manageable for visualization)
- Contains repeated word ("the")
- Clear semantic meaning
- Easy to understand position effects

### Model Parameters

**For learning and visualization:**
- Embedding dimension: d_model = 64 (small enough to visualize)
- Number of heads: 4
- Max sequence length: 12 (test extrapolation from 6→12)
- Vocabulary: ~100 words (simple corpus)

**Training (Phase 1 learned embeddings only):**
- Simple next-token prediction
- Small corpus (few thousand sentences)
- 5-10 epochs
- Expected time: 5-10 minutes

---

## Mathematical Foundations

### Why Do We Need Position?

**Self-Attention is a Set Function:**

```
Attention(X) = softmax(QK^T/√d) V
```

Where Q, K, V are linear projections of input X.

For any permutation P:
```
Attention(P(X)) = P(Attention(X))
```

The operation is equivariant to permutations—order doesn't matter!

### Position Injection Methods

**Method 1: Additive (Sinusoidal, Learned)**
```
X_pos = X_token + X_position
```

Pros: Simple, direct
Cons: Loses some token information through addition

**Method 2: Rotational (RoPE)**
```
Q_pos = R(pos_q) · Q
K_pos = R(pos_k) · K
```

Pros: Preserves magnitude, encodes relative position
Cons: More complex implementation

**Method 3: Attention Bias (ALiBi)**
```
Attention = softmax(QK^T/√d - m|i-j|) V
```

Pros: No embeddings needed, excellent extrapolation
Cons: Linear bias may be too simple for some tasks

---

## References

**Original Papers:**

1. **Sinusoidal:** Vaswani et al. (2017). "Attention Is All You Need"
2. **RoPE:** Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding"
3. **ALiBi:** Press et al. (2021). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"

**Key Implementations:**

- LLaMA (RoPE): Meta AI
- BLOOM (ALiBi): BigScience
- GPT-3 (Learned): OpenAI
- Original Transformer (Sinusoidal): Google

---

## Experiments to Try

### 1. Length Generalization
- Train on sequences of length 6
- Test on 12, 24, 48, 96
- Measure perplexity/accuracy degradation
- Compare all methods

### 2. Position Resolution
- Create task requiring fine-grained position (e.g., "predict position of token X")
- Test which method has best resolution
- Hypothesis: RoPE > Sinusoidal > ALiBi for fine position

### 3. Memory Efficiency
- Measure memory usage for max_len = 512, 2048, 8192
- Learned embeddings scale linearly
- Others constant

### 4. Attention Pattern Analysis
- Extract attention weights from each method
- Cluster patterns
- Hypothesis: ALiBi forces local attention, others more flexible

### 5. Hybrid Methods
- Combine RoPE + ALiBi
- Sinusoidal + ALiBi
- Test if combination improves extrapolation

---

## Quick Start Guide

### Prerequisites

```bash
# From project root, ensure dependencies are installed
uv sync
```

### Option 1: Run Everything (Recommended for First Time)

```bash
cd chapters/02_positional_embeddings
./run_all.sh
```

This runs all phases sequentially (~15-20 minutes total).

### Option 2: Run Individual Phases

#### Test First

```bash
uv run python test_all.py
```

#### Phase 1: Classic Positional Embeddings

```bash
# Sinusoidal (fixed, no training)
uv run python phase1_classic/sinusoidal.py

# Learned (requires training ~2-3 minutes)
uv run python phase1_classic/learned.py --epochs 5

# Visualizations
uv run python phase1_classic/visualize_encodings.py

# Comparison
uv run python phase1_classic/compare_methods.py
```

**Outputs:**
- `data/processed/02_positional_embeddings/sinusoidal_encodings.pt`
- `data/processed/02_positional_embeddings/learned_embeddings.pt`
- `visualizations/plots/02_positional_embeddings/phase1/*.png`
- `visualizations/animations/02_positional_embeddings/phase1/*.gif`

#### Phase 2: RoPE (Rotary Position Embeddings)

```bash
# Implementation demo
uv run python phase2_rope/rope.py --test-attention

# Visualizations (rotations, 3D projections)
uv run python phase2_rope/visualize_rotation.py

# Extrapolation test (train on length 6, test on 12, 24, 48)
uv run python phase2_rope/interpolation_test.py
```

**Outputs:**
- `data/processed/02_positional_embeddings/rope_frequencies.pt`
- `visualizations/plots/02_positional_embeddings/phase2/*.png`
- `visualizations/animations/02_positional_embeddings/phase2/*.gif`

#### Phase 3: ALiBi (Attention with Linear Biases)

```bash
# Implementation demo
uv run python phase3_alibi/alibi.py --test-attention

# Visualizations (bias heatmaps, slopes)
uv run python phase3_alibi/visualize_biases.py

# Extreme extrapolation (train on 6, test up to 96!)
uv run python phase3_alibi/extrapolation_test.py
```

**Outputs:**
- `data/processed/02_positional_embeddings/alibi_biases.pt`
- `visualizations/plots/02_positional_embeddings/phase3/*.png`

#### Phase 4: Ablation Study (No Position)

```bash
# Demonstrate failure without position
uv run python phase4_ablation/no_position.py

# Optional: compare with vs without position (slower, trains models)
uv run python phase4_ablation/no_position.py --compare

# Visualize attention degradation
uv run python phase4_ablation/visualize_failure.py

# Permutation invariance tests
uv run python phase4_ablation/permutation_test.py

# Optional: test ALL permutations of small sequence
uv run python phase4_ablation/permutation_test.py --all-perms
```

**Outputs:**
- `data/processed/02_positional_embeddings/no_position_results.pt`
- `visualizations/plots/02_positional_embeddings/phase4/*.png`

---

## Key Visualizations to Check

### Phase 1
- `phase1/sinusoidal_heatmap.png` - Wave patterns
- `phase1/sinusoidal_3d_rotation.gif` - 3D animation
- `phase1/comparison_heatmaps.png` - Side-by-side comparison

### Phase 2
- `phase2/rope_2d_rotation.png` - 2D rotation visualization
- `phase2/rope_2d_animation.gif` - Rotation animation
- `phase2/rope_extrapolation_comparison.png` - Length generalization

### Phase 3
- `phase3/alibi_bias_heatmaps.png` - Bias matrices
- `phase3/alibi_slope_comparison.png` - Different head slopes
- `phase3/alibi_extreme_extrapolation.png` - Exceptional extrapolation

### Phase 4
- `phase4/no_position_attention_comparison.png` - Attention collapse
- `phase4/no_position_embedding_collapse.png` - Embedding degeneracy
- `phase4/permutation_heatmap.png` - Permutation invariance

---

## Expected Runtime

| Phase | Script | Time (MPS) | Time (CPU) |
|-------|--------|------------|------------|
| Test | `test_all.py` | ~30 sec | ~1 min |
| Phase 1 | Sinusoidal | ~10 sec | ~10 sec |
| Phase 1 | Learned training | ~2-3 min | ~8-10 min |
| Phase 1 | Visualizations | ~30 sec | ~30 sec |
| Phase 2 | RoPE demo | ~10 sec | ~10 sec |
| Phase 2 | Visualizations | ~30 sec | ~30 sec |
| Phase 2 | Interpolation | ~3-5 min | ~10-15 min |
| Phase 3 | ALiBi demo | ~10 sec | ~10 sec |
| Phase 3 | Visualizations | ~30 sec | ~30 sec |
| Phase 3 | Extrapolation | ~3-5 min | ~10-15 min |
| Phase 4 | No position | ~10 sec | ~10 sec |
| Phase 4 | Visualizations | ~30 sec | ~30 sec |
| **Total** | **All phases** | **~15-20 min** | **~45-60 min** |

*Times are approximate and depend on hardware.*

---

## Troubleshooting

### Import Errors

Make sure you're running from the correct directory:

```bash
# From project root
cd chapters/02_positional_embeddings
uv run python <script>.py
```

### Out of Memory

If you get OOM errors, reduce batch sizes in training scripts:

```bash
# Phase 1 learned
uv run python phase1_classic/learned.py --batch-size 16  # default: 32

# Phase 2/3 extrapolation tests use smaller batches by default
```

### Slow on CPU

All scripts use MPS (Apple Silicon) if available, otherwise CUDA, otherwise CPU.

To speed up on CPU:
- Reduce number of epochs
- Use smaller models (already quite small)
- Run individual phases instead of `run_all.sh`

### Missing Visualizations

Make sure you run the implementation scripts before visualization scripts:

```bash
# Example: Phase 1
uv run python phase1_classic/sinusoidal.py    # Must run first
uv run python phase1_classic/learned.py       # Must run first
uv run python phase1_classic/visualize_encodings.py  # Then this
```

---

**Start with Phase 1 and build your intuition step by step. Position is the invisible scaffolding that makes transformers work!**