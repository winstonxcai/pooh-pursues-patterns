# Pooh Pursues Patterns

**Pooh Pursues Patterns** is a hands-on, from-scratch exploration of the inner mechanics of modern language models.
Rather than treating large models as black boxes, this project pulls them apart into understandable pieces, rebuilds them, and studies how they behave — mathematically, visually, and empirically.

This is not a library for production use.
It is a living workbook for developing intuition about how language, structure, memory, and meaning emerge inside neural networks.

Each section is organized as a **chapter**, with experiments, visualizations, and explanations that build on each other.

---

## Chapter 1 — Tokenization & Embeddings

This chapter explores how raw text is converted into model-understandable units.

You will build your own Byte Pair Encoder (BPE) from scratch and train a subword vocabulary on a small corpus. A custom “token visualizer” maps words and sub-word chunks to token IDs so you can see exactly how text is split.

You will then compare one-hot vectors to learned embeddings and plot cosine distances between tokens to observe how semantic relationships form in vector space.

Topics include:

* Byte Pair Encoding (BPE)
* Subword tokenization
* One-hot vs learned embeddings
* Cosine similarity and geometry of meaning

---

## Chapter 2 — Positional Embeddings

Transformers have no built-in concept of order. This chapter shows how position is introduced.

You will implement and compare classic sinusoidal embeddings, learned positional embeddings, RoPE (Rotary Position Embeddings), and ALiBi. A toy sequence will be animated as it is mapped into 3D space so you can literally watch position shaping representation.

Finally, you will remove positional embeddings altogether and observe how badly attention degrades without them.

Topics include:

* Sinusoidal vs learned embeddings
* Rotary (RoPE) embeddings
* ALiBi bias
* Effects of removing positional information

---

## Chapter 3 — Self-Attention & Multi-Head Attention

This chapter builds attention from the ground up.

You will hand-implement dot-product self-attention for a single token, then expand it to a full sequence. From there, you will extend the design to multi-head attention, generating heatmaps of attention weights for each head.

You will then apply causal masking and verify that the model can only attend to the past, never the future.

Topics include:

* Scaled dot-product attention
* Multi-head splitting and concatenation
* Attention visualization
* Causal masking

---

## Chapter 4 — Transformers, QKV, and Stacking

Now the pieces come together.

You will stack the attention mechanism with LayerNorm, residual connections, and a feed-forward network to build a full transformer block. From there, you will create an n-layer “mini-former” and train it on toy data.

To deepen understanding, you will intentionally swap, corrupt, and remove the Q, K, and V projections to observe how each one affects system behavior.

Topics include:

* Transformer block architecture
* Residual connections
* Layer normalization
* Roles of Q, K, and V

---

## Chapter 5 — Sampling and Decoding

This chapter focuses on how models generate text.

You will build an interactive sampling dashboard to control temperature, top-k, and top-p (nucleus sampling) in real time. You will plot entropy versus diversity as parameters change, and observe how setting temperature to zero collapses the generation into repetition.

Topics include:

* Temperature scaling
* Top-k sampling
* Top-p (nucleus) sampling
* Entropy and randomness

---

## Chapter 6 — Key-Value Cache and Fast Inference

During inference, recomputing attention for the entire history is expensive. This chapter implements KV caching to speed up generation.

You will reuse key and value tensors from previous steps and measure the speedup compared to non-cached decoding. A cache hit/miss visualizer will show how memory is reused over time, and memory cost will be profiled for varying context lengths.

Topics include:

* Autoregressive decoding
* KV caching
* Latency vs memory tradeoffs

---

## Chapter 7 — Long-Context Techniques

This chapter explores how models handle long sequences.

You will implement sliding-window attention and compare it to full attention on long documents. Memory-efficient variants such as recomputation and Flash-style strategies will be benchmarked.

Perplexity will be plotted against context length to identify where performance collapses.

Topics include:

* Sliding window attention
* Long-range dependency loss
* Memory-efficient methods
* Context collapse analysis

---

## Chapter 8 — Mixture of Experts (MoE)

Here, you introduce specialization and sparsity.

You will build a simple 2-expert routing layer that dynamically assigns tokens to different experts. You will track expert utilization across a dataset and visualize load imbalance. Dense vs sparse routing will be compared to measure actual FLOP savings.

Topics include:

* Expert routing
* Sparse vs dense computation
* Load balancing
* Efficiency vs accuracy

---

## Chapter 9 — Grouped Query Attention

This chapter investigates a computational shortcut used in modern architectures.

You will convert your mini-transformer to use grouped query attention and measure its speed and memory against standard multi-head attention. You will vary the number of groups and plot the resulting latency and performance.

Topics include:

* Grouped queries
* Memory and speed benchmarking
* Trade-offs between efficiency and fidelity

---

## Chapter 10 — Normalization and Activations

You will recreate normalization and activation functions from scratch.

LayerNorm, RMSNorm, GELU, and SwiGLU will be implemented manually and substituted into identical models. You will track how training and validation loss change and visualize activation distributions across layers.

Topics include:

* LayerNorm vs RMSNorm
* Activation functions
* Gradient flow analysis
* Stability effects

---

## Chapter 11 — Training Objectives

This chapter studies different learning objectives.

You will train small models using masked language modeling, causal language modeling, and prefix language modeling. Loss curves will be compared and generated text will be analyzed to see how each objective shapes behavior.

Topics include:

* Masked LM (BERT-style)
* Causal LM (GPT-style)
* Prefix LM
* Behavioral differences

---

## Chapter 12 — Fine-Tuning, Instruction Tuning, and RLHF

This chapter explores alignment and task specialization.

You will fine-tune a base model on a small domain dataset. Then, you will instruction-tune it using task prefixes. Finally, you will build a tiny reward model and run a minimal PPO loop to perform a simplified RLHF process.

Topics include:

* Supervised fine-tuning
* Prompt-based instruction tuning
* Reward models
* PPO and reinforcement learning

---

## Chapter 13 — Scaling Laws and Capacity

This chapter examines how size changes everything.

You will train tiny, small, and medium models on the same dataset and plot performance versus parameter count. Wall-clock time, memory usage, and throughput will be measured. From this, you will extrapolate and explore the limits of “how small is too small.”

Topics include:

* Scaling curves
* Compute vs performance
* Diminishing returns

---

## Chapter 14 — Quantization

Here, you compress what you have built.

You will apply post-training quantization (PTQ) and quantization-aware training (QAT), export to formats like GGUF or AWQ, and measure the accuracy and speed trade-offs.

Topics include:

* 8-bit / 4-bit quantization
* PTQ vs QAT
* Accuracy loss analysis

---

## Chapter 15 — Inference and Training Stacks

The final chapter focuses on deployment and systems.

You will port a model between HuggingFace, DeepSpeed, vLLM, and ExLlama. Throughput, VRAM usage, and latency will be benchmarked across stacks to understand real-world performance differences.

Topics include:

* Framework comparison
* Hardware efficiency
* Production constraints

---

## Project Goal

Pooh Pursues Patterns exists to answer one central question:

**How do raw tokens eventually become structure, meaning, memory, and reasoning?**

By the end of this project, you will not only know how transformers work —
you will understand why they behave the way they do.

This is a journey toward intuition, not just implementation.

---

## Setup & Environment (using uv)

This project uses the **uv** package manager for fast, reproducible Python environments and dependency management.

Prerequisites:

* Python 3.10+
* uv installed (`pip install uv` or via official installer)

Add core dependencies:

```
uv add torch transformers datasets numpy matplotlib seaborn scikit-learn einops jupyterlab sentencepiece tokenizers tqdm
```

Optional, for performance experiments:

```
uv add flash-attn vllm deepspeed accelerate
```

Sync and install all dependencies:

```
uv sync
```

This will automatically create a virtual environment, install all dependencies from `pyproject.toml`, and generate a `uv.lock` file for reproducible builds.

Suggested hardware:

* CPU: Any modern multi-core
* GPU: NVIDIA with 12GB+ VRAM (for Chapters 8+)
* RAM: 16–32GB recommended

All chapters share a single uv environment managed at the project root.

---

## Folder Structure

```
pooh-pursues-patterns/
│
├── chapters/
│   ├── 01_tokenization/
│   ├── 02_positional_embeddings/
│   ├── 03_attention/
│   ├── 04_transformer_block/
│   ├── 05_sampling/
│   ├── 06_kv_cache/
│   ├── 07_long_context/
│   ├── 08_moe/
│   ├── 09_grouped_query/
│   ├── 10_norms_activations/
│   ├── 11_objectives/
│   ├── 12_alignment/
│   ├── 13_scaling/
│   ├── 14_quantization/
│   └── 15_stacks/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── checkpoints/
│   └── exports/
│
├── visualizations/
│   ├── plots/
│   └── animations/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

Each chapter folder contains:

* `README.md` – conceptual walkthrough
* `main.py` – experiment runner
* `visualize.py` – plots / animations
* `notes.md` – observations + takeaways

---

## Reading List & Paper References

Tokenization & Embeddings

* Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*
* Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*

Positional Encoding

* Vaswani et al., *Attention Is All You Need*
* Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*
* Press et al., *Train Short, Test Long: Attention with Linear Biases (ALiBi)*

Attention & Transformers

* Vaswani et al., *Attention Is All You Need*
* Kaplan et al., *Scaling Laws for Neural Language Models*

Mixture of Experts

* Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer*
* Fedus et al., *Switch Transformers*

Long-Context & Memory

* Beltagy et al., *Longformer*
* Dao et al., *FlashAttention*

Alignment & RLHF

* Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*
* Schulman et al., *Proximal Policy Optimization Algorithms*

Systems & Inference

* Rozière et al., *vLLM: Easy, Fast, Cheap LLM Serving*
* Rasley et al., *DeepSpeed*

Further Reading

* Elhage et al., *A Mathematical Framework for Transformer Circuits*
* Olah et al., *An Overview of Neural Network Interpretability*
