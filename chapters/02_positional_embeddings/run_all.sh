#!/bin/bash

# Chapter 2: Positional Embeddings - Complete Pipeline
# Runs all phases in sequence with visualizations

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

echo "================================================================================"
echo "CHAPTER 2: POSITIONAL EMBEDDINGS - COMPLETE PIPELINE"
echo "================================================================================"
echo ""
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo ""

# Test first
echo "Step 0: Running tests..."
echo "--------------------------------------------------------------------------------"
uv run python test_all.py
echo ""

# Phase 1: Classic Methods
echo "================================================================================"
echo "PHASE 1: CLASSIC POSITIONAL EMBEDDINGS"
echo "================================================================================"
echo ""

echo "Step 1.1: Sinusoidal Embeddings..."
echo "--------------------------------------------------------------------------------"
uv run python phase1_classic/sinusoidal.py
echo ""

echo "Step 1.2: Learned Embeddings (Training)..."
echo "--------------------------------------------------------------------------------"
uv run python phase1_classic/learned.py --epochs 5
echo ""

echo "Step 1.3: Visualizations..."
echo "--------------------------------------------------------------------------------"
uv run python phase1_classic/visualize_encodings.py
echo ""

echo "Step 1.4: Comparison..."
echo "--------------------------------------------------------------------------------"
uv run python phase1_classic/compare_methods.py
echo ""

# Phase 2: RoPE
echo "================================================================================"
echo "PHASE 2: ROPE (ROTARY POSITION EMBEDDINGS)"
echo "================================================================================"
echo ""

echo "Step 2.1: RoPE Implementation..."
echo "--------------------------------------------------------------------------------"
uv run python phase2_rope/rope.py --test-attention
echo ""

echo "Step 2.2: Visualizations..."
echo "--------------------------------------------------------------------------------"
uv run python phase2_rope/visualize_rotation.py
echo ""

echo "Step 2.3: Interpolation/Extrapolation Tests..."
echo "--------------------------------------------------------------------------------"
uv run python phase2_rope/interpolation_test.py
echo ""

# Phase 3: ALiBi
echo "================================================================================"
echo "PHASE 3: ALIBI (ATTENTION WITH LINEAR BIASES)"
echo "================================================================================"
echo ""

echo "Step 3.1: ALiBi Implementation..."
echo "--------------------------------------------------------------------------------"
uv run python phase3_alibi/alibi.py --test-attention
echo ""

echo "Step 3.2: Visualizations..."
echo "--------------------------------------------------------------------------------"
uv run python phase3_alibi/visualize_biases.py
echo ""

echo "Step 3.3: Extreme Extrapolation Tests..."
echo "--------------------------------------------------------------------------------"
uv run python phase3_alibi/extrapolation_test.py
echo ""

# Phase 4: Ablation
echo "================================================================================"
echo "PHASE 4: ABLATION STUDY (NO POSITION)"
echo "================================================================================"
echo ""

echo "Step 4.1: No Position Implementation..."
echo "--------------------------------------------------------------------------------"
uv run python phase4_ablation/no_position.py
echo ""

echo "Step 4.2: Failure Visualizations..."
echo "--------------------------------------------------------------------------------"
uv run python phase4_ablation/visualize_failure.py
echo ""

echo "Step 4.3: Permutation Tests..."
echo "--------------------------------------------------------------------------------"
uv run python phase4_ablation/permutation_test.py
echo ""

# Done
echo "================================================================================"
echo "✅ COMPLETE! All phases finished successfully."
echo "================================================================================"
echo ""
echo "End time: $(date)"
echo ""
echo "Check the following directories for outputs:"
echo "  - ../../visualizations/plots/02_positional_embeddings/"
echo "  - ../../visualizations/animations/02_positional_embeddings/"
echo "  - ../../data/processed/02_positional_embeddings/"
echo ""
echo "Verifying outputs..."
echo ""

# Check data files
if [ -f "../../data/processed/02_positional_embeddings/sinusoidal_encodings.pt" ]; then
    echo "  ✓ Sinusoidal encodings saved"
else
    echo "  ✗ Sinusoidal encodings missing"
fi

if [ -f "../../data/processed/02_positional_embeddings/learned_embeddings.pt" ]; then
    echo "  ✓ Learned embeddings saved"
else
    echo "  ✗ Learned embeddings missing"
fi

if [ -f "../../data/processed/02_positional_embeddings/rope_frequencies.pt" ]; then
    echo "  ✓ RoPE frequencies saved"
else
    echo "  ✗ RoPE frequencies missing"
fi

if [ -f "../../data/processed/02_positional_embeddings/alibi_biases.pt" ]; then
    echo "  ✓ ALiBi biases saved"
else
    echo "  ✗ ALiBi biases missing"
fi

if [ -f "../../data/processed/02_positional_embeddings/no_position_results.pt" ]; then
    echo "  ✓ No position results saved"
else
    echo "  ✗ No position results missing"
fi

echo ""
echo "Pipeline complete! 🎉"
echo ""
