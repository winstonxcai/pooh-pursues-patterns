#!/bin/bash

# run_all.sh - Run all phases of Chapter 3: Self-Attention & Multi-Head Attention
#
# This script runs all implementations and visualizations in sequence.
# Estimated time: 5-10 minutes total

set -e  # Exit on error

echo "================================================================================"
echo "           CHAPTER 3: SELF-ATTENTION & MULTI-HEAD ATTENTION"
echo "                        Running All Phases"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track start time
START_TIME=$(date +%s)

echo -e "${BLUE}Step 0: Running Tests${NC}"
echo "--------------------------------------------------------------------------------"
uv run python test_all.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Tests passed!${NC}"
else
    echo -e "${YELLOW}✗ Tests failed! Continuing anyway...${NC}"
fi
echo ""

echo -e "${BLUE}Phase 1: Single-Head Self-Attention${NC}"
echo "--------------------------------------------------------------------------------"
echo "Running implementation..."
uv run python phase1_single_head/self_attention.py
echo ""
echo "Creating visualizations..."
uv run python phase1_single_head/visualize.py
echo -e "${GREEN}✓ Phase 1 complete!${NC}"
echo ""

echo -e "${BLUE}Phase 2: Multi-Head Attention${NC}"
echo "--------------------------------------------------------------------------------"
echo "Running implementation..."
uv run python phase2_multi_head/multi_head_attention.py
echo ""
echo "Creating visualizations..."
uv run python phase2_multi_head/visualize.py
echo -e "${GREEN}✓ Phase 2 complete!${NC}"
echo ""

echo -e "${BLUE}Phase 3: Advanced Visualizations${NC}"
echo "--------------------------------------------------------------------------------"
echo "Creating attention flow diagrams..."
uv run python phase3_visualization/attention_flow.py
echo ""
echo "Creating 3D attention landscapes..."
echo "(This may take a minute for the animation...)"
uv run python phase3_visualization/attention_3d.py
echo -e "${GREEN}✓ Phase 3 complete!${NC}"
echo ""

echo -e "${BLUE}Phase 4: Causal Masking${NC}"
echo "--------------------------------------------------------------------------------"
echo "Running causal attention..."
uv run python phase4_causal/causal_attention.py
echo ""
echo "Creating visualizations..."
uv run python phase4_causal/visualize.py
echo -e "${GREEN}✓ Phase 4 complete!${NC}"
echo ""

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "================================================================================"
echo -e "${GREEN}🎉 ALL PHASES COMPLETE! 🎉${NC}"
echo "================================================================================"
echo ""
echo "Time elapsed: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Output files created:"
echo "--------------------------------------------------------------------------------"
echo ""
echo "Phase 1 (Single-Head Attention):"
echo "  • visualizations/plots/03_attention/phase1/attention_heatmap.png"
echo "  • visualizations/plots/03_attention/phase1/qkv_flow.png"
echo ""
echo "Phase 2 (Multi-Head Attention):"
echo "  • visualizations/plots/03_attention/phase2/multi_head_grid.png"
echo "  • visualizations/plots/03_attention/phase2/attention_entropy.png"
echo ""
echo "Phase 3 (Advanced Visualizations):"
echo "  • visualizations/plots/03_attention/phase3/attention_flow.png"
echo "  • visualizations/plots/03_attention/phase3/multihead_flow.png"
echo "  • visualizations/plots/03_attention/phase3/3d_attention_surface.png"
echo "  • visualizations/plots/03_attention/phase3/multihead_3d_surfaces.png"
echo "  • visualizations/animations/03_attention/phase3/3d_attention_rotation.gif"
echo ""
echo "Phase 4 (Causal Masking):"
echo "  • visualizations/plots/03_attention/phase4/causal_mask_structure.png"
echo "  • visualizations/plots/03_attention/phase4/bidirectional_vs_causal.png"
echo "  • visualizations/plots/03_attention/phase4/multihead_causal.png"
echo ""
echo "Data files:"
echo "  • data/processed/03_attention/phase1_attention_weights.pt"
echo "  • data/processed/03_attention/phase2_multihead_attention.pt"
echo "  • data/processed/03_attention/phase4_causal_comparison.pt"
echo ""
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Review all visualizations in the plots directory"
echo "  2. Watch the rotating 3D animation (3d_attention_rotation.gif)"
echo "  3. Compare single-head vs multi-head attention patterns"
echo "  4. Study the bidirectional vs causal attention comparison"
echo "  5. Move to Chapter 4: Transformers, QKV, and Stacking"
echo ""
echo "================================================================================"
