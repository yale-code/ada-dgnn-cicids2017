#!/bin/bash
# ADA-DGNN V2 SOTA Experiment Entry Point
# =========================================

set -e

echo "============================================================"
echo "ADA-DGNN V2: State-of-the-Art NIDS Optimization"
echo "============================================================"

# Check Python and dependencies
echo ""
echo "Checking environment..."
python3 --version

# Install dependencies if needed
echo ""
echo "Installing dependencies..."
pip install -q torch numpy pandas scikit-learn matplotlib seaborn optuna

# Create output directories
mkdir -p experiments
mkdir -p datasets

echo ""
echo "Environment ready!"

# Run options
if [ "$1" == "full" ]; then
    echo ""
    echo "Running full optimized model..."
    python3 ada_dgnn_v2_sota.py
elif [ "$1" == "experiments" ]; then
    echo ""
    echo "Running comprehensive experiments..."
    python3 run_experiments.py
elif [ "$1" == "optuna" ]; then
    echo ""
    echo "Running hyperparameter optimization..."
    python3 optuna_search.py --trials 30
elif [ "$1" == "test" ]; then
    echo ""
    echo "Running module tests..."
    python3 kan_layer.py
    python3 multiscale_gnn.py
    python3 data_augmentation.py
    python3 training_strategies.py
    echo ""
    echo "All tests passed!"
else
    echo ""
    echo "Usage: ./entrypoint.sh [full|experiments|optuna|test]"
    echo ""
    echo "Commands:"
    echo "  full         - Run full optimized model (ada_dgnn_v2_sota.py)"
    echo "  experiments  - Run ablation studies and comparisons"
    echo "  optuna       - Run hyperparameter optimization"
    echo "  test         - Run module tests"
    echo ""
    echo "Example: ./entrypoint.sh full"
    exit 0
fi

echo ""
echo "============================================================"
echo "Experiment completed!"
echo "============================================================"
