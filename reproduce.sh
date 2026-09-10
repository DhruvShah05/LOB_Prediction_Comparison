#!/usr/bin/env bash
# reproduce.sh — One-command reproduction of all tables and figures (Change 9).
#
# Usage:
#   bash reproduce.sh           # Run full pipeline
#   bash reproduce.sh --stage train --market crypto  # Run specific stage
#
# Prerequisites:
#   pip install -r requirements.txt
#   Place bitcoin_lob_data.csv in the project root (or it will be auto-downloaded)
#   FI-2010 BenchmarkDatasets/ in the project root (or auto-downloaded)

set -euo pipefail

echo "============================================================"
echo " LOB Prediction Comparison — Full Reproduction"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "============================================================"

# Check Python version
python3 --version

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import xgboost; print(f'XGBoost {xgboost.__version__}')"
python3 -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
python3 -c "import optuna; print(f'Optuna {optuna.__version__}')"

echo ""
echo "=== Stage 1: Prepare data ==="
python3 run_all.py --stage prepare "$@"

echo ""
echo "=== Stage 2: Threshold/horizon sweep ==="
python3 run_all.py --stage sweep "$@"

echo ""
echo "=== Stage 3: Hyperparameter tuning ==="
python3 run_all.py --stage tune "$@"

echo ""
echo "=== Stage 4: Train all models × 5 seeds ==="
python3 run_all.py --stage train "$@"

echo ""
echo "=== Stage 5: Ablation studies ==="
python3 run_all.py --stage ablate "$@"

echo ""
echo "=== Stage 6: Backtest ==="
python3 run_all.py --stage backtest "$@"

echo ""
echo "=== Stage 7: Regime analysis ==="
python3 run_all.py --stage regime "$@"

echo ""
echo "=== Stage 8: Significance tests ==="
python3 run_all.py --stage significance "$@"

echo ""
echo "=== Stage 9: Aggregate → LaTeX tables ==="
python3 run_all.py --stage aggregate "$@"

echo ""
echo "=== Stage 10: Run test suite ==="
python3 -m pytest tests/ -v --tb=short

echo ""
echo "============================================================"
echo " Reproduction complete!"
echo " Results: results/"
echo " LaTeX tables: results/aggregated/results_table.tex"
echo "============================================================"
