"""
run_all.py — Stage-based pipeline (Change 7).

Stages:
  prepare      → scripts/prepare_*.py, quality reports, feature validation
  sweep        → experiments/threshold_sweep.py (both markets) → choose headline labels
  tune         → train/tune.py for every (market, feature_set, label_rule, model)
  train        → main.py × 5 seeds for the full grid
  ablate       → experiments/ablation.py
  backtest     → experiments/backtest.py
  regime       → experiments/regime_split.py
  significance → experiments/run_significance.py
  aggregate    → eval/aggregate.py → LaTeX tables
  figures      → all paper figures from saved outputs

Usage:
  python run_all.py --stage train --market crypto
  python run_all.py --stage all --force
"""

import os
import sys
import glob
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STAGES = [
    'prepare', 'sweep', 'tune', 'train', 'ablate', 'backtest',
    'regime', 'significance', 'aggregate', 'figures'
]

TREE_MODELS = ['xgboost', 'random_forest', 'logistic']
NEURAL_MODELS = ['deeplob_full', 'transformer_windowed', 'level_transformer', 'deeplob_attention']
ALL_MODELS = TREE_MODELS + NEURAL_MODELS

MARKETS = ['crypto', 'fi2010']
FEATURE_SETS = ['raw40']  # raw40_eng added when time allows
LABEL_RULES = ['smoothed']  # point_return added as secondary


def run_cmd(cmd, description=""):
    """Run a command, log output."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Stage: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'=' * 60}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        logger.error(f"FAILED: {description} (exit code {result.returncode})")
        return False
    logger.info(f"COMPLETED: {description}")
    return True


def stage_prepare(markets, force=False):
    """Run data preparation scripts."""
    logger.info("=== STAGE: prepare ===")

    if 'crypto' in markets:
        out = 'data/processed/crypto_BTCUSDT.parquet'
        if force or not os.path.exists(out):
            run_cmd([sys.executable, 'scripts/prepare_crypto.py'], 'Prepare crypto')

    if 'fi2010' in markets:
        out = 'data/processed/fi2010_zscore_train.npy'
        if force or not os.path.exists(out):
            run_cmd([sys.executable, 'scripts/prepare_fi2010.py'], 'Prepare FI-2010')
            run_cmd([sys.executable, 'scripts/prepare_fi2010.py', '--verify-horizons'],
                    'Verify FI-2010 horizons')

    # Run tests
    run_cmd([sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            'Run test suite')


def stage_sweep(markets, force=False):
    """Run threshold/horizon sweep."""
    logger.info("=== STAGE: sweep ===")
    out_dir = 'results/threshold_sweep/'

    if force or not os.path.exists(os.path.join(out_dir, 'crypto_threshold_sweep.csv')):
        market_arg = 'both' if len(markets) > 1 else markets[0]
        run_cmd([sys.executable, 'experiments/threshold_sweep.py',
                 '--market', market_arg, '--output-dir', out_dir],
                'Threshold sweep')


def stage_tune(markets, force=False):
    """Run Optuna tuning for all models."""
    logger.info("=== STAGE: tune ===")

    for market in markets:
        for feature_set in FEATURE_SETS:
            for label_rule in LABEL_RULES:
                for model in ALL_MODELS:
                    out = f'configs/tuned/{market}_{feature_set}_{label_rule}_{model}.yaml'
                    if not force and os.path.exists(out):
                        logger.info(f"Skipping tune: {out} exists")
                        continue

                    run_cmd([
                        sys.executable, 'train/tune.py',
                        '--model', model,
                        '--market', market,
                        '--feature_set', feature_set,
                        '--label_rule', label_rule,
                    ], f'Tune {model} on {market}/{feature_set}/{label_rule}')


def stage_train(markets, force=False):
    """Train all models × 5 seeds."""
    logger.info("=== STAGE: train ===")

    config_pattern = 'configs/*.yaml'
    config_files = sorted(glob.glob(config_pattern))

    # Filter by market
    if markets != MARKETS:
        config_files = [c for c in config_files
                       if any(c.split('/')[-1].startswith(m) for m in markets)]

    # Exclude base.yaml
    config_files = [c for c in config_files if 'base.yaml' not in c]

    seeds = [0, 1, 2, 3, 4]

    for config_file in config_files:
        for seed in seeds:
            # Check if already done
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            out_dir = cfg.get('output_dir', 'results/default/')
            metrics_path = os.path.join(out_dir, f'seed_{seed}', 'metrics.json')

            if not force and os.path.exists(metrics_path):
                logger.info(f"Skipping: {config_file} seed {seed} (exists)")
                continue

            run_cmd([
                sys.executable, 'main.py',
                '--config', config_file,
                '--seed', str(seed),
            ], f'Train {config_file} seed {seed}')


def stage_ablate(markets, force=False):
    """Run ablation studies."""
    logger.info("=== STAGE: ablate ===")
    market_arg = 'both' if len(markets) > 1 else markets[0]
    run_cmd([
        sys.executable, 'experiments/ablation.py',
        '--market', market_arg,
    ], 'Ablation studies')


def stage_backtest(force=False):
    """Run backtest on all crypto results."""
    logger.info("=== STAGE: backtest ===")
    run_cmd([
        sys.executable, 'experiments/backtest.py',
        '--results-dir', 'results/',
        '--output-dir', 'results/backtest/',
    ], 'Backtest')


def stage_regime(force=False):
    """Run regime analysis."""
    logger.info("=== STAGE: regime ===")
    run_cmd([
        sys.executable, 'experiments/regime_split.py',
        '--results-dir', 'results/',
        '--output-dir', 'results/regime/',
    ], 'Regime analysis')


def stage_significance(force=False):
    """Run significance tests."""
    logger.info("=== STAGE: significance ===")
    run_cmd([
        sys.executable, 'experiments/run_significance.py',
        '--results-dir', 'results/',
        '--output-dir', 'results/significance/',
    ], 'Significance tests')


def stage_aggregate(force=False):
    """Aggregate and produce LaTeX tables."""
    logger.info("=== STAGE: aggregate ===")
    run_cmd([
        sys.executable, '-c',
        'from eval.aggregate import aggregate_all; aggregate_all()',
    ], 'Aggregate results')


def stage_figures(force=False):
    """Generate all figures from saved outputs."""
    logger.info("=== STAGE: figures ===")
    logger.info("(Figure generation to be implemented with matplotlib)")
    # Placeholder — figures will be generated from saved CSVs and JSONs


def main():
    parser = argparse.ArgumentParser(description="Stage-based experiment pipeline (Change 7)")
    parser.add_argument('--stage', type=str, default='all',
                        choices=STAGES + ['all'],
                        help="Which stage to run (default: all)")
    parser.add_argument('--market', type=str, default='both',
                        choices=['crypto', 'fi2010', 'both'],
                        help="Which market(s) to process")
    parser.add_argument('--force', action='store_true',
                        help="Force re-run even if outputs exist")
    args = parser.parse_args()

    markets = MARKETS if args.market == 'both' else [args.market]

    stage_runners = {
        'prepare': lambda: stage_prepare(markets, args.force),
        'sweep': lambda: stage_sweep(markets, args.force),
        'tune': lambda: stage_tune(markets, args.force),
        'train': lambda: stage_train(markets, args.force),
        'ablate': lambda: stage_ablate(markets, args.force),
        'backtest': lambda: stage_backtest(args.force),
        'regime': lambda: stage_regime(args.force),
        'significance': lambda: stage_significance(args.force),
        'aggregate': lambda: stage_aggregate(args.force),
        'figures': lambda: stage_figures(args.force),
    }

    if args.stage == 'all':
        for stage in STAGES:
            stage_runners[stage]()
    else:
        stage_runners[args.stage]()


if __name__ == '__main__':
    main()
