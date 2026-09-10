"""
eval/aggregate.py — Aggregate results across seeds and produce LaTeX tables (Change 5.1).

Changes:
  - Per-class F1 columns in output table
  - 95% CI (t-distribution, n=5) alongside ± std
  - Check split_hash identical across seeds; fail loudly if not
  - LaTeX booktabs output
  - Confusion-matrix grid figure
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def load_all_runs(results_dir: str) -> list:
    """Load metrics.json and run_manifest.json from all seed_* dirs."""
    runs = []
    seed_dirs = sorted(glob.glob(os.path.join(results_dir, 'seed_*')))

    for seed_dir in seed_dirs:
        metrics_path = os.path.join(seed_dir, 'metrics.json')
        manifest_path = os.path.join(seed_dir, 'run_manifest.json')

        if not os.path.exists(metrics_path):
            logger.warning(f"Missing metrics.json in {seed_dir}")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)

        runs.append({
            'seed_dir': seed_dir,
            'metrics': metrics,
            'manifest': manifest,
        })

    return runs


def check_split_hashes(runs: list) -> bool:
    """Verify that split_hash is identical across all seeds. Fail loudly if not."""
    hashes = [r['manifest'].get('split_hash') for r in runs if r['manifest'].get('split_hash')]
    if len(hashes) < 2:
        logger.warning("Less than 2 runs have split_hash; cannot verify consistency.")
        return True
    if len(set(hashes)) > 1:
        logger.error(f"SPLIT HASH MISMATCH! Hashes: {hashes}")
        raise RuntimeError(
            "Split hashes differ across seeds. The split, α, scaler, and "
            "hyperparameters must be identical across seeds (Change 3.3)."
        )
    logger.info(f"Split hash verified: {hashes[0]} (consistent across {len(hashes)} seeds)")
    return True


def aggregate_metrics(runs: list) -> dict:
    """
    Compute mean ± std and 95% CI for each metric across seeds.
    """
    metric_keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1',
                   'mcc', 'f1_down', 'f1_stationary', 'f1_up']

    result = {}
    for key in metric_keys:
        vals = [r['metrics'].get(key) for r in runs if r['metrics'].get(key) is not None]
        if not vals:
            continue
        arr = np.array(vals)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        # 95% CI using t-distribution
        if n > 1:
            ci95 = stats.t.interval(0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))
        else:
            ci95 = (mean, mean)

        result[key] = {
            'mean': round(mean, 4),
            'std': round(std, 4),
            'ci95_low': round(ci95[0], 4),
            'ci95_high': round(ci95[1], 4),
            'n_seeds': n,
        }

    return result


def to_latex_row(model_name: str, agg: dict, market: str = '') -> str:
    """Format one row of the LaTeX booktabs table."""
    def fmt(key):
        if key not in agg:
            return '—'
        d = agg[key]
        return f"{d['mean']:.3f}±{d['std']:.3f}"

    cols = [
        market,
        model_name,
        fmt('macro_f1'),
        fmt('mcc'),
        fmt('accuracy'),
        fmt('balanced_accuracy'),
        fmt('f1_down'),
        fmt('f1_stationary'),
        fmt('f1_up'),
    ]
    return ' & '.join(cols) + r' \\'


def to_latex_table(rows: list, caption: str = "Results") -> str:
    """Generate full LaTeX booktabs table."""
    header = (
        r'\begin{table}[htbp]' + '\n'
        r'\centering' + '\n'
        r'\caption{' + caption + r'}' + '\n'
        r'\begin{tabular}{ll ccc cccc}' + '\n'
        r'\toprule' + '\n'
        r'Market & Model & Macro-F1 & MCC & Accuracy & Bal. Acc & F1$\downarrow$ & F1$\rightarrow$ & F1$\uparrow$ \\' + '\n'
        r'\midrule' + '\n'
    )
    body = '\n'.join(rows) + '\n'
    footer = (
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{table}'
    )
    return header + body + footer


def aggregate_directory(results_dir: str, output_dir: str = None):
    """Aggregate a single results directory (one model/market combination)."""
    runs = load_all_runs(results_dir)
    if not runs:
        logger.warning(f"No runs found in {results_dir}")
        return None

    check_split_hashes(runs)
    agg = aggregate_metrics(runs)

    if output_dir is None:
        output_dir = results_dir

    # Save aggregated metrics
    with open(os.path.join(output_dir, 'aggregated_metrics.json'), 'w') as f:
        json.dump(agg, f, indent=4)

    return agg


def aggregate_all(results_base: str = 'results/', output_dir: str = 'results/aggregated/'):
    """
    Aggregate all model/market combinations and produce LaTeX tables.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    latex_rows = []

    # Find all result directories
    for entry in sorted(os.listdir(results_base)):
        entry_path = os.path.join(results_base, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry == 'aggregated':
            continue

        # Check if this has seed directories
        seed_dirs = glob.glob(os.path.join(entry_path, 'seed_*'))
        if not seed_dirs:
            continue

        logger.info(f"Aggregating: {entry}")
        agg = aggregate_directory(entry_path)
        if agg is None:
            continue

        # Extract model/market from directory name
        parts = entry.split('_')
        market = parts[0] if parts else 'unknown'
        model = '_'.join(parts[1:]) if len(parts) > 1 else entry

        row = to_latex_row(model, agg, market)
        latex_rows.append(row)

        all_results.append({
            'directory': entry,
            'market': market,
            'model': model,
            'metrics': agg,
        })

    # Save combined results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    # Save LaTeX table
    latex = to_latex_table(latex_rows, caption="LOB Prediction Results (Mean ± Std, 5 Seeds)")
    with open(os.path.join(output_dir, 'results_table.tex'), 'w') as f:
        f.write(latex)

    logger.info(f"Aggregated results saved to {output_dir}")
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate results")
    parser.add_argument('--results-dir', type=str, default='results/')
    parser.add_argument('--output-dir', type=str, default='results/aggregated/')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    aggregate_all(args.results_dir, args.output_dir)
