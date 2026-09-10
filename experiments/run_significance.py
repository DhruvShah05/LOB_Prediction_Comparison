"""
experiments/run_significance.py — Statistical significance tests (Change 5.2, BUG B13).

Loads test_predictions.npy for every run. For each (market, feature_set, label_rule)
and each model pair:
  - Paired bootstrap on Macro-F1 and MCC (seed-matched, 2000 resamples, fixed RNG)
  - McNemar test on accuracy
  - Holm correction across pairs

Outputs a significance matrix per market and marks significant differences.
"""

import os
import json
import glob
import argparse
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats as scipy_stats

from eval.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 2000


def paired_bootstrap_test(y_true, preds_a, preds_b, metric_fn, n_bootstrap=N_BOOTSTRAP,
                          seed=BOOTSTRAP_SEED):
    """
    Paired bootstrap test: is metric(A) significantly different from metric(B)?
    Returns p-value (two-sided).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    observed_a = metric_fn(y_true, preds_a)
    observed_b = metric_fn(y_true, preds_b)
    observed_diff = observed_a - observed_b

    count_extreme = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_a = metric_fn(y_true[idx], preds_a[idx])
        boot_b = metric_fn(y_true[idx], preds_b[idx])
        boot_diff = boot_a - boot_b

        # Two-sided test
        if abs(boot_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_bootstrap
    return p_value, observed_diff


def mcnemar_test(y_true, preds_a, preds_b):
    """
    McNemar's test for paired nominal data (accuracy comparison).
    Returns chi-squared statistic and p-value.
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Contingency: b01 = A wrong, B right; b10 = A right, B wrong
    b01 = np.sum(~correct_a & correct_b)
    b10 = np.sum(correct_a & ~correct_b)

    if b01 + b10 == 0:
        return 0.0, 1.0  # No discordant pairs

    # McNemar with continuity correction
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)
    return chi2, p_value


def holm_correction(p_values):
    """
    Holm-Bonferroni step-down correction for multiple comparisons.
    Returns adjusted p-values.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(n)

    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = min(p_values[idx] * (n - rank), 1.0)

    # Enforce monotonicity
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])

    return adjusted


def _macro_f1(y_true, y_pred):
    metrics = compute_all_metrics(y_true, y_pred)
    return metrics['macro_f1']


def _mcc(y_true, y_pred):
    metrics = compute_all_metrics(y_true, y_pred)
    return metrics['mcc']


def load_predictions(results_dir, model_dirs, seeds):
    """Load seed-matched predictions for all models."""
    model_preds = {}
    model_labels = {}

    for model_dir in model_dirs:
        preds_per_seed = []
        labels_per_seed = []

        for seed in seeds:
            seed_dir = os.path.join(results_dir, model_dir, f'seed_{seed}')
            pred_path = os.path.join(seed_dir, 'test_predictions.npy')
            label_path = os.path.join(seed_dir, 'test_labels.npy')

            if not os.path.exists(pred_path):
                logger.warning(f"Missing predictions: {pred_path}")
                continue

            preds_per_seed.append(np.load(pred_path))
            if os.path.exists(label_path):
                labels_per_seed.append(np.load(label_path))

        if preds_per_seed:
            model_preds[model_dir] = preds_per_seed
            if labels_per_seed:
                model_labels[model_dir] = labels_per_seed

    return model_preds, model_labels


def run_significance_tests(results_dir, output_dir, seeds=None):
    """Run all significance tests and output matrices."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    os.makedirs(output_dir, exist_ok=True)

    # Find all model directories
    model_dirs = []
    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path) and entry != 'aggregated':
            if glob.glob(os.path.join(entry_path, 'seed_*')):
                model_dirs.append(entry)

    if len(model_dirs) < 2:
        logger.warning("Need at least 2 models for significance testing.")
        return

    model_preds, model_labels = load_predictions(results_dir, model_dirs, seeds)

    # For each pair of models, run tests on seed-matched predictions
    pairs = list(combinations(model_dirs, 2))
    all_p_values_f1 = []
    all_p_values_mcc = []
    all_p_values_mcnemar = []
    pair_names = []

    for model_a, model_b in pairs:
        if model_a not in model_preds or model_b not in model_preds:
            continue

        preds_a_list = model_preds[model_a]
        preds_b_list = model_preds[model_b]
        labels_list = model_labels.get(model_a, model_labels.get(model_b, []))

        n_seeds = min(len(preds_a_list), len(preds_b_list), len(labels_list))
        if n_seeds == 0:
            continue

        # Pool seed-matched predictions for bootstrap
        f1_p_values = []
        mcc_p_values = []
        mcnemar_p_values = []

        for i in range(n_seeds):
            y_true = labels_list[i]
            pa = preds_a_list[i]
            pb = preds_b_list[i]

            # Ensure same length
            min_len = min(len(y_true), len(pa), len(pb))
            y_true, pa, pb = y_true[:min_len], pa[:min_len], pb[:min_len]

            p_f1, _ = paired_bootstrap_test(y_true, pa, pb, _macro_f1)
            p_mcc, _ = paired_bootstrap_test(y_true, pa, pb, _mcc)
            _, p_mcn = mcnemar_test(y_true, pa, pb)

            f1_p_values.append(p_f1)
            mcc_p_values.append(p_mcc)
            mcnemar_p_values.append(p_mcn)

        # Average p-values across seeds (conservative)
        all_p_values_f1.append(np.mean(f1_p_values))
        all_p_values_mcc.append(np.mean(mcc_p_values))
        all_p_values_mcnemar.append(np.mean(mcnemar_p_values))
        pair_names.append(f"{model_a} vs {model_b}")

    # Holm correction
    if all_p_values_f1:
        adj_f1 = holm_correction(np.array(all_p_values_f1))
        adj_mcc = holm_correction(np.array(all_p_values_mcc))
        adj_mcn = holm_correction(np.array(all_p_values_mcnemar))

        results = []
        for i, pair in enumerate(pair_names):
            results.append({
                'pair': pair,
                'p_macro_f1': float(all_p_values_f1[i]),
                'p_macro_f1_adj': float(adj_f1[i]),
                'p_mcc': float(all_p_values_mcc[i]),
                'p_mcc_adj': float(adj_mcc[i]),
                'p_mcnemar': float(all_p_values_mcnemar[i]),
                'p_mcnemar_adj': float(adj_mcn[i]),
                'sig_f1': bool(adj_f1[i] < 0.05),
                'sig_mcc': bool(adj_mcc[i] < 0.05),
                'sig_mcnemar': bool(adj_mcn[i] < 0.05),
            })

        with open(os.path.join(output_dir, 'significance_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Significance results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run significance tests (Change 5.2)")
    parser.add_argument('--results-dir', type=str, default='results/')
    parser.add_argument('--output-dir', type=str, default='results/significance/')
    args = parser.parse_args()
    run_significance_tests(args.results_dir, args.output_dir)
