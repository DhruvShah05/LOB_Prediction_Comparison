"""
experiments/regime_split.py — Per-regime metrics analysis (Change 5.4, R2).

Primary answer to "one asset" criticism. On the crypto test period:
  - Compute rolling realized volatility, rolling spread, rolling depth
  - Bucket test rows into low/mid/high tertiles
  - Recompute all metrics per bucket from saved predictions
  - Per-calendar-day metrics
  - Rolling-origin folds from Change 1.3.3

Figures: Macro-F1 and MCC per regime per model; per-day metrics line plot.
"""

import os
import json
import glob
import argparse
import logging
import numpy as np
import pandas as pd

from eval.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_rolling_volatility(mid_prices, window=1200):
    """
    Rolling realized volatility (std of 250ms mid returns over a window).
    Default window=1200 snapshots = 5 minutes.
    """
    returns = np.diff(mid_prices) / mid_prices[:-1]
    returns = np.concatenate([[0], returns])  # pad first

    vol = pd.Series(returns).rolling(window=window, min_periods=1).std().values
    return vol


def compute_rolling_spread(bid_prices, ask_prices, window=1200):
    """Rolling average spread (ask - bid)."""
    spread = ask_prices - bid_prices
    rolling_spread = pd.Series(spread).rolling(window=window, min_periods=1).mean().values
    return rolling_spread


def bucket_into_tertiles(values):
    """Bucket values into low/mid/high tertiles. Returns integer labels 0/1/2."""
    q33 = np.percentile(values, 33.3)
    q67 = np.percentile(values, 66.7)

    buckets = np.ones(len(values), dtype=int)  # mid = 1
    buckets[values <= q33] = 0  # low
    buckets[values >= q67] = 2  # high

    return buckets, q33, q67


def per_regime_metrics(y_true, y_pred, regime_labels, regime_names=None):
    """Compute metrics for each regime bucket."""
    if regime_names is None:
        regime_names = {0: 'low', 1: 'mid', 2: 'high'}

    results = {}
    for regime_id, regime_name in regime_names.items():
        mask = regime_labels == regime_id
        if mask.sum() < 10:
            logger.warning(f"Regime {regime_name}: only {mask.sum()} samples, skipping")
            continue

        metrics = compute_all_metrics(y_true[mask], y_pred[mask])
        metrics['n_samples'] = int(mask.sum())
        results[regime_name] = metrics

    return results


def per_day_metrics(y_true, y_pred, timestamps):
    """Compute metrics per calendar day."""
    ts = pd.to_datetime(timestamps)
    dates = ts.date

    results = {}
    for date in sorted(set(dates)):
        mask = np.array([d == date for d in dates])
        if mask.sum() < 10:
            continue

        metrics = compute_all_metrics(y_true[mask], y_pred[mask])
        metrics['n_samples'] = int(mask.sum())
        results[str(date)] = metrics

    return results


def run_regime_analysis(results_dir, output_dir, seeds=None):
    """Run regime analysis on all saved crypto predictions."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    os.makedirs(output_dir, exist_ok=True)

    # Find all crypto model directories
    model_dirs = []
    for entry in sorted(os.listdir(results_dir)):
        if not entry.startswith('crypto'):
            continue
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path) and glob.glob(os.path.join(entry_path, 'seed_*')):
            model_dirs.append(entry)

    all_results = {}

    for model_dir in model_dirs:
        logger.info(f"Regime analysis: {model_dir}")
        model_results = {'volatility': {}, 'spread': {}, 'per_day': {}}

        for seed in seeds:
            seed_dir = os.path.join(results_dir, model_dir, f'seed_{seed}')
            pred_path = os.path.join(seed_dir, 'test_predictions.npy')
            label_path = os.path.join(seed_dir, 'test_labels.npy')
            mid_path = os.path.join(seed_dir, 'test_mid.npy')
            ts_path = os.path.join(seed_dir, 'test_timestamps.npy')
            bid_path = os.path.join(seed_dir, 'test_bid.npy')
            ask_path = os.path.join(seed_dir, 'test_ask.npy')

            if not os.path.exists(pred_path) or not os.path.exists(mid_path):
                logger.warning(f"Missing data for {seed_dir}")
                continue

            y_pred = np.load(pred_path)
            y_true = np.load(label_path)
            mid = np.load(mid_path)

            min_len = min(len(y_true), len(y_pred), len(mid))
            y_true, y_pred, mid = y_true[:min_len], y_pred[:min_len], mid[:min_len]

            # Volatility regime
            vol = compute_rolling_volatility(mid)
            vol_buckets, _, _ = bucket_into_tertiles(vol)
            vol_metrics = per_regime_metrics(y_true, y_pred, vol_buckets)

            for regime, metrics in vol_metrics.items():
                if regime not in model_results['volatility']:
                    model_results['volatility'][regime] = []
                model_results['volatility'][regime].append(metrics)

            # Spread regime
            if os.path.exists(bid_path) and os.path.exists(ask_path):
                bid = np.load(bid_path)[:min_len]
                ask = np.load(ask_path)[:min_len]
                spread = compute_rolling_spread(bid, ask)
                spread_buckets, _, _ = bucket_into_tertiles(spread)
                spread_metrics = per_regime_metrics(y_true, y_pred, spread_buckets)

                for regime, metrics in spread_metrics.items():
                    if regime not in model_results['spread']:
                        model_results['spread'][regime] = []
                    model_results['spread'][regime].append(metrics)

            # Per-day metrics
            if os.path.exists(ts_path):
                ts = np.load(ts_path, allow_pickle=True)[:min_len]
                day_metrics = per_day_metrics(y_true, y_pred, ts)
                model_results['per_day'][f'seed_{seed}'] = day_metrics

        all_results[model_dir] = model_results

    # Save results
    with open(os.path.join(output_dir, 'regime_analysis.json'), 'w') as f:
        json.dump(all_results, f, indent=4, default=str)

    logger.info(f"Regime analysis saved to {output_dir}")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Regime split analysis (Change 5.4)")
    parser.add_argument('--results-dir', type=str, default='results/')
    parser.add_argument('--output-dir', type=str, default='results/regime/')
    args = parser.parse_args()
    run_regime_analysis(args.results_dir, args.output_dir)
