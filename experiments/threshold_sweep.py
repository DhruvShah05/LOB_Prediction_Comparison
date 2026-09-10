"""
experiments/threshold_sweep.py — Extended threshold & horizon sweep (Change 5.3, BUG B14).

Grid: label_rule ∈ {point_return, smoothed}
      × horizon ∈ {10, 20, 40, 100, 200, 400} (2.5s – 100s for crypto)
      × threshold_mode ∈ {fixed 0.5/1/2 bp, quantile, std_mult 0.5}

For each cell: class distribution per split, val Macro-F1/MCC for logistic
regression + XGBoost (fixed reasonable params, seed 0).

Output: CSVs + heatmap figure. Also runs on FI-2010 over its available
label columns.
"""

import os
import json
import argparse
import logging
import itertools
import numpy as np
import pandas as pd

from data.labeling import (
    compute_mid_price, compute_returns, resolve_threshold,
    label_by_threshold, horizon_events_to_seconds
)
from eval.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _train_eval_logistic(X_train, y_train, X_val, y_val):
    """Quick logistic regression evaluation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    model = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=500,
        class_weight='balanced', random_state=0, n_jobs=-1
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_v)
    return compute_all_metrics(y_val, preds)


def _train_eval_xgboost(X_train, y_train, X_val, y_val):
    """Quick XGBoost evaluation with fixed reasonable params."""
    import xgboost as xgb
    from sklearn.utils.class_weight import compute_sample_weight

    sw = compute_sample_weight(class_weight='balanced', y=y_train)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', num_class=3, random_state=0,
        n_jobs=-1, tree_method='hist'
    )
    model.fit(X_train, y_train, sample_weight=sw)
    preds = model.predict(X_val)
    return compute_all_metrics(y_val, preds)


def sweep_crypto(parquet_path='data/processed/crypto_BTCUSDT.parquet',
                 output_dir='results/threshold_sweep/'):
    """Run the full sweep on crypto data."""
    import pandas as pd_local
    from data.features import build_feature_matrix, TrainOnlyScaler, CANONICAL_RAW_COLS

    os.makedirs(output_dir, exist_ok=True)

    df = pd_local.read_parquet(parquet_path)

    # Compute mid-price on raw data
    mid_price = compute_mid_price(df).values

    # Build raw feature matrix
    X_raw = df[CANONICAL_RAW_COLS].values

    horizons = [10, 20, 40, 100, 200, 400]
    label_rules = ['point_return', 'smoothed']
    threshold_configs = [
        ('fixed', 0.00005),   # 0.5 bp
        ('fixed', 0.0001),    # 1 bp
        ('fixed', 0.0002),    # 2 bp
        ('quantile', None),
        ('std_mult', 0.5),
    ]

    results = []

    for label_rule, horizon, (tmode, tparam) in itertools.product(
            label_rules, horizons, threshold_configs):

        logger.info(f"Sweep: {label_rule}, H={horizon}, {tmode}({tparam})")

        try:
            returns = compute_returns(mid_price, horizon, label_rule)
            n_valid = len(returns)
            X = X_raw[:n_valid]

            # Split 70/15/15
            tr = int(n_valid * 0.70)
            val = int(n_valid * 0.85)
            gap = max(horizon, 1)

            train_returns = returns[:tr - gap]
            alpha = resolve_threshold(
                train_returns, tmode,
                tparam if tparam is not None else 0.0
            )

            labels = label_by_threshold(returns, alpha)

            X_train = X[:tr - gap]
            y_train = labels[:tr - gap]
            X_val = X[tr:val - gap]
            y_val = labels[tr:val - gap]

            # Class distribution
            _, train_counts = np.unique(y_train, return_counts=True)
            _, val_counts = np.unique(y_val, return_counts=True)
            train_pcts = train_counts / len(y_train) * 100 if len(y_train) > 0 else [0, 0, 0]

            if len(y_train) < 10 or len(np.unique(y_train)) < 2:
                logger.warning(f"  Skipping: insufficient data (train={len(y_train)})")
                continue

            # Evaluate logistic + XGBoost
            log_metrics = _train_eval_logistic(X_train, y_train, X_val, y_val)
            xgb_metrics = _train_eval_xgboost(X_train, y_train, X_val, y_val)

            row = {
                'market': 'crypto',
                'label_rule': label_rule,
                'horizon_events': horizon,
                'horizon_seconds': horizon_events_to_seconds(horizon),
                'threshold_mode': tmode,
                'threshold_param': tparam,
                'threshold_resolved': alpha,
                'n_train': len(y_train),
                'n_val': len(y_val),
                'pct_down': float(train_pcts[0]) if len(train_pcts) > 0 else 0,
                'pct_stat': float(train_pcts[1]) if len(train_pcts) > 1 else 0,
                'pct_up': float(train_pcts[2]) if len(train_pcts) > 2 else 0,
                'logistic_macro_f1': log_metrics['macro_f1'],
                'logistic_mcc': log_metrics['mcc'],
                'xgboost_macro_f1': xgb_metrics['macro_f1'],
                'xgboost_mcc': xgb_metrics['mcc'],
            }
            results.append(row)

        except Exception as e:
            logger.warning(f"  Error: {e}")
            continue

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'crypto_threshold_sweep.csv')
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Crypto sweep saved to {csv_path}")

    return df_results


def sweep_fi2010(output_dir='results/threshold_sweep/'):
    """Run horizon sweep on FI-2010 over its available label columns."""
    os.makedirs(output_dir, exist_ok=True)

    train_path = 'data/processed/fi2010_zscore_train.npy'
    if not os.path.exists(train_path):
        logger.warning("FI-2010 data not found, skipping FI-2010 sweep.")
        return None

    train_data = np.load(train_path)
    results = []

    for label_col_idx, label_col in enumerate(range(144, 149)):
        X_all = train_data[:, :40]  # raw 40
        y_all = train_data[:, label_col].astype(int) - 1  # 1/2/3 → 0/1/2

        # Split
        n = len(X_all)
        tr = int(n * 0.8)
        X_train, y_train = X_all[:tr], y_all[:tr]
        X_val, y_val = X_all[tr:], y_all[tr:]

        if len(np.unique(y_train)) < 2:
            continue

        log_metrics = _train_eval_logistic(X_train, y_train, X_val, y_val)
        xgb_metrics = _train_eval_xgboost(X_train, y_train, X_val, y_val)

        _, counts = np.unique(y_train, return_counts=True)
        pcts = counts / len(y_train) * 100

        results.append({
            'market': 'fi2010',
            'label_column': label_col,
            'label_col_idx': label_col_idx,
            'pct_down': float(pcts[0]) if len(pcts) > 0 else 0,
            'pct_stat': float(pcts[1]) if len(pcts) > 1 else 0,
            'pct_up': float(pcts[2]) if len(pcts) > 2 else 0,
            'logistic_macro_f1': log_metrics['macro_f1'],
            'logistic_mcc': log_metrics['mcc'],
            'xgboost_macro_f1': xgb_metrics['macro_f1'],
            'xgboost_mcc': xgb_metrics['mcc'],
        })

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'fi2010_horizon_sweep.csv')
    df_results.to_csv(csv_path, index=False)
    logger.info(f"FI-2010 sweep saved to {csv_path}")

    return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Threshold & horizon sweep (Change 5.3)")
    parser.add_argument('--market', type=str, default='both', choices=['crypto', 'fi2010', 'both'])
    parser.add_argument('--output-dir', type=str, default='results/threshold_sweep/')
    args = parser.parse_args()

    if args.market in ('crypto', 'both'):
        sweep_crypto(output_dir=args.output_dir)
    if args.market in ('fi2010', 'both'):
        sweep_fi2010(output_dir=args.output_dir)
