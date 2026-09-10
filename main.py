"""
main.py — Single entry point for one (model, market, seed) experiment.

Change 4: Rewritten to use data/loaders.py as the single source of truth.
  4.1 — Deleted inline load_data; calls loaders.py
  4.2 — Reads feature_set, label_rule, threshold_mode, horizon, window_len from config
  4.3 — Routes neural models through data/sequences.py when window_len > 1
  4.4 [BUG B8] — Saves test_predictions.npy, test_probs.npy, test_labels.npy,
                  test_timestamps.npy, test_mid.npy
  4.5 — Saves config_used.json for every run
  4.6 — Expanded manifest
  4.7 — Smoke-test through same code path
"""

import os
import sys
import subprocess
import argparse
import yaml
import json
import logging
import datetime
import time
import hashlib
import numpy as np
import torch
import sklearn
import xgboost as xgb
from torch.utils.data import DataLoader

from utils.seeding import set_seed
from utils.logging import get_logger
from data.features import TrainOnlyScaler
from data.loaders import FI2010Dataset, CryptoDataset
from data.sequences import create_datasets
from eval.metrics import compute_all_metrics

# Model type classification
NEURAL_MODELS = {
    'deeplob', 'deeplob_full', 'feature_conv_bilstm',
    'transformer', 'transformer_windowed', 'scalar_token_transformer',
    'structured_transformer', 'level_transformer',
    'deeplob_attention',
}
TREE_MODELS = {'xgboost', 'random_forest', 'logistic'}

# All temporal models that need windowed input
TEMPORAL_MODELS = {
    'deeplob_full', 'transformer_windowed', 'level_transformer',
    'deeplob_attention',
}


def _get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description="Run one (model, market, seed) experiment.")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--smoke-test', action='store_true',
                        help="Run a quick smoke test with reduced data and epochs")
    args = parser.parse_args()

    # 1. SET SEED FIRST
    set_seed(args.seed)

    logger = get_logger(__name__)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['seed'] = args.seed

    # Setup per-run output directory
    base_out_dir = config.get('output_dir', 'results/default/')
    run_dir = os.path.join(base_out_dir, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"Starting run: config={args.config}, seed={args.seed}")
    logger.info(f"Output dir: {run_dir}")

    # 2. Extract config keys (Change 4.2)
    market = config['market']
    model_type = config.get('model', 'xgboost')
    data_cfg = config.get('data', {})
    feature_set = data_cfg.get('feature_set', 'raw40')
    label_rule = data_cfg.get('label_rule', 'point_return')
    threshold_mode = data_cfg.get('threshold_mode', 'fixed')
    threshold_param = data_cfg.get('threshold_param', 0.0001)
    window_len = data_cfg.get('window_len', 1)
    standardize = data_cfg.get('standardize', True)

    # Temporal models need windowed input
    is_neural = model_type in NEURAL_MODELS
    is_temporal = model_type in TEMPORAL_MODELS

    if is_temporal and window_len <= 1:
        window_len = 100  # Default for temporal models
        logger.info(f"Temporal model {model_type}: setting window_len=100")

    # 3. Data Loading (Change 4.1 — uses loaders.py)
    start_load = time.time()

    if market == 'fi2010':
        # Auto-prepare if needed
        train_path = data_cfg.get('train_path', 'data/processed/fi2010_zscore_train.npy')
        test_path = data_cfg.get('test_path', 'data/processed/fi2010_zscore_test.npy')

        # Check for legacy paths too
        if not os.path.exists(train_path):
            legacy_train = 'data/processed/fi2010_train.npy'
            if os.path.exists(legacy_train):
                train_path = legacy_train
                test_path = 'data/processed/fi2010_test.npy'
            else:
                logger.info("FI-2010 processed files not found. Running prepare_fi2010.py...")
                subprocess.run([sys.executable, 'scripts/prepare_fi2010.py'])

        dataset = FI2010Dataset(
            train_path=train_path,
            test_path=test_path,
            horizon_k=data_cfg.get('horizon_k', 10),
            feature_set=feature_set,
            window_len=window_len,
            standardize=standardize,
            fi2010_variant=data_cfg.get('fi2010_variant', 'Zscore'),
        )
        (X_train, y_train, X_val, y_val, X_test, y_test), metadata = dataset.get_splits()

    elif market == 'crypto':
        parquet_path = data_cfg.get('parquet_path', 'data/processed/crypto_BTCUSDT.parquet')

        # Check legacy path
        if not os.path.exists(parquet_path):
            legacy_path = 'data/processed/crypto_data.parquet'
            if os.path.exists(legacy_path):
                parquet_path = legacy_path
            else:
                logger.info("Crypto processed file not found. Running prepare_crypto.py...")
                subprocess.run([sys.executable, 'scripts/prepare_crypto.py'])

        dataset = CryptoDataset(
            parquet_path=parquet_path,
            symbol=data_cfg.get('symbol', 'BTCUSDT'),
            horizon=data_cfg.get('horizon_events', 40),
            threshold_mode=threshold_mode,
            threshold_param=threshold_param,
            label_rule=label_rule,
            feature_set=feature_set,
            window_len=window_len,
            start_ts=data_cfg.get('start_ts'),
            end_ts=data_cfg.get('end_ts'),
            standardize=standardize,
        )
        (X_train, y_train, X_val, y_val, X_test, y_test), metadata = dataset.get_splits()

    else:
        raise ValueError(f"Unknown market: {market!r}")

    load_time = time.time() - start_load
    logger.info(f"Data loaded in {load_time:.1f}s")

    # Smoke test truncation (Change 4.7 — same code path)
    if args.smoke_test:
        logger.info("SMOKE TEST: Truncating dataset and reducing training iterations.")
        n_smoke = min(200, len(X_train))
        X_train, y_train = X_train[:n_smoke], y_train[:n_smoke]
        X_val, y_val = X_val[:min(100, len(X_val))], y_val[:min(100, len(y_val))]
        X_test, y_test = X_test[:min(100, len(X_test))], y_test[:min(100, len(y_test))]

        # Ensure all 3 classes present
        if len(y_train) >= 3:
            y_train[0], y_train[1], y_train[2] = 0, 1, 2

        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs_max'] = 2
        config['training']['patience'] = 2

        if 'model_params' not in config:
            config['model_params'] = {}
        if model_type in TREE_MODELS:
            config['model_params']['n_estimators'] = 5
            config['model_params']['max_depth'] = 3

    # 4. Scaling — fit ONLY on training data
    if standardize:
        fi2010_variant = data_cfg.get('fi2010_variant', 'Zscore')
        # Change 1.5.4: Skip for FI-2010 Zscore (already z-scored)
        skip_scaling = (market == 'fi2010' and fi2010_variant == 'Zscore')
        if skip_scaling:
            logger.info("Skipping StandardScaler for FI-2010 Zscore (already z-scored)")
        else:
            logger.info("Fitting Z-score scaler on training data only...")
            scaler = TrainOnlyScaler(use_zscore=True)
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

    # 5. Training
    start_train = time.time()

    if is_neural:
        from train.train_neural import train_neural_model

        batch_size = config.get('training', {}).get('batch_size', 128)

        # Change 4.3 — Route through sequences.py
        train_ds, val_ds, test_ds = create_datasets(
            X_train, y_train, X_val, y_val, X_test, y_test,
            window_len=window_len
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = train_neural_model(config, train_loader, val_loader, run_dir)

        # Test inference
        device = next(model.parameters()).device
        model.eval()
        test_preds = []
        test_probs = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())

        test_preds = np.array(test_preds)
        test_probs = np.array(test_probs)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        from train.train_tree import train_tree_model
        model = train_tree_model(config, X_train, y_train, X_val, y_val, run_dir)
        test_preds = model.predict(X_test)

        # Get probabilities for backtest
        if hasattr(model, 'predict_proba'):
            test_probs = model.predict_proba(X_test)
        else:
            test_probs = np.zeros((len(test_preds), 3))
            test_probs[np.arange(len(test_preds)), test_preds.astype(int)] = 1.0

        n_params = 0  # Not applicable for sklearn models

    train_time = time.time() - start_train

    # 6. Metrics
    metrics = compute_all_metrics(y_test, test_preds)
    logger.info(f"Final Test Macro-F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Final Test Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Final Test MCC:       {metrics['mcc']:.4f}")

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 7. Save run manifest (Change 4.6 — expanded)
    manifest = {
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'market': market,
        'model': model_type,
        'seed': args.seed,
        'config_file': args.config,
        'n_params': n_params,
        'train_time_s': round(train_time, 2),
        'feature_set': feature_set,
        'label_rule': label_rule,
        'threshold_mode': threshold_mode,
        'threshold_used': metadata.get('threshold_used'),
        'horizon_events': metadata.get('horizon_events') or metadata.get('horizon_k'),
        'horizon_seconds': metadata.get('horizon_seconds'),
        'window_len': window_len,
        'start_ts': metadata.get('start_ts'),
        'end_ts': metadata.get('end_ts'),
        'gap': metadata.get('gap'),
        'split_hash': metadata.get('split_hash'),
        'n_train': metadata.get('n_train'),
        'n_val': metadata.get('n_val'),
        'n_test': metadata.get('n_test'),
        'class_dist_train': metadata.get('class_dist_train'),
        'class_dist_val': metadata.get('class_dist_val'),
        'class_dist_test': metadata.get('class_dist_test'),
        'git_commit': _get_git_commit(),
        'library_versions': {
            'torch': torch.__version__,
            'sklearn': sklearn.__version__,
            'xgboost': xgb.__version__,
            'numpy': np.__version__,
        }
    }
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4, default=str)

    # 8. Save config_used.json for every run (Change 4.5)
    with open(os.path.join(run_dir, 'config_used.json'), 'w') as f:
        json.dump(config, f, indent=4, default=str)

    # 9. Save raw predictions and probabilities (Change 4.4 — BUG B8 fix)
    np.save(os.path.join(run_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(run_dir, 'test_probs.npy'), test_probs)
    np.save(os.path.join(run_dir, 'test_labels.npy'), y_test)

    # Save timestamps and mid-price for crypto backtest
    if market == 'crypto' and hasattr(dataset, 'test_mid') and dataset.test_mid is not None:
        np.save(os.path.join(run_dir, 'test_mid.npy'), dataset.test_mid)
    if market == 'crypto' and hasattr(dataset, 'test_timestamps') and dataset.test_timestamps is not None:
        np.save(os.path.join(run_dir, 'test_timestamps.npy'), dataset.test_timestamps)
    if market == 'crypto' and hasattr(dataset, 'test_bid') and dataset.test_bid is not None:
        np.save(os.path.join(run_dir, 'test_bid.npy'), dataset.test_bid)
        np.save(os.path.join(run_dir, 'test_ask.npy'), dataset.test_ask)

    logger.info(f"Run completed. Results saved to {run_dir}")


if __name__ == '__main__':
    main()
