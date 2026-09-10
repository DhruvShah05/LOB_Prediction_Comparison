"""
experiments/transfer.py — Cross-market transfer learning (Change 5.7, NICE).

Train on FI-2010 raw40 → test on crypto raw40 and vice versa.
Also: pretrain on FI-2010, fine-tune on first crypto day.
"""

import os
import json
import argparse
import logging
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.loaders import FI2010Dataset, CryptoDataset
from data.features import TrainOnlyScaler
from data.sequences import create_datasets
from train.train_neural import train_neural_model
from eval.metrics import compute_all_metrics
from utils.seeding import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(market, feature_set='raw40', label_rule='smoothed', window_len=100):
    """Load and standardize data for a market."""
    if market == 'fi2010':
        ds = FI2010Dataset(feature_set=feature_set, window_len=window_len)
    else:
        ds = CryptoDataset(feature_set=feature_set, label_rule=label_rule,
                           window_len=window_len)

    (X_tr, y_tr, X_v, y_v, X_te, y_te), meta = ds.get_splits()

    scaler = TrainOnlyScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_v = scaler.transform(X_v)
    X_te = scaler.transform(X_te)

    return X_tr, y_tr, X_v, y_v, X_te, y_te, meta, scaler


def eval_model(model, test_loader, device):
    """Evaluate a trained model on a test loader."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(by.numpy())
    return compute_all_metrics(np.array(trues), np.array(preds))


def run_transfer(source_market, target_market, model_name='transformer_windowed',
                 output_dir='results/transfer/', seed=0):
    """Train on source, test on target."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    window_len = 100
    batch_size = 128

    # Load source and target data
    X_tr_s, y_tr_s, X_v_s, y_v_s, X_te_s, y_te_s, meta_s, scaler_s = \
        load_data(source_market, window_len=window_len)
    X_tr_t, y_tr_t, X_v_t, y_v_t, X_te_t, y_te_t, meta_t, scaler_t = \
        load_data(target_market, window_len=window_len)

    # Train on source
    config = {
        'model': model_name,
        'market': source_market,
        'seed': seed,
        'data': {'window_len': window_len, 'feature_set': 'raw40'},
        'model_params': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1},
        'training': {
            'epochs_max': 30, 'patience': 5, 'batch_size': batch_size,
            'learning_rate': 0.001, 'weight_decay': 1e-4, 'scheduler': 'cosine',
            'grad_clip': 1.0,
        },
        'imbalance': 'class_weight',
    }

    train_ds, val_ds, _ = create_datasets(X_tr_s, y_tr_s, X_v_s, y_v_s,
                                           X_te_s, y_te_s, window_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    run_dir = os.path.join(output_dir, f'{source_market}_to_{target_market}')
    os.makedirs(run_dir, exist_ok=True)

    model = train_neural_model(config, train_loader, val_loader, run_dir)
    device = next(model.parameters()).device

    # Evaluate on source test (baseline)
    _, _, test_ds_s = create_datasets(X_tr_s, y_tr_s, X_v_s, y_v_s,
                                       X_te_s, y_te_s, window_len)
    test_loader_s = DataLoader(test_ds_s, batch_size=batch_size, shuffle=False)
    source_metrics = eval_model(model, test_loader_s, device)

    # Evaluate on target test (transfer)
    _, _, test_ds_t = create_datasets(X_tr_t, y_tr_t, X_v_t, y_v_t,
                                       X_te_t, y_te_t, window_len)
    test_loader_t = DataLoader(test_ds_t, batch_size=batch_size, shuffle=False)
    transfer_metrics = eval_model(model, test_loader_t, device)

    results = {
        'source_market': source_market,
        'target_market': target_market,
        'model': model_name,
        'seed': seed,
        'source_test_metrics': source_metrics,
        'transfer_test_metrics': transfer_metrics,
    }

    with open(os.path.join(run_dir, 'transfer_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Transfer {source_market}→{target_market}: "
                f"Source F1={source_metrics['macro_f1']:.4f}, "
                f"Transfer F1={transfer_metrics['macro_f1']:.4f}")

    return results


def run_finetune(pretrain_market='fi2010', finetune_market='crypto',
                 model_name='transformer_windowed',
                 output_dir='results/transfer/', seed=0):
    """Pretrain on source, fine-tune on target."""
    set_seed(seed)

    window_len = 100
    batch_size = 128

    # Load data
    X_tr_s, y_tr_s, X_v_s, y_v_s, X_te_s, y_te_s, _, _ = \
        load_data(pretrain_market, window_len=window_len)
    X_tr_t, y_tr_t, X_v_t, y_v_t, X_te_t, y_te_t, _, _ = \
        load_data(finetune_market, window_len=window_len)

    # Pretrain
    config = {
        'model': model_name, 'market': pretrain_market, 'seed': seed,
        'data': {'window_len': window_len, 'feature_set': 'raw40'},
        'model_params': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1},
        'training': {
            'epochs_max': 20, 'patience': 5, 'batch_size': batch_size,
            'learning_rate': 0.001, 'weight_decay': 1e-4, 'scheduler': 'cosine',
            'grad_clip': 1.0,
        },
        'imbalance': 'class_weight',
    }

    train_ds, val_ds, _ = create_datasets(X_tr_s, y_tr_s, X_v_s, y_v_s,
                                           X_te_s, y_te_s, window_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    run_dir = os.path.join(output_dir, 'finetune')
    os.makedirs(run_dir, exist_ok=True)

    model = train_neural_model(config, train_loader, val_loader, run_dir)

    # Fine-tune on target with lower LR
    config_ft = copy.deepcopy(config)
    config_ft['training']['learning_rate'] = 1e-4
    config_ft['training']['epochs_max'] = 10
    config_ft['market'] = finetune_market

    train_ds_t, val_ds_t, test_ds_t = create_datasets(
        X_tr_t, y_tr_t, X_v_t, y_v_t, X_te_t, y_te_t, window_len)
    train_loader_t = DataLoader(train_ds_t, batch_size=batch_size, shuffle=True)
    val_loader_t = DataLoader(val_ds_t, batch_size=batch_size, shuffle=False)
    test_loader_t = DataLoader(test_ds_t, batch_size=batch_size, shuffle=False)

    # Continue training (model is already initialized with pretrained weights)
    model = train_neural_model(config_ft, train_loader_t, val_loader_t, run_dir)
    device = next(model.parameters()).device

    finetune_metrics = eval_model(model, test_loader_t, device)

    results = {
        'pretrain_market': pretrain_market,
        'finetune_market': finetune_market,
        'model': model_name,
        'finetune_metrics': finetune_metrics,
    }

    with open(os.path.join(run_dir, 'finetune_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Fine-tune {pretrain_market}→{finetune_market}: F1={finetune_metrics['macro_f1']:.4f}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer learning (Change 5.7)")
    parser.add_argument('--output-dir', type=str, default='results/transfer/')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # FI-2010 → Crypto
    run_transfer('fi2010', 'crypto', output_dir=args.output_dir, seed=args.seed)
    # Crypto → FI-2010
    run_transfer('crypto', 'fi2010', output_dir=args.output_dir, seed=args.seed)
    # Fine-tune: pretrain FI-2010, fine-tune crypto
    run_finetune(output_dir=args.output_dir, seed=args.seed)
