"""
train/tune.py — Two-phase hyperparameter tuning (Change 3.1).

Shared by tree and neural models. One Optuna study per cell:
  tune.py --model xgboost --market crypto --feature_set raw40 --label_rule smoothed

Uses TPE sampler with fixed sampler seed (12345).
  - Trees: n_trials >= 30, all trials use model seed 0
  - Neural: n_trials >= 20

Writes frozen configs to configs/tuned/<market>_<feature_set>_<label_rule>_<model>.yaml
and the full trial log.
"""

import argparse
import json
import os
import logging
import yaml
import optuna
import numpy as np

from sklearn.utils.class_weight import compute_sample_weight

from eval.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed sampler seed for reproducibility
SAMPLER_SEED = 12345
MODEL_SEED = 0  # All trials use seed 0 for the model

# Search spaces per model
SEARCH_SPACES = {
    'xgboost': {
        'max_depth': ('int', 3, 12),
        'learning_rate': ('log_float', 1e-3, 0.3),
        'n_estimators': ('int', 200, 2000),
        'subsample': ('float', 0.5, 1.0),
        'colsample_bytree': ('float', 0.3, 1.0),
        'min_child_weight': ('int', 1, 20),
        'reg_lambda': ('log_float', 1e-3, 10.0),
    },
    'random_forest': {
        'n_estimators': ('categorical', [200, 500, 1000]),
        'max_depth': ('int_or_none', 4, 30),
        'min_samples_leaf': ('int', 1, 50),
        'max_features': ('categorical', ['sqrt', 0.3, 0.5]),
    },
    'deeplob_full': {
        'lr': ('log_float', 1e-4, 1e-2),
        'd_model': ('categorical', [32, 64, 128]),
        'dropout': ('float', 0.0, 0.3),
        'weight_decay': ('log_float', 1e-6, 1e-2),
        'batch_size': ('categorical', [64, 128, 256]),
    },
    'transformer_windowed': {
        'lr': ('log_float', 1e-4, 1e-2),
        'd_model': ('categorical', [32, 64, 128]),
        'num_layers': ('categorical', [1, 2, 4]),
        'dropout': ('float', 0.0, 0.3),
        'weight_decay': ('log_float', 1e-6, 1e-2),
        'batch_size': ('categorical', [64, 128, 256]),
    },
    'level_transformer': {
        'lr': ('log_float', 1e-4, 1e-2),
        'd_model': ('categorical', [32, 64, 128]),
        'num_layers': ('categorical', [1, 2, 4]),
        'dropout': ('float', 0.0, 0.3),
        'weight_decay': ('log_float', 1e-6, 1e-2),
        'batch_size': ('categorical', [64, 128, 256]),
    },
    'deeplob_attention': {
        'lr': ('log_float', 1e-4, 1e-2),
        'dropout': ('float', 0.0, 0.3),
        'weight_decay': ('log_float', 1e-6, 1e-2),
        'batch_size': ('categorical', [64, 128, 256]),
    },
}

TREE_MODELS = {'xgboost', 'random_forest'}
NEURAL_MODELS = {'deeplob_full', 'transformer_windowed', 'level_transformer',
                 'deeplob_attention'}


def _suggest_param(trial, name, spec):
    """Suggest a parameter from its spec tuple."""
    ptype = spec[0]
    if ptype == 'int':
        return trial.suggest_int(name, spec[1], spec[2])
    elif ptype == 'float':
        return trial.suggest_float(name, spec[1], spec[2])
    elif ptype == 'log_float':
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif ptype == 'categorical':
        return trial.suggest_categorical(name, spec[1])
    elif ptype == 'int_or_none':
        use_none = trial.suggest_categorical(f'{name}_none', [True, False])
        if use_none:
            return None
        return trial.suggest_int(name, spec[1], spec[2])
    else:
        raise ValueError(f"Unknown param type: {ptype}")


def tune_tree(model_name, X_train, y_train, X_val, y_val, n_trials=30):
    """Run Optuna tuning for a tree model."""
    import importlib

    MODULE_MAP = {'xgboost': 'xgboost_model', 'random_forest': 'random_forest'}
    module_name = MODULE_MAP.get(model_name, model_name)
    search_space = SEARCH_SPACES[model_name]

    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            params[name] = _suggest_param(trial, name, spec)

        config = {
            'model': model_name,
            'model_params': params,
            'seed': MODEL_SEED,
            'imbalance': {'strategy': 'class_weight'},
        }

        model_module = importlib.import_module(f"models.{module_name}")
        model = model_module.build_model(config)

        # XGBoost sample weights
        fit_kwargs = {}
        if model_name == 'xgboost':
            sw = compute_sample_weight(class_weight='balanced', y=y_train)
            fit_kwargs['sample_weight'] = sw
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['verbose'] = False

        model.fit(X_train, y_train, **fit_kwargs)
        preds = model.predict(X_val)
        metrics = compute_all_metrics(y_val, preds)
        return metrics['macro_f1']

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study


def tune_neural(model_name, X_train, y_train, X_val, y_val,
                window_len=100, n_trials=20):
    """Run Optuna tuning for a neural model."""
    import torch
    from torch.utils.data import DataLoader
    from data.sequences import create_datasets
    from train.train_neural import train_neural_model
    import tempfile

    search_space = SEARCH_SPACES.get(model_name, SEARCH_SPACES['transformer_windowed'])

    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            params[name] = _suggest_param(trial, name, spec)

        batch_size = params.pop('batch_size', 128)
        lr = params.pop('lr', 1e-3)
        weight_decay = params.pop('weight_decay', 1e-4)

        config = {
            'model': model_name,
            'model_params': params,
            'seed': MODEL_SEED,
            'data': {'window_len': window_len, 'feature_set': 'raw40'},
            'training': {
                'epochs_max': 20,  # Reduced for tuning
                'patience': 5,
                'batch_size': batch_size,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'scheduler': 'cosine',
                'grad_clip': 1.0,
            },
            'imbalance': 'class_weight',
            'market': 'crypto',
        }

        train_ds, val_ds, _ = create_datasets(
            X_train, y_train, X_val, y_val, X_val, y_val,
            window_len=window_len
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_neural_model(config, train_loader, val_loader, tmpdir)

        # Evaluate on val
        device = next(model.parameters()).device
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())

        metrics = compute_all_metrics(y_val, np.array(val_preds))
        return metrics['macro_f1']

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning (Change 3.1)")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(SEARCH_SPACES.keys()))
    parser.add_argument('--market', type=str, required=True, choices=['fi2010', 'crypto'])
    parser.add_argument('--feature_set', type=str, default='raw40',
                        choices=['raw40', 'raw40_eng'])
    parser.add_argument('--label_rule', type=str, default='smoothed',
                        choices=['point_return', 'smoothed'])
    parser.add_argument('--n_trials', type=int, default=None,
                        help="Number of Optuna trials (default: 30 for trees, 20 for neural)")
    parser.add_argument('--config', type=str, default=None,
                        help="Base config file to load data settings from")
    args = parser.parse_args()

    # Determine number of trials
    is_tree = args.model in TREE_MODELS
    n_trials = args.n_trials or (30 if is_tree else 20)

    # Load data
    logger.info(f"Loading data: market={args.market}, feature_set={args.feature_set}, "
                f"label_rule={args.label_rule}")

    if args.market == 'fi2010':
        from data.loaders import FI2010Dataset
        dataset = FI2010Dataset(feature_set=args.feature_set)
        (X_train, y_train, X_val, y_val, X_test, y_test), metadata = dataset.get_splits()
    else:
        from data.loaders import CryptoDataset
        dataset = CryptoDataset(
            feature_set=args.feature_set,
            label_rule=args.label_rule,
        )
        (X_train, y_train, X_val, y_val, X_test, y_test), metadata = dataset.get_splits()

    # Standardize
    from data.features import TrainOnlyScaler
    scaler = TrainOnlyScaler(use_zscore=True)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    logger.info(f"Tuning {args.model} with {n_trials} trials...")

    if is_tree:
        best_params, study = tune_tree(args.model, X_train, y_train, X_val, y_val, n_trials)
    else:
        window_len = 100  # default for temporal models
        best_params, study = tune_neural(args.model, X_train, y_train, X_val, y_val,
                                          window_len, n_trials)

    logger.info(f"Best params: {best_params}")
    logger.info(f"Best val Macro-F1: {study.best_value:.4f}")

    # Save tuned config
    out_dir = 'configs/tuned'
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{args.market}_{args.feature_set}_{args.label_rule}_{args.model}.yaml"
    out_path = os.path.join(out_dir, out_name)

    tuned_config = {
        'model_params': best_params,
        'best_val_macro_f1': float(study.best_value),
        'n_trials': n_trials,
        'sampler_seed': SAMPLER_SEED,
    }
    with open(out_path, 'w') as f:
        yaml.dump(tuned_config, f, default_flow_style=False)
    logger.info(f"Tuned config saved to {out_path}")

    # Save full trial log
    log_path = out_path.replace('.yaml', '_trials.json')
    trials = []
    for t in study.trials:
        trials.append({
            'number': t.number,
            'value': t.value,
            'params': t.params,
            'state': str(t.state),
        })
    with open(log_path, 'w') as f:
        json.dump(trials, f, indent=2)
    logger.info(f"Trial log saved to {log_path}")


if __name__ == '__main__':
    main()
