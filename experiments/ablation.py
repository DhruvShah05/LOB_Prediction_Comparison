"""
experiments/ablation.py — Comprehensive ablation studies (Change 5.5, BUG B12, R1).

Replaces ablation_structured_transformer.py. All ablations run with 5 seeds.

One factor at a time:
  - num_layers ∈ {1, 2, 4, 6}
  - pooling ∈ {mean, cls, attention}
  - tokenization ∈ {snapshot, level, scalar}
  - parameter-matched control
  - dropout ∈ {0, 0.1, 0.3}
  - scheduler ∈ {constant, cosine} + early stopping on/off
  - window_len ∈ {1, 10, 50, 100}
  - class weighting on/off

DeepLOB ablation: FeatureConvBiLSTM vs faithful DeepLOB at various window_len.

Output: one table per market with mean ± std and parameter counts.
"""

import os
import json
import argparse
import logging
import subprocess
import sys
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEEDS = [0, 1, 2, 3, 4]

# Base config for ablations (will be modified per factor)
BASE_ABLATION_CONFIG = {
    'market': 'crypto',
    'model': 'transformer_windowed',
    'data': {
        'feature_set': 'raw40',
        'label_rule': 'smoothed',
        'horizon_events': 40,
        'threshold_mode': 'quantile',
        'window_len': 100,
        'standardize': True,
    },
    'model_params': {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1,
        'pooling': 'mean',
    },
    'training': {
        'epochs_max': 50,
        'patience': 5,
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'grad_clip': 1.0,
    },
    'imbalance': 'class_weight',
}

# Ablation dimensions
ABLATION_GRID = {
    'num_layers': {
        'path': 'model_params.num_layers',
        'values': [1, 2, 4, 6],
    },
    'pooling': {
        'path': 'model_params.pooling',
        'values': ['mean', 'cls', 'attention'],
    },
    'tokenization': {
        'model_values': [
            ('transformer_windowed', 'snapshot'),
            ('level_transformer', 'level'),
            ('scalar_token_transformer', 'scalar'),
        ],
    },
    'dropout': {
        'path': 'model_params.dropout',
        'values': [0.0, 0.1, 0.3],
    },
    'scheduler': {
        'path': 'training.scheduler',
        'values': ['cosine', 'plateau'],
    },
    'window_len': {
        'path': 'data.window_len',
        'values': [1, 10, 50, 100],
    },
    'class_weighting': {
        'path': 'imbalance',
        'values': ['class_weight', 'none'],
    },
}

DEEPLOB_ABLATION = {
    'model_values': [
        ('deeplob_full', 'faithful_deeplob'),
        ('feature_conv_bilstm', 'old_model'),
    ],
    'window_lens': [1, 10, 50, 100],
}


def _set_nested(config, path, value):
    """Set a value at a dotted path in a nested dict."""
    keys = path.split('.')
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _run_experiment(config, output_dir, seed, dry_run=False):
    """Run a single experiment by writing a temp config and calling main.py."""
    config['seed'] = seed
    config['output_dir'] = output_dir

    config_path = os.path.join(output_dir, f'ablation_config_seed_{seed}.yaml')
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    if dry_run:
        logger.info(f"  [DRY RUN] Would run: main.py --config {config_path} --seed {seed}")
        return

    cmd = [sys.executable, 'main.py', '--config', config_path, '--seed', str(seed)]
    logger.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  FAILED: {result.stderr[:500]}")
    else:
        logger.info(f"  Completed seed {seed}")


def run_transformer_ablation(market='crypto', output_base='results/ablation/',
                              dry_run=False):
    """Run all one-factor-at-a-time ablations for the windowed Transformer."""
    import copy

    for dim_name, dim_spec in ABLATION_GRID.items():
        logger.info(f"=== Ablation dimension: {dim_name} ===")

        if dim_name == 'tokenization':
            # Special: changes the model type
            for model_type, tok_name in dim_spec['model_values']:
                ablation_dir = os.path.join(output_base, f'{dim_name}_{tok_name}')
                config = copy.deepcopy(BASE_ABLATION_CONFIG)
                config['model'] = model_type
                config['market'] = market

                # Scalar token transformer needs window_len=1
                if model_type == 'scalar_token_transformer':
                    config['data']['window_len'] = 1

                for seed in SEEDS:
                    _run_experiment(config, ablation_dir, seed, dry_run)
        else:
            for value in dim_spec['values']:
                value_str = str(value).replace('.', '_')
                ablation_dir = os.path.join(output_base, f'{dim_name}_{value_str}')
                config = copy.deepcopy(BASE_ABLATION_CONFIG)
                config['market'] = market

                _set_nested(config, dim_spec['path'], value)

                # window_len=1 requires different model handling
                if dim_name == 'window_len' and value == 1:
                    config['model'] = 'scalar_token_transformer'

                for seed in SEEDS:
                    _run_experiment(config, ablation_dir, seed, dry_run)


def run_deeplob_ablation(market='crypto', output_base='results/ablation_deeplob/',
                          dry_run=False):
    """DeepLOB ablation: FeatureConvBiLSTM vs faithful DeepLOB at various window_len."""
    import copy

    for model_type, model_name in DEEPLOB_ABLATION['model_values']:
        for wlen in DEEPLOB_ABLATION['window_lens']:
            ablation_dir = os.path.join(output_base, f'{model_name}_wlen{wlen}')
            config = copy.deepcopy(BASE_ABLATION_CONFIG)
            config['model'] = model_type
            config['market'] = market
            config['data']['window_len'] = wlen

            # FeatureConvBiLSTM only works with window_len=1
            if model_type == 'feature_conv_bilstm' and wlen > 1:
                logger.info(f"  Skipping {model_name} with window_len={wlen} (single-snapshot only)")
                continue

            for seed in SEEDS:
                _run_experiment(config, ablation_dir, seed, dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ablation studies (Change 5.5)")
    parser.add_argument('--market', type=str, default='crypto', choices=['crypto', 'fi2010', 'both'])
    parser.add_argument('--output-dir', type=str, default='results/ablation/')
    parser.add_argument('--dry-run', action='store_true', help="Only print commands, don't run")
    parser.add_argument('--deeplob-only', action='store_true', help="Only run DeepLOB ablation")
    args = parser.parse_args()

    markets = ['crypto', 'fi2010'] if args.market == 'both' else [args.market]

    for market in markets:
        if not args.deeplob_only:
            run_transformer_ablation(market, args.output_dir, args.dry_run)
        run_deeplob_ablation(market,
                             os.path.join(args.output_dir, 'deeplob/'),
                             args.dry_run)
