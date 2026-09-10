"""
train/train_tree.py — Tree model training (Change 3.1, BUG B1 fix).

Optuna is REMOVED from this module. Tuning is done separately via train/tune.py.
This module reads frozen tuned configs and runs with random_state = run_seed.
Raises if model_params contains nulls.
"""

import os
import json
import logging
import importlib
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from eval.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# Maps config 'model' name to the actual Python module filename under models/.
MODEL_MODULE_MAP = {
    'xgboost': 'xgboost_model',
    'random_forest': 'random_forest',
    'logistic': 'logistic',
}


def train_tree_model(config: dict, X_train, y_train, X_val, y_val, run_dir: str):
    """
    Trains a tree/sklearn model using frozen hyperparameters.

    No Optuna here — hyperparameters must be fully resolved before calling this.
    The run seed only affects model random_state (bootstrap/column sampling).
    """
    model_name = config['model']
    module_name = MODEL_MODULE_MAP.get(model_name, model_name)
    model_module = importlib.import_module(f"models.{module_name}")

    model_params = config.get('model_params', {})

    # Validate: no nulls allowed (Change 3.1)
    for key, value in model_params.items():
        if value is None:
            raise ValueError(
                f"model_params.{key} is null! Run `python train/tune.py --model "
                f"{model_name} ...` first to freeze hyperparameters, then set them "
                "in the config. Optuna is no longer run inside the training loop."
            )

    logger.info(f"Training {model_name} with frozen params: {model_params}")
    model = model_module.build_model(config)

    # Handle class weights / sample weights
    fit_kwargs = {}
    imbalance = config.get('imbalance', 'class_weight')
    if isinstance(imbalance, dict):
        strategy = imbalance.get('strategy', 'none')
    else:
        strategy = imbalance

    if strategy == 'class_weight' and model_name == 'xgboost':
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        fit_kwargs['sample_weight'] = sample_weights
        # XGBoost early stopping on validation set
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs['verbose'] = False

    model.fit(X_train, y_train, **fit_kwargs)

    # Log validation metrics
    val_preds = model.predict(X_val)
    val_metrics = compute_all_metrics(y_val, val_preds)
    logger.info(f"Val Macro-F1: {val_metrics['macro_f1']:.4f}, Val MCC: {val_metrics['mcc']:.4f}")

    return model
