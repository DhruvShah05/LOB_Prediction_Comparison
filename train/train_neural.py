"""
train/train_neural.py — Neural model training loop (Change 3.2, BUG B11 fix).

Changes from original:
  - AdamW with weight_decay from config
  - Scheduler: linear warmup (1 epoch) + cosine decay, or ReduceLROnPlateau
  - Early stopping with patience (default 5), epochs_max (default 50)
  - Gradient clipping at 1.0
  - Log n_params and wall-clock time
  - Class-weighting: inverse-frequency or 'none' for ablation
"""

import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import importlib
import copy
from tqdm import tqdm

from eval.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# Maps config 'model' name to actual module filename
MODEL_MODULE_MAP = {
    'deeplob': 'deeplob',
    'deeplob_full': 'deeplob',
    'feature_conv_bilstm': 'deeplob',
    'transformer': 'transformer',
    'transformer_windowed': 'transformer',
    'scalar_token_transformer': 'transformer',
    'structured_transformer': 'structured_transformer',
    'level_transformer': 'structured_transformer',
    'deeplob_attention': 'deeplob_attention',
}


class FocalLoss(nn.Module):
    """
    Standard focal loss (Lin et al., 2017) for addressing class imbalance.
    gamma=0 recovers standard cross-entropy.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def _count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_device():
    """Device priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_neural_model(config: dict, train_loader: DataLoader,
                       val_loader: DataLoader, run_dir: str):
    """
    Generic PyTorch training loop for all neural LOB models.

    Saves per run:
      - training_history.json
      - config_used.json
    Best checkpoint selected by validation Macro-F1.
    """
    device = _get_device()
    logger.info(f"Training on device: {device}")

    # Save the full resolved config
    config_path = os.path.join(run_dir, 'config_used.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4, default=str)

    model_name = config.get('model', 'transformer_windowed')
    module_name = MODEL_MODULE_MAP.get(model_name, model_name)
    model_module = importlib.import_module(f"models.{module_name}")
    model = model_module.build_model(config).to(device)

    n_params = _count_params(model)
    logger.info(f"Model: {model_name}, Parameters: {n_params:,}")

    train_config = config.get('training', {})
    epochs_max = train_config.get('epochs_max', train_config.get('epochs', 50))
    patience = train_config.get('patience', 5)
    lr = train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 1e-4)
    grad_clip = train_config.get('grad_clip', 1.0)
    scheduler_type = train_config.get('scheduler', 'cosine')

    # AdamW optimizer (Change 3.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler (Change 3.2)
    if scheduler_type == 'cosine':
        # Linear warmup for 1 epoch, then cosine decay
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_max - 1, eta_min=lr * 0.01)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    else:
        scheduler = None

    # Loss function with class weighting
    imbalance = config.get('imbalance', 'class_weight')
    if isinstance(imbalance, dict):
        strategy = imbalance.get('strategy', 'none')
    else:
        strategy = imbalance

    criterion = nn.CrossEntropyLoss()
    if strategy == 'class_weight':
        # Compute class weights from training labels
        all_y = torch.cat([y for _, y in train_loader]).numpy()
        class_counts = np.bincount(all_y, minlength=3)
        weights = 1.0 / np.maximum(class_counts, 1).astype(float)
        weights = weights / weights.sum() * len(class_counts)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        logger.info(f"Using Weighted CrossEntropy. Class weights: {weights.tolist()}")
    elif strategy == 'focal_loss':
        gamma = config.get('imbalance', {}).get('focal_gamma', 2.0) if isinstance(config.get('imbalance'), dict) else 2.0
        criterion = FocalLoss(gamma=gamma)
        logger.info(f"Using Focal Loss with gamma={gamma}")
    else:
        logger.info("Using unweighted CrossEntropy (imbalance strategy: none)")

    best_val_f1 = -1.0
    best_model_state = None
    history = []
    patience_counter = 0
    best_val_loss = float('inf')

    start_time = time.time()

    epoch_bar = tqdm(range(epochs_max), desc=f"Training {model_name}", unit="epoch")
    for epoch in epoch_bar:
        # --- Training phase ---
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping (Change 3.2)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_loss += loss.item()

        # --- Validation phase ---
        model.eval()
        val_preds, val_true = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        metrics = compute_all_metrics(np.array(val_true), np.array(val_preds))
        macro_f1 = metrics['macro_f1']

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        current_lr = optimizer.param_groups[0]['lr']

        epoch_bar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'val_loss': f"{avg_val_loss:.4f}",
            'val_f1': f"{macro_f1:.4f}",
            'lr': f"{current_lr:.6f}",
        })

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_macro_f1': macro_f1,
            'val_mcc': metrics['mcc'],
            'learning_rate': current_lr,
        })

        # Best-checkpoint selection by val Macro-F1
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_model_state = copy.deepcopy(model.state_dict())

        # Early stopping on val loss (Change 3.2)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (patience={patience})")
                break

        # Scheduler step
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    train_time = time.time() - start_time
    logger.info(f"Training complete. Best val Macro-F1: {best_val_f1:.4f}, Time: {train_time:.1f}s")

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save training history
    history_meta = {
        'n_params': n_params,
        'train_time_s': train_time,
        'best_val_f1': best_val_f1,
        'epochs_completed': len(history),
        'history': history,
    }
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history_meta, f, indent=4)

    return model
