"""
models/structured_transformer.py — LevelTransformer (Change 2.3, BUG B3/B4, R1).

Tokenization on the canonical raw-40 layout: one token per **level** =
4 values (ask_p_i, ask_v_i, bid_p_i, bid_v_i) plus that level's
spread/mid/imbalance if the engineered set is on.

Two-stage encoder (cleaner to ablate):
  Stage 1 (Level attention):  Within each snapshot, attend across 10 levels.
  Stage 2 (Temporal attention): Across time, attend over pooled snapshot vectors.

Pooling options: {mean, cls, attention}
Exposed: num_layers, d_model, nhead, dropout — all configurable for ablation.
"""

import torch
import torch.nn as nn
import math

from .transformer import PositionalEncoding


class LevelTransformer(nn.Module):
    """
    Two-stage Level Transformer for LOB prediction.

    Stage 1: Level attention — one token per level (10 levels), within each snapshot.
    Stage 2: Temporal attention — one token per snapshot, across time.

    Input: (batch, T, 40) — T snapshots of 40 features in canonical layout.
    """

    def __init__(self, in_features: int = 40, d_model: int = 64, nhead: int = 4,
                 num_layers_level: int = 2, num_layers_temporal: int = 2,
                 num_classes: int = 3, dropout: float = 0.1,
                 pooling: str = 'mean', n_levels: int = 10,
                 features_per_level: int = 4):
        super().__init__()
        self.n_levels = n_levels
        self.features_per_level = features_per_level
        self.pooling = pooling
        self.d_model = d_model

        # Level token projection: (features_per_level,) → (d_model,)
        self.level_proj = nn.Linear(features_per_level, d_model)

        # Level positional encoding (over levels 1..10)
        self.level_pos = PositionalEncoding(d_model, max_len=n_levels + 1, dropout=dropout)

        # Stage 1: Level attention
        level_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4, dropout=dropout
        )
        self.level_encoder = nn.TransformerEncoder(level_encoder_layer,
                                                    num_layers=num_layers_level)

        # Temporal positional encoding (over time steps)
        self.temporal_pos = PositionalEncoding(d_model, max_len=500, dropout=dropout)

        # Stage 2: Temporal attention
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4, dropout=dropout
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer,
                                                       num_layers=num_layers_temporal)

        # Pooling
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if pooling == 'attention':
            self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.attn_pool = nn.MultiheadAttention(embed_dim=d_model, num_heads=1,
                                                    batch_first=True)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (batch, T, 40) — windowed input with canonical layout
           40 features = 10 levels × 4 (ask_p, ask_v, bid_p, bid_v)
        """
        batch_size, T, F = x.shape

        # Reshape to per-level tokens: (batch * T, n_levels, features_per_level)
        x = x.view(batch_size * T, self.n_levels, self.features_per_level)

        # Project level tokens: (batch*T, 10, d_model)
        x = self.level_proj(x)
        x = self.level_pos(x)

        # Stage 1: Level attention within each snapshot
        x = self.level_encoder(x)  # (batch*T, 10, d_model)

        # Pool levels within each snapshot → (batch*T, d_model)
        snapshot_repr = x.mean(dim=1)  # mean over levels

        # Reshape to temporal: (batch, T, d_model)
        snapshot_repr = snapshot_repr.view(batch_size, T, self.d_model)

        # Add temporal positional encoding
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            snapshot_repr = torch.cat([cls_tokens, snapshot_repr], dim=1)

        snapshot_repr = self.temporal_pos(snapshot_repr)

        # Stage 2: Temporal attention
        temporal_out = self.temporal_encoder(snapshot_repr)  # (batch, T', d_model)

        # Final pooling
        if self.pooling == 'cls':
            pooled = temporal_out[:, 0, :]
        elif self.pooling == 'attention':
            query = self.attn_pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.attn_pool(query, temporal_out, temporal_out)
            pooled = pooled.squeeze(1)
        else:  # 'mean'
            pooled = temporal_out.mean(dim=1)

        logits = self.fc(pooled)
        return logits


def build_model(config: dict) -> nn.Module:
    model_params = config.get('model_params', {})

    d_model = model_params.get('d_model', 64)
    nhead = model_params.get('nhead', 4)
    num_layers_level = model_params.get('num_layers_level', 2)
    num_layers_temporal = model_params.get('num_layers_temporal', 2)
    # Support legacy 'num_layers' as shorthand for both
    if 'num_layers' in model_params:
        num_layers_level = model_params['num_layers']
        num_layers_temporal = model_params['num_layers']
    dropout = model_params.get('dropout', 0.1)
    pooling = model_params.get('pooling', 'mean')

    return LevelTransformer(
        in_features=40,
        d_model=d_model,
        nhead=nhead,
        num_layers_level=num_layers_level,
        num_layers_temporal=num_layers_temporal,
        dropout=dropout,
        pooling=pooling,
    )
