"""
models/transformer.py — Windowed standard Transformer (Change 2.2, R1).

Input: (batch, T, F). One token per snapshot.
  Linear(F, d_model) → positional encoding over time → TransformerEncoder →
  pooling (mean over time by default) → Linear(d_model, 3).

The old scalar-token model is preserved as ScalarTokenTransformer for ablation.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over the time dimension."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class WindowedTransformer(nn.Module):
    """
    Windowed standard Transformer for LOB prediction.
    Input: (batch, T, F) — T snapshots of F features each.
    """

    def __init__(self, in_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.1,
                 pooling: str = 'mean'):
        super().__init__()
        self.pooling = pooling

        # Project snapshot features to d_model
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=500, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        if pooling == 'attention':
            self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.attn_pool = nn.MultiheadAttention(embed_dim=d_model, num_heads=1,
                                                    batch_first=True)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (batch, T, F) — windowed input
        """
        batch_size = x.size(0)

        # Project: (batch, T, F) → (batch, T, d_model)
        x = self.input_proj(x)

        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        if self.pooling == 'cls':
            pooled = x[:, 0, :]
        elif self.pooling == 'attention':
            query = self.attn_pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.attn_pool(query, x, x)
            pooled = pooled.squeeze(1)
        else:  # 'mean'
            pooled = x.mean(dim=1)

        logits = self.fc(pooled)
        return logits


class ScalarTokenTransformer(nn.Module):
    """
    Original scalar-token Transformer (preserved for ablation — Change 2.2).
    Treats each scalar feature as a token in a sequence.
    Input: (batch, features) — single snapshot.
    """
    def __init__(self, in_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, num_classes: int = 3):
        super().__init__()

        self.token_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=in_features)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.token_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        pooled = x.mean(dim=1)
        logits = self.fc(pooled)
        return logits


def build_model(config: dict) -> nn.Module:
    model_name = config.get('model', 'transformer_windowed')
    model_params = config.get('model_params', {})
    market = config.get('market', 'crypto')
    feature_set = config.get('data', {}).get('feature_set', 'raw40')

    d_model = model_params.get('d_model', 64)
    nhead = model_params.get('nhead', 4)
    num_layers = model_params.get('num_layers', 2)
    dropout = model_params.get('dropout', 0.1)
    pooling = model_params.get('pooling', 'mean')

    if model_name == 'scalar_token_transformer':
        # Ablation-only: old scalar-token model
        in_features = 144 if market == 'fi2010' and feature_set == 'raw40_eng' else 40
        return ScalarTokenTransformer(
            in_features=in_features, d_model=d_model, nhead=nhead,
            num_layers=num_layers
        )

    # Default: windowed Transformer
    # in_features depends on feature_set
    if feature_set == 'raw40':
        in_features = 40
    elif feature_set == 'raw40_eng':
        in_features = model_params.get('in_features', 40)
    else:
        in_features = 40

    return WindowedTransformer(
        in_features=in_features, d_model=d_model, nhead=nhead,
        num_layers=num_layers, dropout=dropout, pooling=pooling
    )
