"""
models/deeplob_attention.py — DeepLOB-Attention (Change 2.4b).

Reference: Zhang & Zohren (2021), "Multi-Horizon Forecasting for Limit Order
Books: Deep Learning, Recurrent Attention and Temporal Attention."

Architecture: Reuses the DeepLOB encoder (conv blocks + inception → LSTM) from
models/deeplob.py, then replaces the simple last-hidden classifier with an
attention-based decoder that can produce multi-horizon output.

For single-horizon use: the attention decoder attends to all LSTM hidden states
and produces a single (batch, num_classes) output.
"""

import torch
import torch.nn as nn
from .deeplob import DeepLOB


class AttentionDecoder(nn.Module):
    """
    Attention-based decoder for the DeepLOB encoder output.
    Attends to all LSTM hidden states to produce the final classification.
    """

    def __init__(self, hidden_size: int = 64, num_classes: int = 3):
        super().__init__()
        self.hidden_size = hidden_size

        # Attention mechanism
        self.attn_query = nn.Linear(hidden_size, hidden_size)
        self.attn_key = nn.Linear(hidden_size, hidden_size)
        self.attn_value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

        # Output projection
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, lstm_output):
        """
        lstm_output: (batch, T, hidden_size) — all LSTM hidden states
        Returns: (batch, num_classes)
        """
        # Self-attention over temporal dimension
        Q = self.attn_query(lstm_output)   # (B, T, H)
        K = self.attn_key(lstm_output)     # (B, T, H)
        V = self.attn_value(lstm_output)   # (B, T, H)

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, T, T)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, V)  # (B, T, H)

        # Use the last position's context vector
        last_context = context[:, -1, :]  # (B, H)

        logits = self.fc(last_context)
        return logits


class DeepLOBAttention(nn.Module):
    """
    DeepLOB-Attention (Zhang & Zohren 2021).
    Uses the DeepLOB encoder followed by an attention decoder.

    Input: (batch, T, 40) or (batch, 1, T, 40)
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 3, window_len: int = 100,
                 in_features: int = 40, eng_features: int = 0):
        super().__init__()
        self.window_len = window_len

        # Reuse DeepLOB conv blocks and inception (not LSTM)
        self.encoder = DeepLOB(
            num_classes=num_classes, window_len=window_len,
            in_features=in_features, eng_features=eng_features
        )

        # Replace the LSTM + classifier with attention-based decoder
        # The DeepLOB encoder produces LSTM output of hidden_size=64
        self.attention_decoder = AttentionDecoder(
            hidden_size=64, num_classes=num_classes
        )

    def forward(self, x, x_eng=None):
        """
        x: (batch, T, 40) or (batch, 1, T, 40)
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Run through conv blocks and inception (reuse encoder's blocks)
        x = self.encoder.block1(x)
        x = self.encoder.block2(x)
        x = self.encoder.block3(x)

        # Inception
        i1 = self.encoder.inception_1(x)
        i2 = self.encoder.inception_2(x)
        i3 = self.encoder.inception_3(x)
        x = torch.cat([i1, i2, i3], dim=1)  # (B, 192, T', 1)

        # Reshape for LSTM
        x = x.squeeze(3).permute(0, 2, 1)  # (B, T', 192)

        # LSTM
        lstm_out, _ = self.encoder.lstm(x)  # (B, T', 64)

        # Attention decoder
        logits = self.attention_decoder(lstm_out)
        return logits


def build_model(config: dict) -> nn.Module:
    model_params = config.get('model_params', {})
    window_len = config.get('data', {}).get('window_len', 100)
    feature_set = config.get('data', {}).get('feature_set', 'raw40')

    eng_features = 0
    if feature_set == 'raw40_eng':
        eng_features = model_params.get('eng_features', 0)

    return DeepLOBAttention(
        num_classes=model_params.get('num_classes', 3),
        window_len=window_len,
        in_features=40,
        eng_features=eng_features,
    )
