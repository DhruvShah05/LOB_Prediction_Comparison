"""
models/deeplob.py — Faithful DeepLOB implementation (Change 2.1, BUG B2, R1).

Reference: Zhang, Zohren, Roberts (2019), "DeepLOB: Deep Convolutional Neural
Networks for Limit Order Books", and the authors' public implementation.

Input shape: (batch, 1, T=100, 40) — the 40 columns must be in the canonical
per-level order: [ask_p_i, ask_v_i, bid_p_i, bid_v_i for i in 1..10].

Architecture:
  Block 1 (price/volume pairs): Conv2d(1,32,(1,2),stride=(1,2)) → Conv2d(32,32,(4,1)) × 2
  Block 2 (ask/bid sides):      Conv2d(32,32,(1,2),stride=(1,2)) → Conv2d(32,32,(4,1)) × 2
  Block 3 (across levels):      Conv2d(32,32,(1,10))             → Conv2d(32,32,(4,1)) × 2
  Inception module: three parallel branches (64 channels each) → concat → (batch,192,T,1)
  LSTM(hidden=64) → last hidden → Linear(64,3)

All convs followed by LeakyReLU(0.01) and BatchNorm2d.
Parameter count ≈ 140k.

The old single-snapshot model is preserved as FeatureConvBiLSTM for ablation.
"""

import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    """
    Faithful DeepLOB (Zhang, Zohren, Roberts 2019).
    Input: (batch, 1, T, 40) where T = window_len (default 100).
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 3, window_len: int = 100,
                 in_features: int = 40, eng_features: int = 0):
        """
        Parameters
        ----------
        num_classes : number of output classes
        window_len : temporal window length (T)
        in_features : number of raw LOB features (40 for canonical layout)
        eng_features : number of additional engineered features to concatenate
                       to the LSTM output before the classifier (Change 2.1 option a)
        """
        super().__init__()
        self.window_len = window_len
        self.in_features = in_features
        self.eng_features = eng_features

        # Block 1 — price/volume pairs
        # Input: (B, 1, T, 40) → after Conv2d(1,32,(1,2),stride=(1,2)): (B, 32, T, 20)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),  # 'same' along time
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Block 2 — ask/bid sides
        # Input: (B, 32, T, 20) → after Conv2d(32,32,(1,2),stride=(1,2)): (B, 32, T, 10)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Block 3 — across levels
        # Input: (B, 32, T, 10) → after Conv2d(32,32,(1,10)): (B, 32, T, 1)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Inception module — three parallel branches, 64 channels each
        # Input: (B, 32, T', 1)
        # Branch 1: 1×1 → 3×1
        self.inception_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        # Branch 2: 1×1 → 5×1
        self.inception_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        # Branch 3: MaxPool 3×1 → 1×1
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1,
                            batch_first=True)

        # Classifier
        classifier_in = 64 + eng_features
        self.fc = nn.Linear(classifier_in, num_classes)

    def forward(self, x, x_eng=None):
        """
        Parameters
        ----------
        x : tensor of shape (batch, T, 40) or (batch, 1, T, 40)
        x_eng : optional tensor of shape (batch, T, eng_features) — engineered
                features concatenated to LSTM output (Change 2.1 option a)
        """
        # Ensure 4D input
        if x.dim() == 3:
            # (batch, T, 40) → (batch, 1, T, 40)
            x = x.unsqueeze(1)

        # Conv blocks
        x = self.block1(x)   # → (B, 32, T', 20)
        x = self.block2(x)   # → (B, 32, T', 10)
        x = self.block3(x)   # → (B, 32, T', 1)

        # Inception
        i1 = self.inception_1(x)
        i2 = self.inception_2(x)
        i3 = self.inception_3(x)
        x = torch.cat([i1, i2, i3], dim=1)  # → (B, 192, T', 1)

        # Reshape for LSTM: (B, T', 192)
        x = x.squeeze(3).permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T', 64)
        last_hidden = lstm_out[:, -1, :]  # (B, 64)

        # Concatenate engineered features if provided
        if x_eng is not None:
            # Use the last timestep's engineered features
            if x_eng.dim() == 3:
                x_eng = x_eng[:, -1, :]  # (B, eng_features)
            last_hidden = torch.cat([last_hidden, x_eng], dim=1)

        logits = self.fc(last_hidden)
        return logits


class FeatureConvBiLSTM(nn.Module):
    """
    Original single-snapshot model (preserved for ablation — Change 2.1).
    "No history, conv over feature index" — useful evidence that the old
    paper's model was the problem.

    Takes input of shape (batch, features) — single snapshot, no time axis.
    Uses 3 1D Convolutional blocks followed by a 2-layer BiLSTM.
    """
    def __init__(self, in_features: int, num_classes: int = 3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(32)
        )

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, num_classes)  # 64 * 2 (bidirectional)

    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits


def build_model(config: dict) -> nn.Module:
    model_name = config.get('model', 'deeplob_full')
    model_params = config.get('model_params', {})
    market = config.get('market', 'crypto')
    feature_set = config.get('data', {}).get('feature_set', 'raw40')
    window_len = config.get('data', {}).get('window_len', 100)

    if model_name == 'feature_conv_bilstm':
        # Ablation-only: old single-snapshot model
        in_features = 144 if market == 'fi2010' and feature_set == 'raw40_eng' else 40
        return FeatureConvBiLSTM(in_features=in_features)

    # Faithful DeepLOB (default)
    eng_features = 0
    if feature_set == 'raw40_eng':
        # Engineered features concatenated to LSTM output (option a)
        # The exact count depends on the feature engineering output
        eng_features = model_params.get('eng_features', 0)

    return DeepLOB(
        num_classes=model_params.get('num_classes', 3),
        window_len=window_len,
        in_features=40,
        eng_features=eng_features,
    )
