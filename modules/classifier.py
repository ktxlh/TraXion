"""Classifier module for binary anomaly prediction.

This module implements a simple feedforward neural network that maps
stay embeddings to anomaly scores.
"""

from torch import nn


class Classifier(nn.Module):
    """Multi-layer perceptron for binary anomaly classification.

    Takes stay embeddings and produces logits for anomaly detection.
    The architecture progressively reduces dimensionality:
    model_dim -> model_dim//2 -> model_dim//4 -> 1

    Args:
        model_dim: Dimension of input stay embeddings
    """

    def __init__(self, model_dim, dropout=0.1):
        super(Classifier, self).__init__()

        self.stay_classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 4, 1),
        )

    def forward(self, visit_emb):
        # (batch_size, seq_len)
        logits = self.stay_classifier(visit_emb).squeeze(-1)
        return logits
