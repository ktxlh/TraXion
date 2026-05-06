import math

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    Implementation of https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(1, emb_dim)
        self.act = torch.sin

    def forward(self, x):
        """
        x: N, L
        o: N, L, emb_dim
        """
        x = x.unsqueeze(-1)  # N, L, 1
        h = self.linear(x)  # N, L, emb_dim
        # Inplace operation causes RuntimeError for gradient computation. Concat is necessary.
        o = torch.concat([h[..., :1], self.act(h[..., 1:])], dim=-1)
        return o


class LogFreqEmbed(nn.Module):
    """
    Temporal embedding that learns frequencies in log-space.

    Human periodicities span many orders of magnitude (10s sample rate → weekly cycle ≈ 60000x
    range). Parameterizing log(freq) and using exp(log_freq) as multipliers keeps gradients
    well-conditioned across this range, unlike Time2Vec whose linear weights must span the same
    scale directly.

    Frequencies are initialized log-uniformly over [min_period, max_period] (in the same time
    units as x). The first dimension is kept linear (no sin) following the Time2Vec convention.
    """

    def __init__(self, emb_dim, min_period=1 / 360, max_period=168):
        """
        Args:
            emb_dim: Output embedding dimension.
            min_period: Shortest period to initialize to (default: 10s in hours = 1/360).
            max_period: Longest period to initialize to (default: 168h = 1 week).
        """
        super().__init__()
        # Initialize log_scales log-uniformly across [min_freq, max_freq]
        log_freqs = torch.linspace(
            math.log(2 * math.pi / max_period),
            math.log(2 * math.pi / min_period),
            emb_dim,
        )
        self.log_scales = nn.Parameter(log_freqs)
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        x: N, L
        o: N, L, emb_dim
        """
        x = x.unsqueeze(-1)  # N, L, 1
        h = x * self.log_scales.exp() + self.bias  # N, L, emb_dim
        # First dim stays linear; rest are sinusoidal (same convention as Time2Vec)
        o = torch.concat([h[..., :1], torch.sin(h[..., 1:])], dim=-1)
        return o
