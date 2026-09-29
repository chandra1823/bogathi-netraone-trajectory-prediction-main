"""LSTM / GRU encoder-decoder for trajectory forecasting.

Encoder: 2-layer LSTM (or GRU) over the T_OBS feature vectors.
Decoder: autoregressive LSTMCell (or GRUCell) seeded with the encoder's top-layer
state. At every step it predicts the next per-step displacement as a residual on
the previous displacement (a constant-velocity prior the network learns to
correct), and the displacements are cumulatively summed into positions relative
to the last observation. Teacher forcing feeds ground-truth displacements.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

import config


class Seq2SeqPredictor(nn.Module):
    """Encoder-decoder mapping [B, T_OBS, F] features to [B, T_PRED, 2] relative positions."""

    def __init__(self, input_size: int = config.NUM_FEATURES, hidden_size: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS, dropout: float = config.DROPOUT,
                 t_pred: int = config.T_PRED, cell: str = "lstm") -> None:
        """Create encoder, decoder cell, input embedding and output head."""
        super().__init__()
        if cell not in ("lstm", "gru"):
            raise ValueError("cell must be 'lstm' or 'gru'")
        self.cell_type, self.t_pred, self.hidden_size = cell, t_pred, hidden_size
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.input_proj = nn.Sequential(nn.Linear(input_size, hidden_size // 2), nn.ReLU())
        self.encoder = rnn(hidden_size // 2, hidden_size, num_layers=num_layers,
                           dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.dec_embed = nn.Sequential(nn.Linear(2, hidden_size // 4), nn.ReLU())
        cell_cls = nn.LSTMCell if cell == "lstm" else nn.GRUCell
        self.decoder = cell_cls(hidden_size // 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                                  nn.Linear(hidden_size // 2, 2))

    def forward(self, feats: torch.Tensor, last_disp: torch.Tensor,
                target_disp: Optional[torch.Tensor] = None, tf_ratio: float = 0.0) -> torch.Tensor:
        """Forecast relative future positions.

        Args:
            feats: [B, T_OBS, F] normalised features.
            last_disp: [B, 2] last observed displacement (scaled), first decoder input.
            target_disp: [B, T_PRED, 2] ground-truth scaled displacements (teacher forcing).
            tf_ratio: probability of feeding the ground truth at each decoder step.

        Returns:
            [B, T_PRED, 2] scaled positions relative to the last observed point.
        """
        _, state = self.encoder(self.input_proj(feats))
        if self.cell_type == "lstm":
            h, c = state[0][-1], state[1][-1]
        else:
            h, c = state[-1], None
        prev = last_disp
        disps = []
        for t in range(self.t_pred):
            inp = self.dec_embed(prev)
            if c is not None:
                h, c = self.decoder(inp, (h, c))
            else:
                h = self.decoder(inp, h)
            disp = prev + self.head(self.dropout(h))
            disps.append(disp)
            if target_disp is not None and tf_ratio > 0 and torch.rand(()) < tf_ratio:
                prev = target_disp[:, t]
            else:
                prev = disp
        return torch.cumsum(torch.stack(disps, dim=1), dim=1)
