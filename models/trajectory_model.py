"""Common predictor interface and factory for all trajectory models.

Every predictor exposes ``predict(obs_xy: np.ndarray [B, T_OBS, 2]) -> [B, T_PRED, 2]``
in absolute world coordinates (metres), so evaluation and breach logic are
model-agnostic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch

import config
from dataset_loader import NormStats, compute_features, last_displacement
from models.kalman_baseline import ConstantVelocityKF, PolynomialExtrapolator
from models.seq2seq_predictor import Seq2SeqPredictor

logger = logging.getLogger("models")

MODEL_NAMES = ("kalman", "poly1", "poly2", "lstm", "gru")


@runtime_checkable
class TrajectoryPredictor(Protocol):
    """Anything with ``predict(obs_xy [B, T_OBS, 2]) -> [B, T_PRED, 2]``."""

    name: str

    def predict(self, obs_xy: np.ndarray) -> np.ndarray:
        """Forecast absolute future positions."""
        ...


class NeuralPredictor:
    """Wraps a trained :class:`Seq2SeqPredictor` + normalisation stats as a predictor."""

    def __init__(self, model: Seq2SeqPredictor, stats: NormStats,
                 device: Optional[torch.device] = None) -> None:
        """Store the model (in eval mode) and its normalisation statistics."""
        self.device = device or config.get_device()
        self.model = model.to(self.device).eval()
        self.stats = stats
        self.name = model.cell_type

    @torch.no_grad()
    def predict(self, obs_xy: np.ndarray) -> np.ndarray:
        """obs -> features -> network -> relative displacement -> absolute positions."""
        obs = np.asarray(obs_xy, dtype=np.float64)
        feats = self.stats.normalize(compute_features(obs)).astype(np.float32)
        last = (last_displacement(obs) / self.stats.disp_scale).astype(np.float32)
        rel = self.model(torch.from_numpy(feats).to(self.device),
                         torch.from_numpy(last).to(self.device))
        return rel.cpu().numpy().astype(np.float64) * self.stats.disp_scale + obs[:, -1:, :]

    @classmethod
    def from_checkpoint(cls, path: Path, stats_path: Path = config.NORM_STATS_PATH,
                        device: Optional[torch.device] = None) -> "NeuralPredictor":
        """Load weights (CPU/GPU agnostic) and normalisation stats from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found - run train.py first")
        device = device or config.get_device()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = Seq2SeqPredictor(**ckpt["model_config"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, NormStats.load(stats_path), device)


def build_predictor(name: str, device: Optional[torch.device] = None) -> TrajectoryPredictor:
    """Factory: ``kalman | poly1 | poly2 | lstm | gru``."""
    if name == "kalman":
        return ConstantVelocityKF()
    if name in ("poly1", "poly2"):
        return PolynomialExtrapolator(degree=int(name[-1]))
    if name in ("lstm", "gru"):
        return NeuralPredictor.from_checkpoint(config.checkpoint_path(name), device=device)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
