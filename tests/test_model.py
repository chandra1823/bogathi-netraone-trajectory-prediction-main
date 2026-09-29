"""Tests for the seq2seq network and the predictor wrapper."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import config
from dataset_loader import NormStats
from models.seq2seq_predictor import Seq2SeqPredictor
from models.trajectory_model import NeuralPredictor, build_predictor


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_forward_shapes(cell: str) -> None:
    """[B, T_OBS, F] -> [B, T_PRED, 2] with and without teacher forcing."""
    m = Seq2SeqPredictor(cell=cell)
    x = torch.randn(5, config.T_OBS, config.NUM_FEATURES)
    last = torch.randn(5, 2)
    assert m(x, last).shape == (5, config.T_PRED, 2)
    tgt = torch.randn(5, config.T_PRED, 2)
    out = m(x, last, tgt, tf_ratio=1.0)
    assert out.shape == (5, config.T_PRED, 2)
    out.sum().backward()


def test_neural_predictor_absolute_coordinates() -> None:
    """The wrapper returns absolute coordinates anchored at the last observation."""
    torch.manual_seed(0)
    stats = NormStats([0.0] * config.NUM_FEATURES, [1.0] * config.NUM_FEATURES, 1.0)
    model = Seq2SeqPredictor()
    for p in model.head.parameters():
        torch.nn.init.zeros_(p)  # zero residual -> constant-velocity rollout
    pred = NeuralPredictor(model, stats, torch.device("cpu")).predict(
        np.stack([np.arange(config.T_OBS) * 1.0, np.zeros(config.T_OBS)], axis=-1)[None])
    expected_x = config.T_OBS - 1 + np.arange(1, config.T_PRED + 1)
    assert pred.shape == (1, config.T_PRED, 2)
    assert np.allclose(pred[0, :, 0], expected_x, atol=1e-5)
    assert np.allclose(pred[0, :, 1], 0.0, atol=1e-5)


def test_factory_baselines() -> None:
    """Classical baselines are available without a checkpoint."""
    obs = np.zeros((2, config.T_OBS, 2))
    for name in ("kalman", "poly1", "poly2"):
        assert build_predictor(name).predict(obs).shape == (2, config.T_PRED, 2)
    with pytest.raises(ValueError):
        build_predictor("transformer")
