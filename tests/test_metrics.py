"""Tests for ADE/FDE metrics and the alert confusion-matrix maths."""
from __future__ import annotations

import numpy as np
import pytest

from evaluate import ade_fde
from predict_breach import _prf


def test_ade_fde_toy() -> None:
    """Hand-computed toy example.

    Sample 1 errors per step: 0, 1, 2 -> ADE 1, FDE 2
    Sample 2 errors per step: 3, 4, 5 (3-4-5 offsets) -> ADE 4, FDE 5
    Batch ADE = 2.5, FDE = 3.5.
    """
    gt = np.zeros((2, 3, 2))
    pred = np.array([[[0, 0], [1, 0], [0, 2]],
                     [[3, 0], [0, 4], [3, 4]]], dtype=float)
    ade, fde = ade_fde(pred, gt)
    assert ade == pytest.approx(2.5)
    assert fde == pytest.approx(3.5)


def test_ade_fde_perfect() -> None:
    """Perfect predictions give zero error."""
    x = np.random.default_rng(0).normal(size=(4, 12, 2))
    assert ade_fde(x, x) == (0.0, 0.0)


def test_prf() -> None:
    """Precision / recall / F1 / false-alarm-rate from counts."""
    r = _prf(tp=8, fp=2, fn=2, tn=88)
    assert r["precision"] == pytest.approx(0.8)
    assert r["recall"] == pytest.approx(0.8)
    assert r["f1"] == pytest.approx(0.8)
    assert r["false_alarm_rate"] == pytest.approx(2 / 90)
