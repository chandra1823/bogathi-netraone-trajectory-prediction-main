"""Tests for the classical baselines."""
from __future__ import annotations

import numpy as np

import config
from models.kalman_baseline import ConstantVelocityKF, PolynomialExtrapolator


def _straight_line(b: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free constant-velocity tracks split into (obs, fut)."""
    rng = np.random.default_rng(1)
    p0 = rng.uniform(0, 400, size=(b, 1, 2))
    v = rng.uniform(-10, 10, size=(b, 1, 2))
    t = np.arange(config.T_OBS + config.T_PRED)[None, :, None] * config.DT
    xy = p0 + v * t
    return xy[:, :config.T_OBS], xy[:, config.T_OBS:]


def test_kalman_exact_on_straight_line() -> None:
    """With noise-free constant-velocity input the CV Kalman filter is exact."""
    obs, fut = _straight_line()
    pred = ConstantVelocityKF().predict(obs)
    assert pred.shape == fut.shape
    assert np.allclose(pred, fut, atol=1e-8)


def test_kalman_velocity_estimate() -> None:
    """Filtered velocity equals the true velocity on a straight line."""
    obs, fut = _straight_line(1)
    state, cov = ConstantVelocityKF().filter(obs)
    v_true = (obs[0, 1] - obs[0, 0]) / config.DT
    assert np.allclose(state[0, 2:], v_true)
    assert cov.shape == (1, 4, 4)
    assert np.allclose(cov[0], cov[0].T)


def test_kalman_matrices() -> None:
    """F, H, Q, R have the expected shapes; Q and R are symmetric PSD."""
    kf = ConstantVelocityKF()
    assert kf.F.shape == (4, 4) and kf.H.shape == (2, 4)
    for m in (kf.Q, kf.R):
        assert np.allclose(m, m.T)
        assert np.all(np.linalg.eigvalsh(m) >= -1e-12)


def test_polynomial_exact_on_line_and_parabola() -> None:
    """Degree-1 is exact on lines; degree-2 is exact on constant-acceleration paths."""
    obs, fut = _straight_line()
    assert np.allclose(PolynomialExtrapolator(1).predict(obs), fut, atol=1e-6)
    t = np.arange(config.T_OBS + config.T_PRED) * config.DT
    xy = np.stack([3 * t + 0.5 * t ** 2, 10 - 2 * t ** 2], axis=-1)[None]
    pred = PolynomialExtrapolator(2).predict(xy[:, :config.T_OBS])
    assert np.allclose(pred, xy[:, config.T_OBS:], atol=1e-6)
