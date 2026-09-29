"""Tests for the early-warning trigger: time-to-breach, intercept vector, alert text."""
from __future__ import annotations

import numpy as np
import pytest

import config
from predict_breach import Alert, check_breach, intercept_vector, time_to_breach


def _approach(step: float = 4.0, y_last: float = 440.0) -> tuple[np.ndarray, np.ndarray]:
    """Straight northbound approach at x=500 moving ``step`` m per frame."""
    y_obs = y_last - step * np.arange(config.T_OBS)[::-1]
    obs = np.stack([np.full(config.T_OBS, 500.0), y_obs], axis=-1)
    y_fut = y_last + step * np.arange(1, config.T_PRED + 1)
    fut = np.stack([np.full(config.T_PRED, 500.0), y_fut], axis=-1)
    return obs, fut


def test_time_to_breach_formula() -> None:
    """t_breach = (k + alpha) * DT."""
    assert time_to_breach(2, 0.5) == pytest.approx(2.5 * config.DT)
    assert time_to_breach(0, 0.0) == 0.0


def test_check_breach_timing_and_point() -> None:
    """Last obs at y=440 moving 4 m/frame -> crossing at k=2, alpha=0.5 -> 1.0 s."""
    obs, fut = _approach()
    alert = check_breach(104, obs, fut)
    assert alert is not None
    assert alert.segment_index == 2 and alert.alpha == pytest.approx(0.5)
    assert alert.time_to_breach == pytest.approx(1.0)
    assert alert.crossing_point == pytest.approx((500.0, 450.0))
    assert alert.heading_deg == pytest.approx(90.0)
    assert alert.speed == pytest.approx(4.0 / config.DT)


def test_intercept_vector_angle_range() -> None:
    """Heading is in [0, 360) measured from +x, speed = length / DT."""
    h, v = intercept_vector((0, 0), (-1, -1))
    assert h == pytest.approx(225.0)
    assert v == pytest.approx(np.sqrt(2) / config.DT)
    h, _ = intercept_vector((0, 0), (1, -1e-9))
    assert 0.0 <= h < 360.0


def test_alert_format_exact() -> None:
    """The alert line matches the mandated format character for character."""
    a = Alert(104, 2.4, (512.3, 450.0), 42.0, 3.8, 5, 0.9)
    assert a.format() == ("[PREDICTIVE ALERT] Target ID #104 trajectory intersects Restricted Fence "
                          "in 2.4 seconds! Intercept Vector: [Angle 42°, Velocity 3.8 m/s]")


def test_no_alert_when_path_misses() -> None:
    """A path moving parallel to the fence raises no alert."""
    obs = np.stack([np.linspace(200, 228, config.T_OBS), np.full(config.T_OBS, 420.0)], axis=-1)
    fut = np.stack([np.linspace(232, 276, config.T_PRED), np.full(config.T_PRED, 420.0)], axis=-1)
    assert check_breach(1, obs, fut) is None


def test_no_alert_beyond_horizon() -> None:
    """A slow target whose crossing lies beyond T_PRED raises no alert."""
    obs, fut = _approach(step=0.5, y_last=400.0)
    assert check_breach(1, obs, fut) is None


def test_no_alert_if_history_inside() -> None:
    """Targets already inside the restricted zone are not evaluated."""
    obs, fut = _approach(y_last=470.0)
    assert check_breach(1, obs, fut) is None


def test_polygon_fence() -> None:
    """Breach logic works for a closed polygon perimeter."""
    square = [(480.0, 445.0), (520.0, 445.0), (520.0, 500.0), (480.0, 500.0)]
    obs, fut = _approach()
    alert = check_breach(7, obs, fut, fence=square, closed=True, zone=square)
    assert alert is not None
    assert alert.crossing_point == pytest.approx((500.0, 445.0))
    assert alert.time_to_breach == pytest.approx(1.25 * config.DT)
