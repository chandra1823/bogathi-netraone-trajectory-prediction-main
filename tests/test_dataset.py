"""Tests for windowing, features, splits and normalisation (Part A)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from dataset_loader import (NormStats, TrajectoryDataset, compute_features, make_windows,
                            relative_future, split_track_ids)


def test_generator_schema_and_start_outside(small_tracks: pd.DataFrame) -> None:
    """Generator emits the documented schema and every track starts outside the zone."""
    assert list(small_tracks.columns) == ["track_id", "frame", "timestamp", "x", "y",
                                          "entity_type", "behavior"]
    first = small_tracks.sort_values("frame").groupby("track_id").first()
    assert (first["y"] < config.FENCE_Y).all()
    assert set(first["entity_type"]) <= set(config.ENTITY_PARAMS)


def test_window_shapes(small_tracks: pd.DataFrame) -> None:
    """Windows have [N, T_OBS, 2] / [N, T_PRED, 2] shapes and consecutive frames."""
    ids = small_tracks.track_id.unique()[:10]
    w = make_windows(small_tracks, ids, stride=1)
    expected = sum(max(0, len(g) - config.T_OBS - config.T_PRED + 1)
                   for _, g in small_tracks[small_tracks.track_id.isin(ids)].groupby("track_id"))
    assert w.obs_xy.shape == (expected, config.T_OBS, 2)
    assert w.fut_xy.shape == (expected, config.T_PRED, 2)
    assert w.obs_t.shape == (expected, config.T_OBS)
    assert np.allclose(np.diff(w.obs_t, axis=1), config.DT)
    assert len(w.track_id) == len(w.entity) == len(w.behavior) == expected


def test_no_track_leaks_across_splits(small_tracks: pd.DataFrame) -> None:
    """Every track id belongs to exactly one of train / val / test."""
    tr, va, te = split_track_ids(small_tracks)
    s_tr, s_va, s_te = set(tr), set(va), set(te)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
    assert s_tr | s_va | s_te == set(small_tracks.track_id.unique())
    # windows built from each split carry only that split's ids
    assert set(make_windows(small_tracks, te).track_id) <= s_te
    assert set(make_windows(small_tracks, tr).track_id) <= s_tr


def test_split_ratios(small_tracks: pd.DataFrame) -> None:
    """Split sizes are close to 70/15/15."""
    tr, va, te = split_track_ids(small_tracks)
    n = len(tr) + len(va) + len(te)
    assert abs(len(tr) / n - 0.70) < 0.05
    assert abs(len(va) / n - 0.15) < 0.05


def test_features_values() -> None:
    """Features encode normalised position, displacement and velocity."""
    obs = np.stack([np.arange(config.T_OBS) * 2.0 + 100, np.full(config.T_OBS, 300.0)], axis=-1)
    f = compute_features(obs[None])
    assert f.shape == (1, config.T_OBS, config.NUM_FEATURES)
    assert np.isfinite(f).all()
    assert np.allclose(f[0, :, 0], obs[:, 0] / config.WORLD_X[1])
    assert np.allclose(f[0, :, 1], 0.5)
    assert np.allclose(f[0, :, 2], 2.0) and np.allclose(f[0, :, 3], 0.0)
    assert np.allclose(f[0, :, 4], 2.0 / config.DT)


def test_relative_target_is_translation_invariant() -> None:
    """Shifting a window leaves the relative target unchanged."""
    rng = np.random.default_rng(0)
    obs, fut = rng.normal(size=(4, config.T_OBS, 2)), rng.normal(size=(4, config.T_PRED, 2))
    shift = np.array([123.0, -45.0])
    assert np.allclose(relative_future(obs, fut), relative_future(obs + shift, fut + shift))


def test_norm_stats_and_dataset(small_tracks: pd.DataFrame, tmp_path) -> None:
    """Train-only z-scoring gives ~zero-mean features; dataset returns 4 tensors."""
    tr, _, _ = split_track_ids(small_tracks)
    w = make_windows(small_tracks, tr)
    stats = NormStats.fit(w.obs_xy, w.fut_xy)
    z = stats.normalize(compute_features(w.obs_xy)).reshape(-1, config.NUM_FEATURES)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
    stats.save(tmp_path / "s.json")
    assert NormStats.load(tmp_path / "s.json") == stats
    ds = TrajectoryDataset(w, stats)
    feats, obs_xy, fut_xy, tid = ds[0]
    assert feats.shape == (config.T_OBS, config.NUM_FEATURES)
    assert obs_xy.shape == (config.T_OBS, 2) and fut_xy.shape == (config.T_PRED, 2)
    batch = ds[[0, 1, 2]]
    assert batch[0].shape == (3, config.T_OBS, config.NUM_FEATURES)
