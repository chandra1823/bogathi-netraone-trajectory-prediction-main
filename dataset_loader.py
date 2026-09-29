"""Part A: windowing, feature engineering, track-level splits and DataLoaders.

Each sample is a sliding window over one track:

* observed history ``obs_xy``  [T_OBS, 2] (+ timestamps)
* ground-truth future ``fut_xy`` [T_PRED, 2]
* per-step features ``obs_feats`` [T_OBS, F] =
  ``[x_norm, y_norm, dx, dy, vx, vy]`` (z-scored with TRAIN statistics)

The network target is the future displacement relative to the last observed
point (translation invariant); absolute coordinates are restored for metrics.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

import config

logger = logging.getLogger("dataset")


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------
def load_tracks(path: Path = config.TRACKS_CSV) -> pd.DataFrame:
    """Load the tracks CSV produced by ``generate_synthetic_tracks.py``."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run generate_synthetic_tracks.py first")
    df = pd.read_csv(path)
    if "behavior" not in df.columns:
        df["behavior"] = "unknown"
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def split_track_ids(df: pd.DataFrame, ratios: Tuple[float, float, float] = config.SPLIT_RATIOS,
                    seed: int = config.SEED) -> Tuple[List[int], List[int], List[int]]:
    """Split track ids 70/15/15, stratified by behaviour, so no track spans two splits."""
    rng = np.random.default_rng(seed)
    per_track = df.groupby("track_id")["behavior"].first()
    train, val, test = [], [], []
    for _, ids in per_track.groupby(per_track):
        arr = np.array(sorted(ids.index.tolist()))
        rng.shuffle(arr)
        n_tr = int(round(ratios[0] * len(arr)))
        n_va = int(round(ratios[1] * len(arr)))
        train += arr[:n_tr].tolist()
        val += arr[n_tr:n_tr + n_va].tolist()
        test += arr[n_tr + n_va:].tolist()
    return sorted(train), sorted(val), sorted(test)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
@dataclass
class WindowSet:
    """A set of sliding windows (numpy arrays) plus per-window metadata."""

    obs_xy: np.ndarray       # [N, T_OBS, 2]
    obs_t: np.ndarray        # [N, T_OBS]
    fut_xy: np.ndarray       # [N, T_PRED, 2]
    track_id: np.ndarray     # [N]
    entity: np.ndarray       # [N] (str)
    behavior: np.ndarray     # [N] (str)

    def __len__(self) -> int:
        """Number of windows."""
        return len(self.obs_xy)


def make_windows(df: pd.DataFrame, track_ids: Sequence[int], stride: int = 1,
                 t_obs: int = config.T_OBS, t_pred: int = config.T_PRED) -> WindowSet:
    """Slide a (t_obs + t_pred) window over every track in ``track_ids``."""
    sub = df[df.track_id.isin(set(track_ids))]
    total = t_obs + t_pred
    obs, obs_t, fut, tids, ents, behs = [], [], [], [], [], []
    for tid, grp in sub.groupby("track_id", sort=True):
        xy = grp[["x", "y"]].to_numpy(np.float64)
        ts = grp["timestamp"].to_numpy(np.float64)
        n = len(xy)
        if n < total:
            continue
        starts = np.arange(0, n - total + 1, stride)
        idx = starts[:, None] + np.arange(total)[None, :]
        win = xy[idx]
        obs.append(win[:, :t_obs])
        fut.append(win[:, t_obs:])
        obs_t.append(ts[idx][:, :t_obs])
        tids.append(np.full(len(starts), tid))
        ents.append(np.full(len(starts), grp["entity_type"].iloc[0], dtype=object))
        behs.append(np.full(len(starts), grp["behavior"].iloc[0], dtype=object))
    if not obs:
        empty = np.zeros((0, t_obs, 2))
        return WindowSet(empty, np.zeros((0, t_obs)), np.zeros((0, t_pred, 2)),
                         np.zeros(0, int), np.zeros(0, object), np.zeros(0, object))
    return WindowSet(np.concatenate(obs), np.concatenate(obs_t), np.concatenate(fut),
                     np.concatenate(tids), np.concatenate(ents), np.concatenate(behs))


# ---------------------------------------------------------------------------
# Feature engineering and normalisation
# ---------------------------------------------------------------------------
def compute_features(obs_xy: np.ndarray) -> np.ndarray:
    """Raw per-step features ``[x_norm, y_norm, dx, dy, vx, vy]`` -> [B, T_OBS, 6].

    * ``x_norm, y_norm``: centroid scaled into [0, 1] with the world bounds
    * ``dx, dy``: displacement from the previous frame (the first step copies
      the second so the window stays exactly T_OBS long)
    * ``vx, vy``: instantaneous velocity = displacement / DT
    """
    obs_xy = np.asarray(obs_xy, dtype=np.float64)
    squeeze = obs_xy.ndim == 2
    if squeeze:
        obs_xy = obs_xy[None]
    lo = np.array([config.WORLD_X[0], config.WORLD_Y[0]])
    hi = np.array([config.WORLD_X[1], config.WORLD_Y[1]])
    xy_norm = (obs_xy - lo) / (hi - lo)
    d = np.diff(obs_xy, axis=1)
    d = np.concatenate([d[:, :1], d], axis=1)
    v = d / config.DT
    feats = np.concatenate([xy_norm, d, v], axis=-1)
    return feats[0] if squeeze else feats


def last_displacement(obs_xy: np.ndarray) -> np.ndarray:
    """Last observed per-step displacement [B, 2] (decoder seed input)."""
    return np.asarray(obs_xy)[:, -1] - np.asarray(obs_xy)[:, -2]


def relative_future(obs_xy: np.ndarray, fut_xy: np.ndarray) -> np.ndarray:
    """Future positions relative to the last observed point [B, T_PRED, 2]."""
    return np.asarray(fut_xy) - np.asarray(obs_xy)[:, -1:, :]


@dataclass
class NormStats:
    """Z-score statistics for input features and a scalar scale for displacements."""

    feat_mean: List[float]
    feat_std: List[float]
    disp_scale: float

    @classmethod
    def fit(cls, obs_xy: np.ndarray, fut_xy: np.ndarray) -> "NormStats":
        """Fit statistics on TRAIN windows only."""
        feats = compute_features(obs_xy).reshape(-1, config.NUM_FEATURES)
        mean, std = feats.mean(axis=0), feats.std(axis=0)
        std[std < 1e-6] = 1.0
        steps = np.diff(np.concatenate([obs_xy[:, -1:], fut_xy], axis=1), axis=1)
        scale = float(np.sqrt(np.mean(steps ** 2)))
        return cls(mean.tolist(), std.tolist(), max(scale, 1e-3))

    def normalize(self, feats: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation."""
        return (feats - np.asarray(self.feat_mean)) / np.asarray(self.feat_std)

    def save(self, path: Path = config.NORM_STATS_PATH) -> None:
        """Save to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = config.NORM_STATS_PATH) -> "NormStats":
        """Load from JSON."""
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """Returns ``(obs_feats, obs_xy, fut_xy, track_id)`` float32 tensors."""

    def __init__(self, windows: WindowSet, stats: NormStats) -> None:
        """Pre-compute normalised features for all windows."""
        self.obs_feats = torch.from_numpy(
            stats.normalize(compute_features(windows.obs_xy)).astype(np.float32))
        self.obs_xy = torch.from_numpy(windows.obs_xy.astype(np.float32))
        self.fut_xy = torch.from_numpy(windows.fut_xy.astype(np.float32))
        self.track_id = torch.from_numpy(windows.track_id.astype(np.int64))

    def __len__(self) -> int:
        """Number of windows."""
        return len(self.obs_xy)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One sample, or a whole batch when ``i`` is a list of indices (fast path)."""
        if isinstance(i, list):
            i = torch.as_tensor(i)
        return self.obs_feats[i], self.obs_xy[i], self.fut_xy[i], self.track_id[i]


def make_loader(ds: TrajectoryDataset, batch_size: int, shuffle: bool,
                num_samples: Optional[int] = None, pin: bool = False,
                seed: int = config.SEED) -> DataLoader:
    """DataLoader that indexes whole batches at once (avoids per-item collation).

    With ``shuffle`` and ``num_samples`` each epoch draws a fresh random subset
    of ``num_samples`` windows (without replacement).
    """
    if shuffle:
        n = min(num_samples or len(ds), len(ds))
        base = RandomSampler(ds, num_samples=n, generator=torch.Generator().manual_seed(seed))
    else:
        base = SequentialSampler(ds)
    sampler = BatchSampler(base, batch_size=batch_size, drop_last=False)
    return DataLoader(ds, sampler=sampler, batch_size=None, num_workers=0, pin_memory=pin)


def build_dataloaders(csv_path: Path = config.TRACKS_CSV, batch_size: int = config.BATCH_SIZE,
                      train_stride: int = config.TRAIN_STRIDE, val_stride: int = config.VAL_STRIDE,
                      eval_stride: int = config.EVAL_STRIDE,
                      samples_per_epoch: Optional[int] = config.SAMPLES_PER_EPOCH,
                      device: Optional[torch.device] = None, save_stats: bool = True
                      ) -> Tuple[Dict[str, DataLoader], NormStats, Dict[str, WindowSet]]:
    """Build train / val / test DataLoaders with train-only normalisation stats."""
    df = load_tracks(csv_path)
    tr_ids, va_ids, te_ids = split_track_ids(df)
    windows = {
        "train": make_windows(df, tr_ids, train_stride),
        "val": make_windows(df, va_ids, val_stride),
        "test": make_windows(df, te_ids, eval_stride),
    }
    stats = NormStats.fit(windows["train"].obs_xy, windows["train"].fut_xy)
    if save_stats:
        stats.save()
    pin = device is not None and device.type == "cuda"
    loaders = {
        name: make_loader(TrajectoryDataset(w, stats), batch_size if name == "train" else 4096,
                          shuffle=name == "train",
                          num_samples=samples_per_epoch if name == "train" else None, pin=pin)
        for name, w in windows.items()
    }
    logger.info("tracks train/val/test = %d/%d/%d  windows = %d/%d/%d",
                len(tr_ids), len(va_ids), len(te_ids),
                len(windows["train"]), len(windows["val"]), len(windows["test"]))
    return loaders, stats, windows
