"""Central configuration for the Netra-One trajectory forecasting project.

Every constant used by the pipeline (time base, world geometry, fence, generator
parameters, model / training hyper-parameters, file paths, plot styles) lives
here so that the rest of the code never hard-codes values.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to the repository root -> works on Windows, Linux and Colab)
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
CHECKPOINT_DIR: Path = ROOT_DIR / "checkpoints"
OUTPUT_DIR: Path = ROOT_DIR / "sample_outputs"
TRACKS_CSV: Path = DATA_DIR / "tracks.csv"
BEST_MODEL_PATH: Path = CHECKPOINT_DIR / "best_model.pt"
NORM_STATS_PATH: Path = CHECKPOINT_DIR / "norm_stats.json"
ALERT_LOG_PATH: Path = OUTPUT_DIR / "alerts_log.txt"

# ---------------------------------------------------------------------------
# Time base and horizons
# ---------------------------------------------------------------------------
DT: float = 0.4          # seconds per frame (2.5 Hz, same as ETH/UCY)
T_OBS: int = 8           # observed frames (3.2 s of history)
T_PRED: int = 12         # predicted frames -> 4.8 s horizon
SEED: int = 42

# ---------------------------------------------------------------------------
# World and perimeter geometry (units: metres)
# ---------------------------------------------------------------------------
WORLD_X: Tuple[float, float] = (0.0, 1000.0)
WORLD_Y: Tuple[float, float] = (0.0, 600.0)
FENCE_LINE: List[Tuple[float, float]] = [(100.0, 450.0), (900.0, 450.0)]
FENCE_Y: float = 450.0   # restricted zone is y > FENCE_Y (between the fence posts)
RESTRICTED_ZONE: List[Tuple[float, float]] = [
    (100.0, 450.0), (900.0, 450.0), (900.0, 600.0), (100.0, 600.0)
]
GEOM_EPS: float = 1e-9   # tolerance for collinearity tests

# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
NUM_TRACKS: int = 3000
TRACK_LEN_RANGE: Tuple[int, int] = (40, 80)          # frames per track
SENSOR_NOISE_RANGE: Tuple[float, float] = (0.3, 1.0) # Gaussian sigma [m]

# speed range [m/s], max turn-rate [rad/s], turn-noise std [rad/s],
# accel-noise std [m/s^2], probability per frame of an abrupt manoeuvre
ENTITY_PARAMS: Dict[str, Dict[str, float]] = {
    "pedestrian": dict(v_min=1.0, v_max=2.0, max_turn=0.8, turn_noise=0.15,
                       acc_noise=0.15, jump_prob=0.0),
    "runner":     dict(v_min=3.0, v_max=6.0, max_turn=0.7, turn_noise=0.12,
                       acc_noise=0.4, jump_prob=0.0),
    "vehicle":    dict(v_min=5.0, v_max=15.0, max_turn=0.25, turn_noise=0.03,
                       acc_noise=0.6, jump_prob=0.0),
    "drone":      dict(v_min=4.0, v_max=20.0, max_turn=1.6, turn_noise=0.25,
                       acc_noise=1.5, jump_prob=0.06),
}
ENTITY_PROBS: Dict[str, float] = {
    "pedestrian": 0.25, "runner": 0.25, "vehicle": 0.25, "drone": 0.25,
}
BEHAVIOR_PROBS: Dict[str, float] = {
    "direct_breach": 0.2, "angled_breach": 0.2, "patrol": 0.2,
    "near_miss": 0.2, "wander": 0.2,
}
BREACH_BEHAVIORS: Tuple[str, ...] = ("direct_breach", "angled_breach")
HEADING_GAIN: float = 1.5        # proportional steering gain toward target heading
NEAR_MISS_TURN_DIST: Tuple[float, float] = (15.0, 60.0)  # distance to fence at turn-away [m]
PATROL_Y_RANGE: Tuple[float, float] = (390.0, 440.0)
WANDER_Y_MAX: float = 300.0
MAX_REJECTION_TRIES: int = 50

# ---------------------------------------------------------------------------
# Dataset / features
# ---------------------------------------------------------------------------
SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
TRAIN_STRIDE: int = 1
VAL_STRIDE: int = 2              # validation windows used for early stopping
EVAL_STRIDE: int = 1             # test windows used for the final benchmark
SAMPLES_PER_EPOCH: int = 32768   # random subset of train windows drawn each epoch (CPU budget)
TRAIN_EVAL_SAMPLES: int = 8192   # fixed train subset scored free-running each epoch (loss curve)
FEATURE_NAMES: Tuple[str, ...] = ("x_norm", "y_norm", "dx", "dy", "vx", "vy")
NUM_FEATURES: int = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Seq2seq model + training
# ---------------------------------------------------------------------------
HIDDEN_SIZE: int = 128
NUM_LAYERS: int = 2
DROPOUT: float = 0.1
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 256
EPOCHS: int = 50
PATIENCE: int = 10
GRAD_CLIP: float = 1.0
TF_START: float = 0.5
TF_END: float = 0.0
LR_FACTOR: float = 0.5
LR_PATIENCE: int = 3
DEFAULT_LOSS: str = "smoothl1"

# ---------------------------------------------------------------------------
# Kalman filter noise model
# ---------------------------------------------------------------------------
KF_SIGMA_ACCEL: float = 1.5      # white-noise acceleration std [m/s^2]
KF_SIGMA_MEAS: float = 0.7       # measurement noise std [m]
KF_INIT_VEL_VAR: float = 25.0    # initial velocity variance [(m/s)^2]

# ---------------------------------------------------------------------------
# Alerting / visualisation
# ---------------------------------------------------------------------------
ALERT_CONSECUTIVE: int = 1       # frames an alert must persist before it fires
DEMO_DELAY: float = 0.15         # seconds between frames in --demo
FIG_DPI: int = 150
PLOT_FENCE_MARGIN: float = 100.0 # zoomed plots include the fence if a target is this close [m]
COLORS: Dict[str, str] = {
    "history": "tab:blue", "future": "tab:green", "pred": "red",
    "fence": "red", "zone": "#ffcccc", "kalman": "tab:orange",
    "poly1": "tab:purple", "poly2": "tab:brown", "lstm": "red", "gru": "tab:pink",
}


def checkpoint_path(cell: str) -> Path:
    """Checkpoint file for a recurrent cell type (LSTM is the primary model)."""
    return BEST_MODEL_PATH if cell == "lstm" else CHECKPOINT_DIR / f"best_model_{cell}.pt"


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard requirement
        pass


def get_device() -> "torch.device":  # noqa: F821
    """Return CUDA if available (e.g. Colab T4), otherwise CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once and make stdout UTF-8 safe (for the ° sign)."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # pragma: no cover
            pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
