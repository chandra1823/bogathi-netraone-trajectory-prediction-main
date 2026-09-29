"""Synthetic intruder-track generator and ETH/UCY ingestion.

Produces a tidy DataFrame with one row per observation::

    track_id, frame, timestamp, x, y, entity_type, behavior

Motion model: constant-turn-rate + acceleration kinematics
(state = x, y, heading, speed) with a proportional steering controller that
pulls the heading toward a behaviour-specific target, bounded turn rate,
process noise on turn rate / acceleration and additive Gaussian sensor noise.

Usage::

    python generate_synthetic_tracks.py --num-tracks 3000
    python generate_synthetic_tracks.py --source ethucy --path datasets/eth/biwi_eth.txt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from predict_breach import fence_segments, first_path_crossing

logger = logging.getLogger("generator")

COLUMNS: List[str] = ["track_id", "frame", "timestamp", "x", "y", "entity_type", "behavior"]
FENCE_SEGMENTS = fence_segments(config.FENCE_LINE)


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def heading_to(src: Tuple[float, float], dst: Tuple[float, float]) -> float:
    """Heading (rad, from +x axis) of the vector src -> dst."""
    return float(np.arctan2(dst[1] - src[1], dst[0] - src[0]))


class KinematicTarget:
    """Constant-turn-rate + acceleration point target with steering control."""

    def __init__(self, x: float, y: float, heading: float, speed: float,
                 params: Dict[str, float], rng: np.random.Generator) -> None:
        """Initialise the kinematic state and entity-specific limits."""
        self.x, self.y, self.heading, self.speed = x, y, heading, speed
        self.cruise = speed
        self.p = params
        self.rng = rng

    def step(self, target_heading: float) -> Tuple[float, float]:
        """Advance one frame (DT seconds) steering toward ``target_heading``."""
        p, rng, dt = self.p, self.rng, config.DT
        # abrupt manoeuvre (rogue drones): heading jolt + new cruise speed
        if p["jump_prob"] > 0 and rng.random() < p["jump_prob"]:
            self.heading += rng.uniform(-np.pi / 2, np.pi / 2)
            self.cruise = rng.uniform(p["v_min"], p["v_max"])
        err = wrap_angle(target_heading - self.heading)
        omega = np.clip(config.HEADING_GAIN * err, -p["max_turn"], p["max_turn"])
        omega += rng.normal(0.0, p["turn_noise"])
        accel = 0.5 * (self.cruise - self.speed) + rng.normal(0.0, p["acc_noise"])
        self.speed = float(np.clip(self.speed + accel * dt, p["v_min"], p["v_max"]))
        self.heading = wrap_angle(self.heading + omega * dt)
        self.x += self.speed * np.cos(self.heading) * dt
        self.y += self.speed * np.sin(self.heading) * dt
        return self.x, self.y


# ---------------------------------------------------------------------------
# Behaviour policies: each returns (start_state, target_heading_fn)
# ---------------------------------------------------------------------------
Policy = Callable[[KinematicTarget, int], float]


def _breach_setup(rng: np.random.Generator, speed: float, n: int, angled: bool
                  ) -> Tuple[Tuple[float, float, float], Policy]:
    """Direct / angled approach that crosses the fence mid-track."""
    fx0, fx1 = config.FENCE_LINE[0][0], config.FENCE_LINE[1][0]
    x_cross = rng.uniform(fx0 + 60, fx1 - 60)
    ang = rng.uniform(np.deg2rad(20), np.deg2rad(55)) * rng.choice([-1, 1]) if angled \
        else rng.normal(0.0, np.deg2rad(5))
    k_cross = rng.integers(config.T_OBS + 6, max(config.T_OBS + 7, n - 6))
    dist = min(speed * config.DT * k_cross, config.FENCE_Y - 10.0)
    x0 = x_cross - dist * np.tan(ang)
    y0 = config.FENCE_Y - dist
    aim = (x_cross, config.FENCE_Y)
    beyond = (x_cross + 200.0 * np.tan(ang), config.FENCE_Y + 200.0)

    def policy(t: KinematicTarget, _k: int) -> float:
        goal = aim if t.y < config.FENCE_Y - 20.0 else beyond
        return heading_to((t.x, t.y), goal)

    return (x0, y0, heading_to((x0, y0), aim)), policy


def _near_miss_setup(rng: np.random.Generator, speed: float, n: int,
                     params: Dict[str, float]) -> Tuple[Tuple[float, float, float], Policy]:
    """Approach the fence, then turn away / retreat before reaching it."""
    fx0, fx1 = config.FENCE_LINE[0][0], config.FENCE_LINE[1][0]
    turn_radius = params["v_max"] / params["max_turn"]
    d_turn = max(rng.uniform(*config.NEAR_MISS_TURN_DIST), 1.3 * turn_radius + 5.0)
    x_aim = rng.uniform(fx0 + 60, fx1 - 60)
    ang = rng.uniform(-np.deg2rad(40), np.deg2rad(40))
    k_turn = rng.integers(config.T_OBS + 4, max(config.T_OBS + 5, n - 12))
    dist = min(speed * config.DT * k_turn + d_turn, config.FENCE_Y - 10.0)
    x0 = x_aim - dist * np.tan(ang)
    y0 = config.FENCE_Y - dist
    retreat = rng.uniform(np.deg2rad(-160), np.deg2rad(-20))
    state = {"turned": False}

    def policy(t: KinematicTarget, _k: int) -> float:
        if not state["turned"] and t.y >= config.FENCE_Y - d_turn:
            state["turned"] = True
        if state["turned"]:
            return retreat
        return heading_to((t.x, t.y), (x_aim, config.FENCE_Y))

    return (x0, y0, heading_to((x0, y0), (x_aim, config.FENCE_Y))), policy


def _patrol_setup(rng: np.random.Generator, speed: float, n: int
                  ) -> Tuple[Tuple[float, float, float], Policy]:
    """Move parallel to the fence along a patrol line just outside it."""
    y_line = rng.uniform(*config.PATROL_Y_RANGE)
    direction = rng.choice([0.0, np.pi])
    travel = speed * config.DT * n
    if direction == 0.0:
        x0 = rng.uniform(config.WORLD_X[0] + 20, max(config.WORLD_X[0] + 21, config.WORLD_X[1] - travel - 20))
    else:
        x0 = rng.uniform(min(config.WORLD_X[1] - 21, config.WORLD_X[0] + travel + 20), config.WORLD_X[1] - 20)
    y0 = y_line + rng.normal(0, 3.0)

    def policy(t: KinematicTarget, _k: int) -> float:
        correction = np.clip(0.02 * (y_line - t.y), -0.4, 0.4)
        return wrap_angle(direction + (correction if direction == 0.0 else -correction))

    return (x0, y0, direction), policy


def _wander_setup(rng: np.random.Generator, speed: float, n: int
                  ) -> Tuple[Tuple[float, float, float], Policy]:
    """Random-walk heading changes far from the fence."""
    x0 = rng.uniform(150, 850)
    y0 = rng.uniform(40, config.WANDER_Y_MAX - 40)
    state = {"goal": rng.uniform(-np.pi, np.pi)}

    def policy(t: KinematicTarget, k: int) -> float:
        if k % 10 == 0:
            state["goal"] = wrap_angle(state["goal"] + rng.normal(0, 1.0))
        # soft walls keep wanderers inside the far field
        if t.y > config.WANDER_Y_MAX:
            state["goal"] = -np.pi / 2
        elif t.y < 20:
            state["goal"] = np.pi / 2
        elif t.x < 50:
            state["goal"] = 0.0
        elif t.x > 950:
            state["goal"] = np.pi
        return state["goal"]

    return (x0, y0, state["goal"]), policy


def simulate_track(entity: str, behavior: str, rng: np.random.Generator
                   ) -> Optional[np.ndarray]:
    """Simulate one noisy track; returns [N, 2] or None if it violates its behaviour."""
    params = config.ENTITY_PARAMS[entity]
    n = int(rng.integers(config.TRACK_LEN_RANGE[0], config.TRACK_LEN_RANGE[1] + 1))
    speed = float(rng.uniform(params["v_min"], params["v_max"]))
    if behavior in config.BREACH_BEHAVIORS:
        start, policy = _breach_setup(rng, speed, n, angled=behavior == "angled_breach")
    elif behavior == "near_miss":
        start, policy = _near_miss_setup(rng, speed, n, params)
    elif behavior == "patrol":
        start, policy = _patrol_setup(rng, speed, n)
    elif behavior == "wander":
        start, policy = _wander_setup(rng, speed, n)
    else:
        raise ValueError(f"unknown behaviour {behavior!r}")

    tgt = KinematicTarget(*start, speed=speed, params=params, rng=rng)
    pts = [(tgt.x, tgt.y)]
    for k in range(1, n):
        pts.append(tgt.step(policy(tgt, k)))
    xy = np.asarray(pts, dtype=np.float64)
    sigma = rng.uniform(*config.SENSOR_NOISE_RANGE)
    xy = xy + rng.normal(0.0, sigma, size=xy.shape)

    # keep the in-world prefix only
    inside = ((xy[:, 0] >= config.WORLD_X[0]) & (xy[:, 0] <= config.WORLD_X[1])
              & (xy[:, 1] >= config.WORLD_Y[0]) & (xy[:, 1] <= config.WORLD_Y[1]))
    cut = int(np.argmin(inside)) if not inside.all() else len(xy)
    xy = xy[:cut]
    if len(xy) < config.T_OBS + config.T_PRED or xy[0, 1] >= config.FENCE_Y:
        return None

    hit = first_path_crossing(xy, FENCE_SEGMENTS)
    if behavior in config.BREACH_BEHAVIORS:
        # the crossing must leave room for at least T_OBS frames of clean history
        if hit is None or hit[0] < config.T_OBS:
            return None
    elif hit is not None:
        return None
    return xy


def generate_dataset(num_tracks: int = config.NUM_TRACKS, seed: int = config.SEED) -> pd.DataFrame:
    """Generate ``num_tracks`` synthetic tracks with a balanced behaviour mix."""
    rng = np.random.default_rng(seed)
    entities = list(config.ENTITY_PROBS)
    behaviors = list(config.BEHAVIOR_PROBS)
    e_p = np.array([config.ENTITY_PROBS[e] for e in entities])
    b_p = np.array([config.BEHAVIOR_PROBS[b] for b in behaviors])
    rows: List[pd.DataFrame] = []
    rejected = 0
    for tid in range(num_tracks):
        entity = str(rng.choice(entities, p=e_p / e_p.sum()))
        behavior = str(rng.choice(behaviors, p=b_p / b_p.sum()))
        xy = None
        for _ in range(config.MAX_REJECTION_TRIES):
            xy = simulate_track(entity, behavior, rng)
            if xy is not None:
                break
            rejected += 1
        if xy is None:
            raise RuntimeError(f"could not simulate {entity}/{behavior} after retries")
        frames = np.arange(len(xy))
        rows.append(pd.DataFrame({
            "track_id": tid, "frame": frames, "timestamp": np.round(frames * config.DT, 3),
            "x": xy[:, 0], "y": xy[:, 1], "entity_type": entity, "behavior": behavior,
        }))
    logger.info("rejection-sampled %d invalid candidate tracks", rejected)
    return pd.concat(rows, ignore_index=True)[COLUMNS]


def load_ethucy(path: Path) -> pd.DataFrame:
    """Ingest an ETH/UCY text file (``frame ped_id x y``) into the common schema.

    Frame numbers are re-indexed to consecutive integers (ETH/UCY annotate every
    10th video frame at 2.5 Hz). Tracks with temporal gaps are split so that each
    ``track_id`` is contiguous.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["frame", "ped_id", "x", "y"],
                      engine="python")
    uniq = np.sort(raw["frame"].unique())
    step = float(np.min(np.diff(uniq))) if len(uniq) > 1 else 1.0
    raw["frame"] = np.round((raw["frame"] - uniq[0]) / step).astype(int)
    raw = raw.sort_values(["ped_id", "frame"])
    out: List[pd.DataFrame] = []
    next_id = 0
    for _, grp in raw.groupby("ped_id"):
        seg_id = np.cumsum(np.diff(grp["frame"].to_numpy(), prepend=grp["frame"].iloc[0]) > 1)
        for _, seg in grp.groupby(seg_id):
            f = seg["frame"].to_numpy() - seg["frame"].iloc[0]
            out.append(pd.DataFrame({
                "track_id": next_id, "frame": f, "timestamp": np.round(f * config.DT, 3),
                "x": seg["x"].to_numpy(float), "y": seg["y"].to_numpy(float),
                "entity_type": "pedestrian", "behavior": "unknown",
            }))
            next_id += 1
    return pd.concat(out, ignore_index=True)[COLUMNS]


def summarize(df: pd.DataFrame) -> None:
    """Log summary statistics of a track DataFrame."""
    g = df.sort_values(["track_id", "frame"]).groupby("track_id")
    step = np.hypot(g["x"].diff(), g["y"].diff()) / config.DT
    df = df.assign(speed=step)
    lengths = g.size()
    logger.info("rows=%d tracks=%d  length: mean=%.1f min=%d max=%d frames",
                len(df), df["track_id"].nunique(), lengths.mean(), lengths.min(), lengths.max())
    per_track = df.groupby("track_id")[["entity_type", "behavior"]].first()
    logger.info("tracks per entity:\n%s", per_track["entity_type"].value_counts().to_string())
    logger.info("tracks per behaviour:\n%s", per_track["behavior"].value_counts().to_string())
    logger.info("measured speed [m/s] per entity (incl. sensor noise):\n%s",
                df.groupby("entity_type")["speed"].describe()[["mean", "std", "min", "max"]]
                .round(2).to_string())
    breach = sum(first_path_crossing(t[["x", "y"]].to_numpy(), FENCE_SEGMENTS) is not None
                 for _, t in df.groupby("track_id"))
    logger.info("tracks crossing the fence: %d / %d (%.1f%%)", breach, len(per_track),
                100.0 * breach / len(per_track))
    logger.info("x range [%.1f, %.1f], y range [%.1f, %.1f]",
                df.x.min(), df.x.max(), df.y.min(), df.y.max())


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "ethucy"], default="synthetic")
    ap.add_argument("--path", type=Path, default=None, help="ETH/UCY txt file (for --source ethucy)")
    ap.add_argument("--num-tracks", type=int, default=config.NUM_TRACKS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.TRACKS_CSV)
    return ap.parse_args()


def main() -> None:
    """CLI entry point."""
    config.setup_logging()
    args = parse_args()
    if args.source == "ethucy":
        if args.path is None:
            raise SystemExit("--path is required with --source ethucy")
        df = load_ethucy(args.path)
    else:
        df = generate_dataset(args.num_tracks, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.3f")
    logger.info("saved %s", args.out)
    summarize(df)


if __name__ == "__main__":
    main()
