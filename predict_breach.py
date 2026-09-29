"""Predictive perimeter-breach early warning + trajectory visualisation.

Pipeline: observed track -> trajectory forecaster (LSTM / Kalman / ...) ->
geometric path/fence intersection -> time-to-breach + intercept vector ->
``[PREDICTIVE ALERT]`` line + alerts log + plots.

Examples::

    python predict_breach.py --model lstm --num-samples 9
    python predict_breach.py --model kalman --track-id 104
    python predict_breach.py --model lstm --demo
    python predict_breach.py --model lstm --evaluate-alerts
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

import config

if TYPE_CHECKING:  # heavy imports only for type checking
    import pandas as pd
    from models.trajectory_model import TrajectoryPredictor

logger = logging.getLogger("breach")

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


# ===========================================================================
# 1. Geometry primitives (written from scratch)
# ===========================================================================
def cross(o: Point, a: Point, b: Point) -> float:
    """2-D cross product of (a - o) x (b - o)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def orientation(p: Point, q: Point, r: Point, eps: float = config.GEOM_EPS) -> int:
    """Orientation of the ordered triplet (p, q, r).

    Returns 0 if collinear, 1 if counter-clockwise, -1 if clockwise.
    """
    val = cross(p, q, r)
    if abs(val) <= eps:
        return 0
    return 1 if val > 0 else -1


def on_segment(p: Point, q: Point, r: Point, eps: float = config.GEOM_EPS) -> bool:
    """True if point ``q`` lies on segment ``pr`` given that p, q, r are collinear."""
    return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Orientation-based test whether segments p1p2 and q1q2 intersect.

    Handles the general proper-crossing case as well as the collinear /
    touching-endpoint special cases.
    """
    o1, o2 = orientation(p1, p2, q1), orientation(p1, p2, q2)
    o3, o4 = orientation(q1, q2, p1), orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True
    return False


def segment_intersection(p1: Point, p2: Point, q1: Point, q2: Point
                         ) -> Optional[Tuple[float, Point]]:
    """Earliest intersection of moving segment p1->p2 with fence segment q1q2.

    Returns ``(alpha, point)`` where ``alpha`` in [0, 1] is the fraction along
    p1->p2 at which the first contact happens, or ``None`` if they don't meet.
    For collinear overlaps the first overlapping point along p1->p2 is used.
    """
    if not segments_intersect(p1, p2, q1, q2):
        return None
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (q2[0] - q1[0], q2[1] - q1[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) > config.GEOM_EPS:
        qp = (q1[0] - p1[0], q1[1] - p1[1])
        alpha = (qp[0] * s[1] - qp[1] * s[0]) / denom
        alpha = min(max(alpha, 0.0), 1.0)
    else:  # collinear overlap -> first overlapping point along p1->p2
        rr = r[0] * r[0] + r[1] * r[1]
        if rr <= config.GEOM_EPS:  # degenerate (stationary) path segment
            alpha = 0.0
        else:
            cands = [0.0] if on_segment(q1, p1, q2) else []
            for q in (q1, q2):
                t = ((q[0] - p1[0]) * r[0] + (q[1] - p1[1]) * r[1]) / rr
                if -config.GEOM_EPS <= t <= 1.0 + config.GEOM_EPS:
                    cands.append(min(max(t, 0.0), 1.0))
            alpha = min(cands)
    point = (p1[0] + alpha * (p2[0] - p1[0]), p1[1] + alpha * (p2[1] - p1[1]))
    return alpha, point


def fence_segments(points: Sequence[Point], closed: bool = False) -> List[Segment]:
    """Turn a polyline (or polygon if ``closed``) into a list of fence segments."""
    pts = [tuple(map(float, p)) for p in points]
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if closed and len(pts) > 2:
        segs.append((pts[-1], pts[0]))
    return segs


def point_in_polygon(pt: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test (boundary points count as outside)."""
    x, y = pt
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_int:
                inside = not inside
    return inside


def first_path_crossing(path_xy: np.ndarray, segments: Sequence[Segment]
                        ) -> Optional[Tuple[int, float, Point]]:
    """First contact of a polyline path with any fence segment.

    Args:
        path_xy: [N, 2] ordered path points.
        segments: fence segments.

    Returns:
        ``(k, alpha, point)`` for the first path segment k (p_k -> p_{k+1}) that
        touches the fence, or ``None``.
    """
    pts = np.asarray(path_xy, dtype=float)
    # fast bounding-box rejection per path segment
    fx = [min(a[0], b[0]) for a, b in segments] + [max(a[0], b[0]) for a, b in segments]
    fy = [min(a[1], b[1]) for a, b in segments] + [max(a[1], b[1]) for a, b in segments]
    fxmin, fxmax, fymin, fymax = min(fx), max(fx), min(fy), max(fy)
    for k in range(len(pts) - 1):
        p1, p2 = pts[k], pts[k + 1]
        if (max(p1[0], p2[0]) < fxmin or min(p1[0], p2[0]) > fxmax
                or max(p1[1], p2[1]) < fymin or min(p1[1], p2[1]) > fymax):
            continue
        best: Optional[Tuple[float, Point]] = None
        for q1, q2 in segments:
            hit = segment_intersection((p1[0], p1[1]), (p2[0], p2[1]), q1, q2)
            if hit is not None and (best is None or hit[0] < best[0]):
                best = hit
        if best is not None:
            return k, best[0], best[1]
    return None


# ===========================================================================
# 2. Early-warning trigger
# ===========================================================================
@dataclass
class Alert:
    """A predictive breach alert with intercept vector."""

    target_id: int
    time_to_breach: float      # seconds from the last observation
    crossing_point: Point      # (x, y) in metres
    heading_deg: float         # 0-360 deg from +x axis
    speed: float               # m/s along the crossing segment
    segment_index: int         # k (0 = last_obs -> p1)
    alpha: float               # fraction along segment k

    def format(self) -> str:
        """Exact alert line required by the Netra-One spec."""
        return (f"[PREDICTIVE ALERT] Target ID #{self.target_id} trajectory intersects "
                f"Restricted Fence in {self.time_to_breach:.1f} seconds! "
                f"Intercept Vector: [Angle {self.heading_deg:.0f}°, Velocity {self.speed:.1f} m/s]")

    def format_point(self) -> str:
        """Secondary line with the predicted crossing point."""
        return (f"    -> predicted crossing point (x, y) = ({self.crossing_point[0]:.1f} m, "
                f"{self.crossing_point[1]:.1f} m)")


def time_to_breach(k: int, alpha: float, dt: float = config.DT) -> float:
    """t_breach = (k + alpha) * DT for a crossing on path segment k at fraction alpha."""
    return (k + alpha) * dt


def intercept_vector(p_start: Point, p_end: Point, dt: float = config.DT) -> Tuple[float, float]:
    """Heading (deg, 0-360 from +x) and speed (m/s) of the crossing segment."""
    dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
    heading = math.degrees(math.atan2(dy, dx)) % 360.0
    return heading, math.hypot(dx, dy) / dt


def history_outside(obs_xy: np.ndarray, zone: Sequence[Point] = config.RESTRICTED_ZONE,
                    segments: Optional[Sequence[Segment]] = None) -> bool:
    """True if all observed points lie outside the zone and the history never crossed."""
    segments = segments if segments is not None else fence_segments(config.FENCE_LINE)
    if any(point_in_polygon((float(p[0]), float(p[1])), zone) for p in obs_xy):
        return False
    return first_path_crossing(obs_xy, segments) is None


def check_breach(target_id: int, obs_xy: np.ndarray, pred_xy: np.ndarray,
                 fence: Sequence[Point] = config.FENCE_LINE, closed: bool = False,
                 zone: Sequence[Point] = config.RESTRICTED_ZONE) -> Optional[Alert]:
    """Evaluate one target: does its predicted path cross the fence within T_PRED?

    Args:
        target_id: identifier printed in the alert.
        obs_xy: [T_OBS, 2] observed history.
        pred_xy: [T_PRED, 2] predicted future positions.
        fence: fence polyline / polygon vertices.
        closed: treat ``fence`` as a closed polygon.
        zone: restricted-zone polygon used to reject already-inside targets.

    Returns:
        An :class:`Alert` or ``None``.
    """
    segs = fence_segments(fence, closed)
    if not history_outside(obs_xy, zone, segs):
        return None
    path = np.vstack([obs_xy[-1:], pred_xy])
    hit = first_path_crossing(path, segs)
    if hit is None:
        return None
    k, alpha, point = hit
    heading, speed = intercept_vector(tuple(path[k]), tuple(path[k + 1]))
    return Alert(int(target_id), time_to_breach(k, alpha), point, heading, speed, k, alpha)


def emit_alert(alert: Alert, log_path: Path = config.ALERT_LOG_PATH, model: str = "") -> None:
    """Print an alert (exact format) + crossing point and append it to the alerts log."""
    print(alert.format())
    print(alert.format_point())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        fh.write(f"{stamp} [{model}] {alert.format()} | crossing=({alert.crossing_point[0]:.1f}, "
                 f"{alert.crossing_point[1]:.1f})\n")


# ===========================================================================
# 3. Alert evaluation (precision / recall / F1 / false alarms / lead time)
# ===========================================================================
def _prf(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Precision, recall, F1 and false-alarm rate from a confusion matrix."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": p, "recall": r, "f1": f1, "false_alarm_rate": far,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_alerts(predictor: "TrajectoryPredictor", tracks: "pd.DataFrame",
                    batch_size: int = 4096) -> Dict[str, float]:
    """Score alerts of ``predictor`` on every test track, streamed frame by frame.

    * Window level: for every window whose history is outside the zone, the
      label is "ground-truth future path crosses the fence within T_PRED" and the
      prediction is "predicted path crosses the fence".
    * Track level: the first alert per track. A true positive is an alert on a
      breach track raised before the real crossing; lead time = t_cross - t_alert.
      Any alert on a non-breach track is a false alarm.
    """
    segs = fence_segments(config.FENCE_LINE)
    obs_list, fut_list, meta = [], [], []
    track_cross: Dict[int, Optional[float]] = {}
    for tid, grp in tracks.groupby("track_id"):
        xy = grp.sort_values("frame")[["x", "y"]].to_numpy(float)
        hit = first_path_crossing(xy, segs)
        track_cross[int(tid)] = None if hit is None else (hit[0] + hit[1]) * config.DT
        last_end = len(xy) - config.T_PRED
        if hit is not None:
            last_end = min(last_end, hit[0] + 1)  # history must stay outside
        for end in range(config.T_OBS, last_end + 1):
            obs, fut = xy[end - config.T_OBS:end], xy[end:end + config.T_PRED]
            if not history_outside(obs, segments=segs):
                continue
            obs_list.append(obs)
            fut_list.append(fut)
            meta.append((int(tid), (end - 1) * config.DT))
    obs_arr, fut_arr = np.stack(obs_list), np.stack(fut_list)
    preds = np.concatenate([predictor.predict(obs_arr[i:i + batch_size])
                            for i in range(0, len(obs_arr), batch_size)])

    tp = fp = fn = tn = 0
    first_alert: Dict[int, float] = {}
    streak: Dict[int, int] = {}
    for i, (tid, t_now) in enumerate(meta):
        gt = first_path_crossing(np.vstack([obs_arr[i, -1:], fut_arr[i]]), segs) is not None
        pr = first_path_crossing(np.vstack([obs_arr[i, -1:], preds[i]]), segs) is not None
        tp += gt and pr
        fp += (not gt) and pr
        fn += gt and not pr
        tn += (not gt) and not pr
        streak[tid] = streak.get(tid, 0) + 1 if pr else 0
        if streak[tid] >= config.ALERT_CONSECUTIVE and tid not in first_alert:
            first_alert[tid] = t_now
    window = _prf(tp, fp, fn, tn)

    ttp = tfp = tfn = ttn = 0
    leads: List[float] = []
    for tid, t_cross in track_cross.items():
        alerted = tid in first_alert
        if t_cross is not None:
            if alerted and first_alert[tid] <= t_cross:
                ttp += 1
                leads.append(t_cross - first_alert[tid])
            else:
                tfn += 1
        else:
            tfp += alerted
            ttn += not alerted
    track = _prf(ttp, tfp, tfn, ttn)
    res = {f"window_{k}": v for k, v in window.items()}
    res.update({f"track_{k}": v for k, v in track.items()})
    res["mean_lead_time_s"] = float(np.mean(leads)) if leads else 0.0
    res["median_lead_time_s"] = float(np.median(leads)) if leads else 0.0
    res["num_windows"] = len(meta)
    res["num_tracks"] = len(track_cross)
    return res


# ===========================================================================
# 4. Visualisation
# ===========================================================================
def _draw_scene(ax, obs: np.ndarray, fut: Optional[np.ndarray], pred: np.ndarray,
                alert: Optional[Alert], title: str, pred_label: str = "Predicted future",
                annotate: bool = True, zoom: bool = True) -> None:
    """Draw one forecast scene on ``ax`` with the mandated colour / line styles."""
    zone = np.asarray(config.RESTRICTED_ZONE)
    ax.fill(zone[:, 0], zone[:, 1], color=config.COLORS["zone"], alpha=0.5,
            label="Restricted zone (y > 450)", zorder=0)
    fence = np.asarray(config.FENCE_LINE)
    ax.plot(fence[:, 0], fence[:, 1], "-", color=config.COLORS["fence"], lw=3,
            label="Virtual defense perimeter", zorder=2)
    ax.plot(obs[:, 0], obs[:, 1], "-o", color=config.COLORS["history"], ms=3.5, lw=2,
            label="Observed history", zorder=4)
    if fut is not None:
        ax.plot(np.r_[obs[-1:, 0], fut[:, 0]], np.r_[obs[-1:, 1], fut[:, 1]], "-o",
                color=config.COLORS["future"], ms=3.5, lw=2, label="Actual future", zorder=3)
    ax.plot(np.r_[obs[-1:, 0], pred[:, 0]], np.r_[obs[-1:, 1], pred[:, 1]], "--o",
            color=config.COLORS["pred"], ms=3.5, lw=2, mfc="white", label=pred_label, zorder=5)
    if alert is not None:
        ax.plot(*alert.crossing_point, marker="*", color="gold", mec="black", ms=18,
                zorder=6, label="Predicted breach point")
        if annotate:
            txt = (f"ALERT: breach in {alert.time_to_breach:.1f} s\n"
                   f"Intercept: {alert.heading_deg:.0f}°, {alert.speed:.1f} m/s")
            ax.annotate(txt, xy=alert.crossing_point, xytext=(0.03, 0.97),
                        textcoords="axes fraction", va="top", fontsize=8.5,
                        bbox=dict(boxstyle="round", fc="lightyellow", ec="red"),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.2), zorder=7)
    if zoom:
        pts = [obs, pred] + ([fut] if fut is not None else [])
        allp = np.vstack(pts)
        if allp[:, 1].max() > config.FENCE_Y - config.PLOT_FENCE_MARGIN:
            # keep the perimeter in view for targets operating near it
            allp = np.vstack([allp, [[allp[0, 0], config.FENCE_Y + 5.0]]])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        cx, cy = (lo + hi) / 2
        span = max(hi[0] - lo[0], hi[1] - lo[1], 30.0) * 0.6 + 6
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
    else:
        ax.set_xlim(*config.WORLD_X)
        ax.set_ylim(*config.WORLD_Y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.3)


def _ade_fde(pred: np.ndarray, fut: np.ndarray) -> Tuple[float, float]:
    """ADE / FDE of a single trajectory."""
    d = np.linalg.norm(pred - fut, axis=-1)
    return float(d.mean()), float(d[-1])


def plot_single(obs: np.ndarray, fut: np.ndarray, pred: np.ndarray, alert: Optional[Alert],
                target_id: int, model_name: str, path: Path) -> None:
    """Publication-quality single forecast figure (trajectory_forecast.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ade, fde = _ade_fde(pred, fut)
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_scene(ax, obs, fut, pred, alert,
                f"Target #{target_id} — {model_name.upper()} forecast "
                f"({config.T_OBS} obs → {config.T_PRED} pred, {config.T_PRED * config.DT:.1f} s)\n"
                f"ADE = {ade:.2f} m   FDE = {fde:.2f} m",
                pred_label=f"Predicted future ({model_name.upper()})")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)
    logger.info("saved %s", path)


def plot_grid(samples: List[dict], model_name: str, path: Path) -> None:
    """Grid of forecasts for one model (forecast_grid_<model>.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    cols = 3 if n > 4 else n
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False)
    for ax, s in zip(axes.ravel(), samples):
        ade, fde = _ade_fde(s["pred"], s["fut"])
        tag = "ALERT" if s["alert"] else "no alert"
        _draw_scene(ax, s["obs"], s["fut"], s["pred"], s["alert"],
                    f"#{s['tid']} {s['entity']}/{s['behavior']} — {tag}\nADE {ade:.2f} m · FDE {fde:.2f} m",
                    annotate=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f"{model_name.upper()} forecasts on test tracks", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)
    logger.info("saved %s", path)


def plot_comparison_grid(cases: List[dict], names: Tuple[str, str], path: Path) -> None:
    """Side-by-side model comparison on the same test windows (comparison_grid.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cases)
    rows = int(math.ceil(n / 2))
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4.4 * rows), squeeze=False)
    for i, c in enumerate(cases):
        r, c0 = divmod(i, 2)
        for j, name in enumerate(names):
            ax = axes[r, 2 * c0 + j]
            pred, alert = c["preds"][name], c["alerts"][name]
            ade, fde = _ade_fde(pred, c["fut"])
            tag = "ALERT" if alert else "no alert"
            _draw_scene(ax, c["obs"], c["fut"], pred, alert,
                        f"{name.upper()} | #{c['tid']} {c['behavior']} ({c['entity']})\n"
                        f"ADE {ade:.2f} m · FDE {fde:.2f} m · {tag}",
                        pred_label="Predicted future", annotate=False)
    for ax in axes.ravel()[2 * n:]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10, frameon=False)
    fig.suptitle(f"{names[0].upper()} vs {names[1].upper()} on identical test windows "
                 "(breach / near-miss / patrol)", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)
    logger.info("saved %s", path)


# ===========================================================================
# 5. Window selection helpers + CLI
# ===========================================================================
def _track_arrays(df: "pd.DataFrame", tid: int) -> np.ndarray:
    """[N, 2] positions of one track."""
    return df[df.track_id == tid].sort_values("frame")[["x", "y"]].to_numpy(float)


def pick_window(xy: np.ndarray, rng: np.random.Generator, pre_cross_frames: int = 6
                ) -> Optional[int]:
    """Choose the end index of an observation window for plotting.

    For breach tracks the window ends ``pre_cross_frames`` before the real
    crossing (an early-warning situation); otherwise a random valid window near
    the fence approach is chosen.
    """
    n = len(xy)
    if n < config.T_OBS + config.T_PRED:
        return None
    hit = first_path_crossing(xy, fence_segments(config.FENCE_LINE))
    if hit is not None:
        end = hit[0] + 1 - pre_cross_frames
        end = max(config.T_OBS, min(end, n - config.T_PRED))
        return end if end <= hit[0] + 1 else None
    # near-miss / patrol: window whose future comes closest to the fence
    ends = np.arange(config.T_OBS, n - config.T_PRED + 1)
    closest = [xy[e:e + config.T_PRED, 1].max() for e in ends]
    return int(ends[int(np.argmax(closest))]) if rng.random() < 0.7 else int(rng.choice(ends))


def _select_cases(df: "pd.DataFrame", test_ids: Sequence[int], rng: np.random.Generator,
                  per_behavior: Dict[str, int]) -> List[dict]:
    """Pick test windows for plotting, stratified by behaviour."""
    meta = df.groupby("track_id")[["entity_type", "behavior"]].first()
    cases: List[dict] = []
    for beh, cnt in per_behavior.items():
        behs = config.BREACH_BEHAVIORS if beh == "breach" else (beh,)
        ids = [t for t in test_ids if meta.loc[t, "behavior"] in behs]
        rng.shuffle(ids)
        for tid in ids:
            if sum(c["group"] == beh for c in cases) >= cnt:
                break
            xy = _track_arrays(df, tid)
            end = pick_window(xy, rng)
            if end is None or not history_outside(xy[end - config.T_OBS:end]):
                continue
            cases.append(dict(tid=int(tid), group=beh, obs=xy[end - config.T_OBS:end],
                              fut=xy[end:end + config.T_PRED],
                              entity=meta.loc[tid, "entity_type"], behavior=meta.loc[tid, "behavior"]))
    return cases


def run_samples(model: str, num_samples: int, df: "pd.DataFrame", test_ids: Sequence[int]) -> None:
    """Forecast ``num_samples`` test windows, print alerts and save all plots."""
    from models.trajectory_model import build_predictor

    rng = np.random.default_rng(config.SEED)
    predictor = build_predictor(model)
    n_breach = max(1, int(round(num_samples * 4 / 9)))
    n_rest = num_samples - n_breach
    per = {"breach": n_breach, "near_miss": (n_rest + 1) // 2, "patrol": n_rest // 2}
    cases = _select_cases(df, test_ids, rng, per)
    preds = predictor.predict(np.stack([c["obs"] for c in cases]))
    for c, p in zip(cases, preds):
        c["pred"] = p
        c["alert"] = check_breach(c["tid"], c["obs"], p)
        if c["alert"]:
            emit_alert(c["alert"], model=model)
        else:
            logger.info("Target ID #%d (%s): no predicted breach within %.1f s",
                        c["tid"], c["behavior"], config.T_PRED * config.DT)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_grid(cases, model, config.OUTPUT_DIR / f"forecast_grid_{model}.png")

    # hero figure: the best-predicted alerted breach case
    alerted = [c for c in cases if c["alert"] is not None and c["group"] == "breach"] or cases
    hero = min(alerted, key=lambda c: _ade_fde(c["pred"], c["fut"])[0])
    plot_single(hero["obs"], hero["fut"], hero["pred"], hero["alert"], hero["tid"], model,
                config.OUTPUT_DIR / "trajectory_forecast.png")

    # side-by-side comparison against the other model family
    other = "kalman" if model not in ("kalman", "poly1", "poly2") else "lstm"
    try:
        other_pred = build_predictor(other)
    except FileNotFoundError as exc:
        logger.warning("skipping comparison grid: %s", exc)
        return
    comp = _select_cases(df, test_ids, np.random.default_rng(config.SEED + 1),
                         {"breach": 2, "near_miss": 2, "patrol": 2})
    obs = np.stack([c["obs"] for c in comp])
    names = (model, other)
    outs = {model: predictor.predict(obs), other: other_pred.predict(obs)}
    for i, c in enumerate(comp):
        c["preds"] = {nm: outs[nm][i] for nm in names}
        c["alerts"] = {nm: check_breach(c["tid"], c["obs"], outs[nm][i]) for nm in names}
    plot_comparison_grid(comp, names, config.OUTPUT_DIR / "comparison_grid.png")


def run_track(model: str, track_id: int, df: "pd.DataFrame") -> None:
    """Evaluate one track (any split) at the window just before its fence approach."""
    from models.trajectory_model import build_predictor

    if track_id not in set(df.track_id.unique()):
        raise SystemExit(f"track {track_id} not found")
    predictor = build_predictor(model)
    xy = _track_arrays(df, track_id)
    end = pick_window(xy, np.random.default_rng(config.SEED))
    if end is None:
        raise SystemExit(f"track {track_id} too short")
    obs, fut = xy[end - config.T_OBS:end], xy[end:end + config.T_PRED]
    pred = predictor.predict(obs[None])[0]
    alert = check_breach(track_id, obs, pred)
    ade, fde = _ade_fde(pred, fut)
    logger.info("track #%d window frames %d-%d  ADE=%.2f m FDE=%.2f m", track_id,
                end - config.T_OBS, end - 1, ade, fde)
    if alert:
        emit_alert(alert, model=model)
    else:
        print(f"Target ID #{track_id}: no predicted breach within {config.T_PRED * config.DT:.1f} s")
    path = config.OUTPUT_DIR / f"track_{track_id}_{model}.png"
    plot_single(obs, fut, pred, alert, track_id, model, path)


def run_demo(model: str, df: "pd.DataFrame", test_ids: Sequence[int], delay: float,
             track_id: Optional[int] = None) -> None:
    """Stream a test breach track frame by frame, printing live status and alerts."""
    from models.trajectory_model import build_predictor

    predictor = build_predictor(model)
    meta = df.groupby("track_id")["behavior"].first()
    if track_id is None:
        cands = [t for t in test_ids if meta.loc[t] in config.BREACH_BEHAVIORS]
        track_id = int(np.random.default_rng(config.SEED).choice(cands))
    xy = _track_arrays(df, track_id)
    segs = fence_segments(config.FENCE_LINE)
    hit = first_path_crossing(xy, segs)
    t_cross = None if hit is None else (hit[0] + hit[1]) * config.DT
    print(f"=== LIVE DEMO: target #{track_id} ({meta.loc[track_id]}), model={model}, "
          f"actual crossing at t={t_cross if t_cross is None else round(t_cross, 1)} s ===")
    first_alert = None
    for end in range(config.T_OBS, len(xy) + 1):
        obs = xy[end - config.T_OBS:end]
        t_now = (end - 1) * config.DT
        if not history_outside(obs, segments=segs):
            print(f"t={t_now:5.1f}s  pos=({obs[-1, 0]:6.1f},{obs[-1, 1]:6.1f})  "
                  "TARGET INSIDE RESTRICTED ZONE — tracking only")
            if t_cross is not None and first_alert is not None:
                break
            continue
        pred = predictor.predict(obs[None])[0]
        alert = check_breach(track_id, obs, pred)
        if alert:
            print(f"t={t_now:5.1f}s  ", end="")
            emit_alert(alert, model=f"{model}-demo")
            first_alert = first_alert if first_alert is not None else t_now
        else:
            print(f"t={t_now:5.1f}s  pos=({obs[-1, 0]:6.1f},{obs[-1, 1]:6.1f})  clear")
        time.sleep(delay)
    if first_alert is not None and t_cross is not None:
        print(f"=== first alert at t={first_alert:.1f}s, actual crossing at t={t_cross:.1f}s "
              f"-> lead time {t_cross - first_alert:.1f}s ===")


def write_alert_metrics(results: Dict[str, Dict[str, float]], out_dir: Path = config.OUTPUT_DIR) -> str:
    """Write alert metrics for several models to markdown + csv; returns the markdown."""
    import pandas as pd

    df = pd.DataFrame(results).T
    df.index.name = "model"
    df.to_csv(out_dir / "alert_metrics.csv", float_format="%.4f")
    lines = [
        "| Model | Level | Precision | Recall | F1 | False-alarm rate | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, r in results.items():
        for lvl in ("window", "track"):
            lines.append(
                f"| {m.upper()} | {lvl} | {r[f'{lvl}_precision']:.3f} | {r[f'{lvl}_recall']:.3f} | "
                f"{r[f'{lvl}_f1']:.3f} | {r[f'{lvl}_false_alarm_rate']:.3f} | {int(r[f'{lvl}_tp'])} | "
                f"{int(r[f'{lvl}_fp'])} | {int(r[f'{lvl}_fn'])} | {int(r[f'{lvl}_tn'])} |")
    lines.append("")
    lines.append("| Model | Mean lead time [s] | Median lead time [s] | Windows | Tracks |")
    lines.append("|---|---|---|---|---|")
    for m, r in results.items():
        lines.append(f"| {m.upper()} | {r['mean_lead_time_s']:.2f} | {r['median_lead_time_s']:.2f} | "
                     f"{int(r['num_windows'])} | {int(r['num_tracks'])} |")
    md = "\n".join(lines) + "\n"
    (out_dir / "alert_metrics.md").write_text(md, encoding="utf-8")
    return md


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru", "kalman", "poly1", "poly2"])
    ap.add_argument("--num-samples", type=int, default=9)
    ap.add_argument("--track-id", type=int, default=None)
    ap.add_argument("--demo", action="store_true", help="stream a track frame by frame")
    ap.add_argument("--delay", type=float, default=config.DEMO_DELAY, help="seconds per demo frame")
    ap.add_argument("--evaluate-alerts", action="store_true",
                    help="alert precision/recall/lead-time on the test split")
    ap.add_argument("--data", type=Path, default=config.TRACKS_CSV)
    return ap.parse_args()


def main() -> None:
    """CLI entry point."""
    config.setup_logging()
    config.set_seed()
    args = parse_args()
    from dataset_loader import load_tracks, split_track_ids

    df = load_tracks(args.data)
    _, _, test_ids = split_track_ids(df)
    if args.demo:
        run_demo(args.model, df, test_ids, args.delay, args.track_id)
    elif args.track_id is not None:
        run_track(args.model, args.track_id, df)
    elif args.evaluate_alerts:
        from models.trajectory_model import build_predictor

        res = evaluate_alerts(build_predictor(args.model), df[df.track_id.isin(test_ids)])
        print(write_alert_metrics({args.model: res}))
    else:
        run_samples(args.model, args.num_samples, df, test_ids)


if __name__ == "__main__":
    main()
