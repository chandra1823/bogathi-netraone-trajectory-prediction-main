"""ADE/FDE benchmark of all forecasters on the TEST split + alert evaluation.

Outputs (sample_outputs/):
    results_table.md, results.csv          overall / per-entity / per-behaviour ADE-FDE + latency
    ade_fde_bar.png, error_vs_step.png     comparison charts
    alert_metrics.md, alert_metrics.csv    alert precision / recall / F1 / FAR / lead time

Usage::

    python evaluate.py
    python evaluate.py --models kalman poly1 poly2 lstm gru
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
from dataset_loader import load_tracks, make_windows, split_track_ids
from models.trajectory_model import TrajectoryPredictor, build_predictor
from predict_breach import evaluate_alerts, write_alert_metrics

logger = logging.getLogger("evaluate")

DISPLAY = {"kalman": "CV Kalman Filter", "poly1": "Polynomial (deg 1)",
           "poly2": "Polynomial (deg 2)", "lstm": "LSTM seq2seq", "gru": "GRU seq2seq"}


def ade_fde(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Average and final displacement error (mean over the batch).

    ADE = mean over samples and steps of ||pred - gt||_2;
    FDE = mean over samples of ||pred_T - gt_T||_2.
    """
    d = np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1)
    return float(d.mean()), float(d[..., -1].mean())


def timed_predict(model: TrajectoryPredictor, obs: np.ndarray, batch: int = 4096
                  ) -> Tuple[np.ndarray, float, float]:
    """Predict all windows; returns (pred, batched ms/traj, single-trajectory ms)."""
    model.predict(obs[:8])  # warm-up
    t0 = time.perf_counter()
    pred = np.concatenate([model.predict(obs[i:i + batch]) for i in range(0, len(obs), batch)])
    batched = (time.perf_counter() - t0) * 1e3 / len(obs)
    n_single = min(200, len(obs))
    t0 = time.perf_counter()
    for i in range(n_single):
        model.predict(obs[i:i + 1])
    single = (time.perf_counter() - t0) * 1e3 / n_single
    return pred, batched, single


def breakdown(dist: np.ndarray, labels: np.ndarray) -> Dict[str, Tuple[float, float, int]]:
    """Per-group (ADE, FDE, count) from a [N, T_PRED] distance matrix."""
    out = {}
    for g in sorted(set(labels)):
        m = labels == g
        out[str(g)] = (float(dist[m].mean()), float(dist[m, -1].mean()), int(m.sum()))
    return out


def to_markdown(rows: List[Dict], group_col: str, models: List[str]) -> str:
    """Pivot per-group rows into a markdown table ``group | ADE/FDE per model``."""
    groups = sorted({r["group"] for r in rows if r["breakdown"] == group_col})
    head = "| " + group_col + " | N | " + " | ".join(f"{DISPLAY[m]} ADE / FDE" for m in models) + " |"
    sep = "|---|---|" + "---|" * len(models)
    lines = [head, sep]
    for g in groups:
        cells, n = [], 0
        best = min(r["ade"] for r in rows if r["breakdown"] == group_col and r["group"] == g)
        for m in models:
            r = next(r for r in rows if r["breakdown"] == group_col and r["group"] == g and r["model"] == m)
            n = r["n"]
            txt = f"{r['ade']:.2f} / {r['fde']:.2f}"
            cells.append(f"**{txt}**" if abs(r["ade"] - best) < 1e-9 else txt)
        lines.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_bars(overall: pd.DataFrame, path) -> None:
    """Grouped ADE / FDE bar chart per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(overall))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - 0.2, overall["ADE_m"], 0.4, label="ADE", color="tab:blue")
    b2 = ax.bar(x + 0.2, overall["FDE_m"], 0.4, label="FDE", color="tab:red")
    ax.bar_label(b1, fmt="%.2f", fontsize=9)
    ax.bar_label(b2, fmt="%.2f", fontsize=9)
    ax.set_xticks(x, [DISPLAY[m] for m in overall["model"]])
    ax.set_ylabel("error [m]")
    ax.set_title(f"Test-split ADE / FDE ({config.T_OBS} obs → {config.T_PRED} pred = "
                 f"{config.T_PRED * config.DT:.1f} s horizon, lower is better)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_error_vs_step(step_err: Dict[str, np.ndarray], path) -> None:
    """Mean displacement error at each prediction step (error growth over the horizon)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(1, config.T_PRED + 1) * config.DT
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m, e in step_err.items():
        ax.plot(t, e, "-o", ms=4, lw=2, color=config.COLORS.get(m), label=DISPLAY[m])
    ax.set_xlabel("prediction horizon [s]")
    ax.set_ylabel("mean displacement error [m]")
    ax.set_title("Error growth over the prediction horizon (test split)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    config.setup_logging()
    config.set_seed()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["kalman", "poly1", "poly2", "lstm"])
    ap.add_argument("--alert-models", nargs="+", default=["kalman", "lstm"])
    args = ap.parse_args()

    df = load_tracks()
    _, _, test_ids = split_track_ids(df)
    test = make_windows(df, test_ids, config.EVAL_STRIDE)
    logger.info("test split: %d tracks, %d windows", len(test_ids), len(test))

    rows: List[Dict] = []
    overall_rows: List[Dict] = []
    step_err: Dict[str, np.ndarray] = {}
    for name in args.models:
        model = build_predictor(name)
        pred, ms_batch, ms_single = timed_predict(model, test.obs_xy)
        dist = np.linalg.norm(pred - test.fut_xy, axis=-1)
        ade, fde = ade_fde(pred, test.fut_xy)
        step_err[name] = dist.mean(axis=0)
        overall_rows.append({"model": name, "ADE_m": ade, "FDE_m": fde,
                             "ms_per_traj_batched": ms_batch, "ms_per_traj_single": ms_single,
                             "n_windows": len(test)})
        rows.append({"model": name, "breakdown": "overall", "group": "all", "ade": ade, "fde": fde,
                     "n": len(test)})
        for col, labels in (("entity", test.entity), ("behavior", test.behavior)):
            for g, (a, f, n) in breakdown(dist, labels).items():
                rows.append({"model": name, "breakdown": col, "group": g, "ade": a, "fde": f, "n": n})
        logger.info("%-7s ADE %.3f m  FDE %.3f m  | %.4f ms/traj batched, %.3f ms single",
                    name, ade, fde, ms_batch, ms_single)

    overall = pd.DataFrame(overall_rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / "results.csv", index=False, float_format="%.4f")

    best_ade = overall["ADE_m"].min()
    md = [f"### Overall (test split: {len(test_ids)} tracks, {len(test)} windows, "
          f"{config.T_OBS} obs → {config.T_PRED} pred = {config.T_PRED * config.DT:.1f} s)", "",
          "| Model | ADE [m] | FDE [m] | Inference [ms/traj, batched] | Inference [ms/traj, single] |",
          "|---|---|---|---|---|"]
    for r in overall_rows:
        b = "**" if abs(r["ADE_m"] - best_ade) < 1e-9 else ""
        md.append(f"| {b}{DISPLAY[r['model']]}{b} | {b}{r['ADE_m']:.3f}{b} | {b}{r['FDE_m']:.3f}{b} | "
                  f"{r['ms_per_traj_batched']:.4f} | {r['ms_per_traj_single']:.3f} |")
    md += ["", "### By entity type (ADE / FDE in metres, best in bold)", "",
           to_markdown(rows, "entity", args.models),
           "", "### By behaviour (ADE / FDE in metres, best in bold)", "",
           to_markdown(rows, "behavior", args.models), ""]
    table = "\n".join(md)
    (config.OUTPUT_DIR / "results_table.md").write_text(table, encoding="utf-8")
    plot_bars(overall, config.OUTPUT_DIR / "ade_fde_bar.png")
    plot_error_vs_step(step_err, config.OUTPUT_DIR / "error_vs_step.png")
    print(table)

    logger.info("evaluating predictive alerts on %d test tracks ...", len(test_ids))
    test_df = df[df.track_id.isin(test_ids)]
    alert_res = {m: evaluate_alerts(build_predictor(m), test_df) for m in args.alert_models}
    print(write_alert_metrics(alert_res))


if __name__ == "__main__":
    main()
