"""Train the LSTM/GRU seq2seq trajectory forecaster.

Adam + ReduceLROnPlateau, gradient clipping, teacher forcing decayed linearly
from TF_START to TF_END, early stopping on validation ADE; the best checkpoint
and loss curves are saved.

Usage::

    python train.py --cell lstm --epochs 50
    python train.py --cell gru --loss mse
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import config
from dataset_loader import NormStats, build_dataloaders, make_loader
from models.kalman_baseline import ConstantVelocityKF
from models.seq2seq_predictor import Seq2SeqPredictor

logger = logging.getLogger("train")


def prepare_batch(batch: Tuple[torch.Tensor, ...], stats: NormStats, device: torch.device
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move a batch to ``device`` and build scaled decoder seed / targets.

    Returns ``(feats, last_disp, target_rel, target_disp)`` where targets are in
    displacement-scale units relative to the last observed point.
    """
    feats, obs_xy, fut_xy, _ = (t.to(device, non_blocking=True) for t in batch)
    scale = stats.disp_scale
    last_disp = (obs_xy[:, -1] - obs_xy[:, -2]) / scale
    target_rel = (fut_xy - obs_xy[:, -1:, :]) / scale
    prev = torch.cat([torch.zeros_like(target_rel[:, :1]), target_rel[:, :-1]], dim=1)
    target_disp = target_rel - prev
    return feats, last_disp, target_rel, target_disp


def make_loss(name: str) -> nn.Module:
    """Loss factory: ``smoothl1`` (default) or ``mse``."""
    return nn.SmoothL1Loss() if name == "smoothl1" else nn.MSELoss()


@torch.no_grad()
def evaluate_loader(model: Seq2SeqPredictor, loader: DataLoader, stats: NormStats,
                    loss_fn: nn.Module, device: torch.device) -> Dict[str, float]:
    """Loss (scaled units) plus ADE/FDE in metres over a DataLoader."""
    model.eval()
    tot_loss, tot_ade, tot_fde, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        feats, last_disp, target_rel, _ = prepare_batch(batch, stats, device)
        pred = model(feats, last_disp)
        b = feats.shape[0]
        tot_loss += loss_fn(pred, target_rel).item() * b
        dist = torch.linalg.norm((pred - target_rel) * stats.disp_scale, dim=-1)
        tot_ade += dist.mean(dim=1).sum().item()
        tot_fde += dist[:, -1].sum().item()
        n += b
    return {"loss": tot_loss / n, "ade": tot_ade / n, "fde": tot_fde / n}


def tf_schedule(epoch: int, epochs: int) -> float:
    """Linear teacher-forcing decay TF_START -> TF_END over the training run."""
    if epochs <= 1:
        return config.TF_END
    frac = min(epoch / (epochs - 1), 1.0)
    return config.TF_START + (config.TF_END - config.TF_START) * frac


def plot_loss_curves(history: List[Dict[str, float]], best_epoch: int, kf_ref: Dict[str, float],
                     cell: str) -> None:
    """Save sample_outputs/loss_curves.png (train vs val loss, val ADE/FDE)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.plot(ep, [h["train_eval_loss"] for h in history], "-o", ms=3, color="tab:blue",
            label="train loss (free-running)")
    a1.plot(ep, [h["val_loss"] for h in history], "-s", ms=3, color="tab:orange",
            label="validation loss (free-running)")
    a1.plot(ep, [h["train_loss"] for h in history], "-", lw=1, alpha=0.45, color="tab:blue",
            label="train batch loss (teacher forcing 0.5→0)")
    a1.axvline(best_epoch, color="gray", ls=":", label=f"best epoch ({best_epoch})")
    a1.set_yscale("log")
    a1.set_xlabel("epoch")
    a1.set_ylabel(f"{history[0]['loss_name']} loss (scaled units, log)")
    a1.set_title(f"{cell.upper()} seq2seq — training convergence")
    a1.grid(alpha=0.3, which="both")
    a1.legend()
    a2.plot(ep, [h["val_ade"] for h in history], "-o", ms=3, color="tab:red", label="val ADE (LSTM)"
            if cell == "lstm" else "val ADE")
    a2.plot(ep, [h["val_fde"] for h in history], "-s", ms=3, color="tab:purple", label="val FDE")
    a2.axhline(kf_ref["ade"], color="tab:orange", ls="--", label=f"Kalman val ADE ({kf_ref['ade']:.2f} m)")
    a2.axhline(kf_ref["fde"], color="tab:orange", ls=":", label=f"Kalman val FDE ({kf_ref['fde']:.2f} m)")
    a2.axvline(best_epoch, color="gray", ls=":")
    a2.set_xlabel("epoch")
    a2.set_ylabel("error [m]")
    a2.set_title(f"Validation ADE / FDE ({config.T_PRED * config.DT:.1f} s horizon)")
    a2.grid(alpha=0.3)
    a2.legend()
    fig.tight_layout()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.OUTPUT_DIR / "loss_curves.png", dpi=config.FIG_DPI)
    plt.close(fig)


def kalman_reference(val_windows) -> Dict[str, float]:
    """Kalman ADE/FDE on the validation windows (horizontal reference line)."""
    pred = ConstantVelocityKF().predict(val_windows.obs_xy)
    d = np.linalg.norm(pred - val_windows.fut_xy, axis=-1)
    return {"ade": float(d.mean()), "fde": float(d[:, -1].mean())}


def train(args: argparse.Namespace) -> Dict[str, float]:
    """Full training run; returns best validation metrics."""
    config.set_seed(args.seed)
    device = config.get_device()
    logger.info("device: %s", device)
    loaders, stats, windows = build_dataloaders(batch_size=args.batch_size, device=device,
                                               samples_per_epoch=args.samples_per_epoch)
    # fixed train subset evaluated without teacher forcing (comparable to val loss)
    sub = np.random.default_rng(args.seed).choice(len(windows["train"]),
                                                  size=min(config.TRAIN_EVAL_SAMPLES, len(windows["train"])),
                                                  replace=False)
    train_eval_loader = make_loader(Subset(loaders["train"].dataset, sub.tolist()), 4096, shuffle=False)
    kf_ref = kalman_reference(windows["val"])
    logger.info("Kalman reference on val: ADE=%.3f m FDE=%.3f m", kf_ref["ade"], kf_ref["fde"])

    model_cfg = dict(input_size=config.NUM_FEATURES, hidden_size=config.HIDDEN_SIZE,
                     num_layers=config.NUM_LAYERS, dropout=config.DROPOUT,
                     t_pred=config.T_PRED, cell=args.cell)
    model = Seq2SeqPredictor(**model_cfg).to(device)
    logger.info("model params: %d", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=config.LR_FACTOR,
                                                       patience=config.LR_PATIENCE)
    loss_fn = make_loss(args.loss)
    ckpt_path = config.checkpoint_path(args.cell)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best = {"ade": float("inf"), "fde": float("inf"), "epoch": 0}
    bad_epochs = 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tf = tf_schedule(epoch - 1, args.epochs)
        t0, run_loss, n = time.time(), 0.0, 0
        for batch in loaders["train"]:
            feats, last_disp, target_rel, target_disp = prepare_batch(batch, stats, device)
            pred = model(feats, last_disp, target_disp, tf_ratio=tf)
            loss = loss_fn(pred, target_rel)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
            run_loss += loss.item() * feats.shape[0]
            n += feats.shape[0]
        val = evaluate_loader(model, loaders["val"], stats, loss_fn, device)
        tr_eval = evaluate_loader(model, train_eval_loader, stats, loss_fn, device)
        sched.step(val["ade"])
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": run_loss / n, "train_eval_loss": tr_eval["loss"],
                        "train_ade": tr_eval["ade"], "val_loss": val["loss"],
                        "val_ade": val["ade"], "val_fde": val["fde"], "lr": lr, "tf_ratio": tf,
                        "loss_name": args.loss, "epoch_time_s": time.time() - t0})
        improved = val["ade"] < best["ade"] - 1e-4
        if improved:
            best = {"ade": val["ade"], "fde": val["fde"], "epoch": epoch}
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(), "model_config": model_cfg,
                        "epoch": epoch, "val_ade": val["ade"], "val_fde": val["fde"],
                        "loss": args.loss}, ckpt_path)
        else:
            bad_epochs += 1
        logger.info("epoch %3d | train(tf) %.4f | train %.4f | val %.4f | train ADE %.3f m | "
                    "val ADE %.3f m FDE %.3f m | lr %.1e | tf %.2f | %.1fs%s", epoch, run_loss / n,
                    tr_eval["loss"], val["loss"], tr_eval["ade"], val["ade"], val["fde"],
                    lr, tf, time.time() - t0, "  *best*" if improved else "")
        if bad_epochs >= config.PATIENCE:
            logger.info("early stopping: no val-ADE improvement for %d epochs", config.PATIENCE)
            break

    logger.info("training done in %.1f min; best epoch %d val ADE %.3f m FDE %.3f m -> %s",
                (time.time() - t_start) / 60, best["epoch"], best["ade"], best["fde"], ckpt_path)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / f"training_log_{args.cell}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    plot_loss_curves(history, best["epoch"], kf_ref, args.cell)
    return best


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--loss", choices=["smoothl1", "mse"], default=config.DEFAULT_LOSS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--samples-per-epoch", type=int, default=config.SAMPLES_PER_EPOCH,
                    help="random train windows drawn per epoch (0 = all)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    return ap.parse_args()


def main() -> None:
    """CLI entry point."""
    config.setup_logging()
    train(parse_args())


if __name__ == "__main__":
    main()
