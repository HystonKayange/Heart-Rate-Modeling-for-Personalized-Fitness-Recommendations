#!/usr/bin/env python3
"""
Build public figures for the clean reevaluation package.

Outputs (default: examples/figures/public/):
  01_mae_comparison.png       — published vs standard-protocol MAE
  02_workout_predictions.png  — true vs predicted HR (held-out samples)
  03_error_scatter.png        — predicted mean HR vs true mean HR (sample)

Example:
  python plot_public_figures.py \\
    --name paper-faithful-run-val \\
    --paper-faithful \\
    --sport run --history-source all-prior \\
    --seq-length 128 --feature-set basic
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

EXAMPLES = Path(__file__).resolve().parent
PROJECT = EXAMPLES.parent
sys.path.insert(0, str(PROJECT))

from Model.activity_features import add_activity_features
from Model.data import WorkoutDataset, WorkoutDatasetConfig, workout_dataset_collate_fn
from Model.dbn import DBNConfig, DBNModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="paper-faithful-run-val")
    p.add_argument("--data", default=str(PROJECT / "output" / "endomondo_filtered.feather"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--outdir", default=str(EXAMPLES / "figures" / "public"))
    p.add_argument("--sport", default="run")
    p.add_argument("--history-source", choices=["split", "all-prior"], default="all-prior")
    p.add_argument("--seq-length", type=int, default=128)
    p.add_argument("--feature-set", choices=["basic", "run_intensity", "run_personal"], default="basic")
    p.add_argument("--paper-faithful", action="store_true", default=True)
    p.add_argument("--no-paper-faithful", action="store_false", dest="paper_faithful")
    p.add_argument("--physiological", action="store_true")
    p.add_argument("--residual", action="store_true")
    p.add_argument("--adafs", action="store_true")
    p.add_argument("--adafs-variant", choices=["legacy", "paper"], default="paper")
    p.add_argument("--eval-stride", type=int, default=32)
    p.add_argument("--n-workouts", type=int, default=4, help="number of sample workouts to plot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-scatter", type=int, default=800, help="max held-out workouts in scatter")
    return p.parse_args()


def window_starts(length, seq_length, stride):
    if length <= seq_length:
        return [0]
    starts = list(range(0, length - seq_length, stride))
    if not starts or starts[-1] + seq_length < length:
        starts.append(length - seq_length)
    return starts


def forecast_stitched(model, activity, times, workout_id, subject_id, history, history_length, device, seq_length, stride):
    length = activity.shape[0]
    starts = window_starts(length, seq_length, stride)
    n = len(starts)
    win_act = torch.stack([activity[s : s + seq_length] for s in starts]).float().to(device)
    win_t = torch.stack([times[s : s + seq_length] for s in starts]).float().to(device)
    wids = torch.as_tensor([workout_id] * n).to(device)
    sids = torch.as_tensor([subject_id] * n).to(device)
    if history is not None:
        h = history.unsqueeze(0).float().to(device).repeat(n, 1, 1)
        hl = torch.as_tensor([history_length] * n).to(device)
    else:
        h, hl = None, None
    with torch.no_grad():
        out = model.forecast_batch(
            activity=win_act,
            times=win_t,
            workout_id=wids,
            subject_id=sids,
            history=h,
            history_length=hl,
        )
    pred_sum = torch.zeros(length, device=device)
    pred_count = torch.zeros(length, device=device)
    for j, start in enumerate(starts):
        end = min(start + out.size(1), length)
        n_steps = end - start
        pred_sum[start:end] += out[j, :n_steps]
        pred_count[start:end] += 1
    return (pred_sum / pred_count.clamp_min(1)).cpu().numpy()


def fig_mae_comparison(outdir: Path):
    """Bar chart: published notebook MAE vs standard-protocol mean workout MAE."""
    labels = [
        "Published\n(notebook MAE)",
        "As-published path\n(held-out, short window)",
        "Paper-faithful\n(full workout)",
        "Best engineering\n(full workout)",
    ]
    # Published: notebook ~5.07, paper states 5.2
    # As-published clean short-horizon from reeval/as-published
    # Paper-faithful and intensity-val from result.txt
    values = [5.2, 9.37, 7.42, 7.37]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.65, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAE (BPM)")
    ax.set_title("Heart-rate prediction error: published figure vs standard protocol")
    ax.set_ylim(0, max(values) * 1.25)
    ax.axhline(7.4, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(3.4, 7.55, "≈7.4 BPM standard-protocol level", fontsize=8, color="gray", va="bottom")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = outdir / "01_mae_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Wrote", path)


def load_model_and_data(args):
    if args.paper_faithful:
        args.physiological = True
        args.adafs = True
        args.adafs_variant = "paper"

    df = pd.read_feather(args.data)
    train_pop = df["in_train"]
    if args.sport:
        train_pop = train_pop & df["sport"].eq(args.sport)
    df, activity_columns = add_activity_features(
        df,
        args.feature_set,
        population_mask=train_pop,
        subject_id_column="userId",
        time_column="start_dt",
        sport=args.sport,
    )

    data_config = WorkoutDatasetConfig(
        subject_id_column="userId",
        workout_id_column="id",
        time_since_start_column="time_grid",
        time_of_start_column="start_dt",
        heart_rate_column="heart_rate",
        heart_rate_normalized_column="heart_rate_normalized",
        activity_columns=activity_columns,
        weather_columns=[],
        history_max_length=512,
        chunk_size=None,
        stride=None,
    )

    heldout_mask = ~df["in_train"]
    if args.sport:
        heldout_mask = heldout_mask & df["sport"].eq(args.sport)

    if args.history_source == "split":
        dataset = WorkoutDataset(df[heldout_mask], data_config)
        indices = list(range(len(dataset)))
    else:
        full = WorkoutDataset(df, data_config)
        indices = np.flatnonzero(heldout_mask.to_numpy()).tolist()
        dataset = Subset(full, indices)

    config = DBNConfig(
        data_config=data_config,
        seq_length=args.seq_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_adafs=args.adafs,
        use_physiological_head=args.physiological or args.paper_faithful,
        use_physiological_residual=args.residual,
        adafs_variant=args.adafs_variant,
        paper_faithful=args.paper_faithful,
    )
    model = DBNModel(config=config, workouts_info=df[["userId", "id"]])
    ckpt = Path(args.checkpoint) if args.checkpoint else EXAMPLES / "reeval" / args.name / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=config.device))
    model.to(config.device)
    model.eval()
    return model, dataset, config.device, args


def fig_workout_predictions(model, dataset, device, args, outdir: Path):
    rng = np.random.default_rng(args.seed)
    n = min(args.n_workouts, len(dataset))
    # Prefer mid-length workouts for readable plots
    lengths = []
    for i in range(min(len(dataset), 2000)):
        item = dataset[i]
        lengths.append((i, int(item["full_workout_length"])))
    mid = [i for i, L in lengths if 180 <= L <= 400]
    if len(mid) < n:
        mid = [i for i, _ in lengths]
    pick = rng.choice(mid, size=n, replace=False)

    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, pick):
        item = dataset[int(idx)]
        L = int(item["full_workout_length"])
        activity = torch.as_tensor(item["activity"][:L])
        times = torch.as_tensor(item["time"][:L])
        true = np.asarray(item["heart_rate"][:L], dtype=float)
        hist = torch.as_tensor(item["history"]) if item["history"] is not None else None
        hl = int(item["history_length"])
        pred = forecast_stitched(
            model,
            activity,
            times,
            item["workout_id"],
            item["subject_id"],
            hist,
            hl,
            device,
            args.seq_length,
            args.eval_stride,
        )
        t_min = np.arange(L) * 10.0 / 60.0  # 10 s grid → minutes
        mae = float(np.mean(np.abs(pred - true)))
        ax.plot(t_min, true, color="#1f77b4", linewidth=1.5, label="True HR")
        ax.plot(t_min, pred, color="#ff7f0e", linewidth=1.5, linestyle="--", label="Predicted HR")
        ax.fill_between(t_min, pred - 5, pred + 5, color="#ff7f0e", alpha=0.15)
        ax.set_ylabel("HR (BPM)")
        ax.set_title(
            f"User {item['subject_id']}  workout {item['workout_id']}  |  MAE {mae:.1f} BPM",
            fontsize=10,
        )
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (min)")
    fig.suptitle("Held-out run workouts: true vs predicted heart rate (stitched)", fontsize=12, y=1.01)
    fig.tight_layout()
    path = outdir / "02_workout_predictions.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", path)


def fig_error_scatter(model, dataset, device, args, outdir: Path):
    rng = np.random.default_rng(args.seed + 1)
    n = min(args.max_scatter, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)

    true_means, pred_means, maes = [], [], []
    for idx in indices:
        item = dataset[int(idx)]
        L = int(item["full_workout_length"])
        if L < 32:
            continue
        activity = torch.as_tensor(item["activity"][:L])
        times = torch.as_tensor(item["time"][:L])
        true = np.asarray(item["heart_rate"][:L], dtype=float)
        hist = torch.as_tensor(item["history"]) if item["history"] is not None else None
        pred = forecast_stitched(
            model,
            activity,
            times,
            item["workout_id"],
            item["subject_id"],
            hist,
            int(item["history_length"]),
            device,
            args.seq_length,
            args.eval_stride,
        )
        true_means.append(true.mean())
        pred_means.append(pred.mean())
        maes.append(float(np.mean(np.abs(pred - true))))

    true_means = np.asarray(true_means)
    pred_means = np.asarray(pred_means)
    maes = np.asarray(maes)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.scatter(true_means, pred_means, s=8, alpha=0.35, c="#4C72B0", edgecolors="none")
    lims = [min(true_means.min(), pred_means.min()) - 5, max(true_means.max(), pred_means.max()) + 5]
    ax.plot(lims, lims, "k--", linewidth=1, label="Ideal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True mean HR (BPM)")
    ax.set_ylabel("Predicted mean HR (BPM)")
    ax.set_title("Workout mean HR (held-out sample)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1]
    ax.hist(maes, bins=40, color="#55A868", edgecolor="black", linewidth=0.3)
    ax.axvline(np.median(maes), color="#C44E52", linestyle="--", label=f"Median {np.median(maes):.1f}")
    ax.axvline(np.mean(maes), color="#4C72B0", linestyle="-", label=f"Mean {np.mean(maes):.1f}")
    ax.set_xlabel("Per-workout MAE (BPM)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-workout MAE")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle(f"Standard protocol | run={args.name} | n={len(maes)} workouts", fontsize=11)
    fig.tight_layout()
    path = outdir / "03_error_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Wrote", path)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1 does not need GPU
    fig_mae_comparison(outdir)

    model, dataset, device, args = load_model_and_data(args)
    fig_workout_predictions(model, dataset, device, args, outdir)
    fig_error_scatter(model, dataset, device, args, outdir)

    # Short caption file for the public package
    caption = outdir / "FIGURES.md"
    caption.write_text(
        """# Public figures (standard evaluation protocol)

Generated by `examples/plot_public_figures.py`.

| File | Description |
|------|-------------|
| `01_mae_comparison.png` | Published notebook MAE (5.2 BPM) vs reevaluation MAE under the standard protocol |
| `02_workout_predictions.png` | True and predicted heart rate on held-out run workouts (stitched full session) |
| `03_error_scatter.png` | Mean HR scatter and per-workout MAE distribution on a held-out sample |

## Notes

1. Figure 1 mixes different metric definitions on purpose. The left bar is the paper figure. The right bars use held-out evaluation.
2. Figures 2 and 3 use the checkpoint named in the script (`--name`, default `paper-faithful-run-val`).
3. ±5 BPM band in Figure 2 is a display band only. It is not a model uncertainty estimate.
"""
    )
    print("Wrote", caption)


if __name__ == "__main__":
    main()
