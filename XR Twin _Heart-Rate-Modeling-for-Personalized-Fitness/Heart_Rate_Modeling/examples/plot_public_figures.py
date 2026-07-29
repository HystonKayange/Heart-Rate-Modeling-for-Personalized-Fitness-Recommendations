#!/usr/bin/env python3
"""
Build public figures for the clean reevaluation package.

Outputs (default: examples/figures/public/):
  01_mae_comparison.png       — protocol-specific DBN/ODE MAE comparison
  02_workout_predictions.png  — true vs predicted HR (held-out samples)
  03_error_scatter.png        — predicted mean HR vs true mean HR (sample)
  04_cohort_bias.png          — MAE and signed bias by average-HR cohort

Example:
  python plot_public_figures.py \\
    --name run-huber-128-delta12-intensity-trainprior-val \\
    --no-paper-faithful --physiological --residual \\
    --sport run --history-source train-prior \\
    --seq-length 128 --feature-set run_intensity
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

from Model.activity_features import add_activity_features, chronological_train_val_masks
from Model.data import WorkoutDataset, WorkoutDatasetConfig, workout_dataset_collate_fn
from Model.dbn import DBNConfig, DBNModel, load_compatible_state_dict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="run-huber-128-delta12-intensity-trainprior-val")
    p.add_argument("--data", default=str(PROJECT / "output" / "endomondo_filtered.feather"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--outdir", default=str(EXAMPLES / "figures" / "public"))
    p.add_argument("--sport", default="run")
    p.add_argument("--history-source", choices=["split", "train-prior", "all-prior"], default="train-prior")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seq-length", type=int, default=128)
    p.add_argument("--feature-set", choices=["basic", "run_intensity", "run_personal"], default="run_intensity")
    p.add_argument("--paper-faithful", action="store_true", default=False)
    p.add_argument("--no-paper-faithful", action="store_false", dest="paper_faithful")
    p.add_argument("--physiological", action="store_true", default=True)
    p.add_argument("--no-physiological", action="store_false", dest="physiological")
    p.add_argument("--residual", action="store_true", default=True)
    p.add_argument("--no-residual", action="store_false", dest="residual")
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


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def result_float(path: Path, key: str, fallback: float) -> float:
    values = read_key_values(path)
    try:
        return float(values[key])
    except (KeyError, TypeError, ValueError):
        return fallback


def fig_mae_comparison(outdir: Path):
    """Protocol-specific bar chart for published, DBN, and ODE MAE values."""
    strict_result = EXAMPLES / "reeval" / "run-huber-128-delta12-intensity-trainprior-val" / "result.txt"
    sequential_result = EXAMPLES / "reeval" / "run-huber-128-delta12-intensity-val" / "result.txt"
    ode_result = (
        PROJECT.parents[1]
        / "baselines"
        / "ml-heart-rate-models-main"
        / "examples"
        / "reeval_ode"
        / "ode-run-clean-val"
        / "result.txt"
    )
    labels = [
        "Published DBN\nnotebook metric",
        "DBN strict\ntrain-prior",
        "DBN sequential\nhistory",
        "Hybrid ODE\nFitRec rerun",
    ]
    values = [
        5.2,
        result_float(strict_result, "full_mean_workout_MAE", 8.1225),
        result_float(sequential_result, "full_mean_workout_MAE", 7.3739),
        result_float(ode_result, "full_mean_workout_MAE", 8.7876),
    ]
    colors = ["#8C8C8C", "#2F6F9F", "#5B8E7D", "#B25D3C"]

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.65, edgecolor="black", linewidth=0.6)
    bars[0].set_hatch("//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAE (BPM, lower is better)")
    ax.set_title("Protocol-specific heart-rate prediction error")
    ax.set_ylim(0, max(values) * 1.25)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(
        0,
        values[0] + 1.05,
        "paper-period\nnotebook output",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4A4A4A",
    )
    ax.annotate(
        "Clean FitRec full-workout comparison",
        xy=(2.5, max(values[1:]) + 0.15),
        xytext=(1.5, max(values) * 1.13),
        arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.8},
        ha="center",
        fontsize=9,
        color="#333333",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
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
    in_train_mask = df["in_train"]
    if args.sport:
        in_train_mask = in_train_mask & df["sport"].eq(args.sport)
    train_fit_mask, _ = chronological_train_val_masks(
        df,
        in_train_mask,
        subject_id_column="userId",
        time_column="start_dt",
        val_fraction=args.val_fraction,
    )
    df, activity_columns = add_activity_features(
        df,
        args.feature_set,
        population_mask=train_fit_mask if train_fit_mask.any() else in_train_mask,
        subject_id_column="userId",
        time_column="start_dt",
        sport=args.sport,
    )
    history_allowed_column = "_history_allowed_train_fit"
    df[history_allowed_column] = train_fit_mask

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
    elif args.history_source == "train-prior":
        strict_config = dataclasses.replace(data_config, history_allowed_column=history_allowed_column)
        eval_pool_mask = train_fit_mask | heldout_mask
        full = WorkoutDataset(df[eval_pool_mask], strict_config)
        eval_pool_positions = np.flatnonzero(eval_pool_mask.to_numpy())
        heldout_global = set(np.flatnonzero(heldout_mask.to_numpy()).tolist())
        indices = [i for i, g in enumerate(eval_pool_positions) if g in heldout_global]
        dataset = Subset(full, indices)
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
    load_compatible_state_dict(model, torch.load(ckpt, map_location=config.device))
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

    fig.suptitle(f"Strict train-prior held-out sample (n={len(maes)} workouts)", fontsize=11)
    fig.tight_layout()
    path = outdir / "03_error_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Wrote", path)


def fig_cohort_bias(outdir: Path):
    analysis_dir = (
        EXAMPLES
        / "reeval"
        / "run-huber-128-delta12-intensity-trainprior-val"
        / "analysis-train-prior-run"
    )
    cohort_path = analysis_dir / "cohort_summary.csv"
    segment_path = analysis_dir / "segment_summary.csv"
    if not cohort_path.exists() or not segment_path.exists():
        print("Skipping cohort figure; analysis CSV files are missing.")
        return

    cohort = pd.read_csv(cohort_path)
    cohort = cohort[cohort["grouping"].eq("avg_hr_bin")].copy()
    order = ["<120", "120-150", "150-170", ">=170"]
    cohort["value"] = pd.Categorical(cohort["value"], categories=order, ordered=True)
    cohort = cohort.sort_values("value")

    segment = pd.read_csv(segment_path)
    segment_order = ["first_2min", "min_2_to_10", "after_10min", "hr_lt_120", "hr_120_150", "hr_150_170", "hr_ge_170"]
    segment = segment.set_index("segment").reindex(segment_order).dropna(subset=["pooled_mae"]).reset_index()
    segment_labels = ["0-2 min", "2-10 min", ">10 min", "HR <120", "HR 120-150", "HR 150-170", "HR >=170"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax = axes[0]
    x = np.arange(len(cohort))
    mae_bars = ax.bar(x - 0.18, cohort["mean_mae"], width=0.36, color="#2F6F9F", label="Mean MAE")
    bias_colors = ["#B25D3C" if v > 0 else "#5B8E7D" for v in cohort["mean_bias"]]
    bias_bars = ax.bar(x + 0.18, cohort["mean_bias"], width=0.36, color=bias_colors, label="Signed bias")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in cohort["value"]])
    ax.set_xlabel("Workout average HR cohort (BPM)")
    ax.set_ylabel("BPM")
    ax.set_title("Cohort MAE and signed bias")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    for bar in list(mae_bars) + list(bias_bars):
        v = bar.get_height()
        va = "bottom" if v >= 0 else "top"
        y = v + (0.35 if v >= 0 else -0.35)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{v:.1f}", ha="center", va=va, fontsize=8)

    ax = axes[1]
    y = np.arange(len(segment))
    ax.barh(y, segment["pooled_mae"], color="#6B7A8F", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(segment_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Pooled MAE (BPM)")
    ax.set_title("Error by time and HR segment")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
    for yi, v in zip(y, segment["pooled_mae"]):
        ax.text(v + 0.25, yi, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, max(segment["pooled_mae"]) * 1.2)

    fig.suptitle("Strict train-prior diagnostics on held-out run workouts", fontsize=12)
    fig.tight_layout()
    path = outdir / "04_cohort_bias.png"
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
    fig_cohort_bias(outdir)

    # Short caption file for the public package
    caption = outdir / "FIGURES.md"
    caption.write_text(
        """# Public figures (protocol-specific reevaluation)

Generated by `examples/plot_public_figures.py`.

| File | Description |
|------|-------------|
| `01_mae_comparison.png` | Published notebook MAE, DBN reevaluation results, and clean FitRec ODE baseline |
| `02_workout_predictions.png` | True and predicted heart rate on held-out run workouts (stitched full session) |
| `03_error_scatter.png` | Mean HR scatter and per-workout MAE distribution on a held-out sample |
| `04_cohort_bias.png` | Strict train-prior cohort bias and segment error diagnostics |

## Notes

1. Figure 1 mixes different metric definitions on purpose. The first bar is the paper-period notebook result. The other bars use held-out full-workout reevaluation.
2. Figures 2 and 3 use the checkpoint named in the script (`--name`).
3. ±5 BPM band in Figure 2 is a display band only. It is not a model uncertainty estimate.
4. Figure 4 uses the strict train-prior analysis directory.
"""
    )
    print("Wrote", caption)


if __name__ == "__main__":
    main()
