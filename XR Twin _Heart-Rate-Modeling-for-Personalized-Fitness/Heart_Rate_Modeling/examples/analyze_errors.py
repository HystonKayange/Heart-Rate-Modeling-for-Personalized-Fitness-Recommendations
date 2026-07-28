"""
Full-workout error analysis for clean held-out checkpoints.

The script stitches overlapping 64-step predictions across each held-out workout
and writes cohort summaries that show where the MAE is coming from.

Example:
    python analyze_errors.py --name physiological-residual --physiological --residual
"""
import argparse
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, Subset

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXAMPLES_DIR.parent
sys.path.append(str(PROJECT_DIR))

from Model.data import WorkoutDataset, WorkoutDatasetConfig, workout_dataset_collate_fn
from Model.dbn import DBNConfig, DBNModel
from Model.activity_features import add_activity_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name under examples/reeval")
    p.add_argument("--adafs", action="store_true", help="enable adaptive feature selection")
    p.add_argument("--physiological", action="store_true", help="enable the Eq. 9 head")
    p.add_argument("--residual", action="store_true", help="add the physiological residual head")
    p.add_argument(
        "--contextual-residual",
        action="store_true",
        help="feed state, embeddings, activity, and time into the physiological residual head",
    )
    p.add_argument(
        "--personalized-physio",
        action="store_true",
        help="P2: subject-stable physio params + embedding-conditioned intensity",
    )
    p.add_argument("--physio-subject-stable", action="store_true")
    p.add_argument("--intensity-embedding", action="store_true")
    p.add_argument("--data", default=str(PROJECT_DIR / "output" / "endomondo_filtered.feather"))
    p.add_argument("--checkpoint", default=None, help="checkpoint path; defaults to reeval/<name>/best_model.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-length", type=int, default=None, help="model window length; inferred from result.txt when omitted")
    p.add_argument("--feature-set", choices=["basic", "run_intensity", "run_personal"], default=None)
    p.add_argument("--eval-stride", type=int, default=32)
    p.add_argument("--sport", default=None, help="optional held-out sport filter, e.g. run or bike")
    p.add_argument("--limit", type=int, default=None, help="debug only: score first N held-out workouts")
    p.add_argument(
        "--history-source",
        choices=["split", "all-prior"],
        default="all-prior",
        help="build held-out history from held-out rows only, or from all chronological prior workouts",
    )
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    if args.personalized_physio:
        args.physiological = True
        args.physio_subject_stable = True
        args.intensity_embedding = True
    if args.residual and not args.physiological:
        p.error("--residual requires --physiological")
    if args.contextual_residual and not args.residual:
        p.error("--contextual-residual requires --residual")
    return args


def make_data_config(activity_columns):
    return WorkoutDatasetConfig(
        subject_id_column="userId",
        workout_id_column="id",
        time_since_start_column="time_grid",
        time_of_start_column="start_dt",
        heart_rate_column="heart_rate",
        heart_rate_normalized_column="heart_rate_normalized",
        activity_columns=activity_columns,
        weather_columns=[],
        history_max_length=512,
    )


def make_heldout_dataloader(df, data_config, batch_size, history_source, limit, sport):
    eval_config = dataclasses.replace(data_config, chunk_size=None, stride=None)
    heldout_mask = ~df["in_train"]
    if sport is not None:
        heldout_mask = heldout_mask & df["sport"].eq(sport)
    if history_source == "split":
        dataset = WorkoutDataset(df[heldout_mask], eval_config)
        if limit is not None:
            dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    else:
        full_dataset = WorkoutDataset(df, eval_config)
        heldout_indices = np.flatnonzero(heldout_mask.to_numpy()).tolist()
        if limit is not None:
            heldout_indices = heldout_indices[:limit]
        dataset = Subset(full_dataset, heldout_indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=workout_dataset_collate_fn,
        shuffle=False,
    )


def build_model(df, data_config, args):
    config = DBNConfig(
        data_config=data_config,
        seq_length=args.seq_length,
        learning_rate=1e-3,
        seed=0,
        n_epochs=1,
        lstm_hidden_dim=128,
        lstm_layers=2,
        dbn_hidden_dim=64,
        subject_embedding_dim=8,
        encoder_embedding_dim=8,
        dropout=0.2,
        use_adafs=args.adafs,
        use_physiological_head=args.physiological,
        use_physiological_residual=args.residual,
        use_contextual_residual=args.contextual_residual,
        physio_subject_stable_params=args.physio_subject_stable,
        intensity_use_embedding=args.intensity_embedding,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return DBNModel(config=config, workouts_info=df[["userId", "id"]])


def window_starts(length, seq_length, stride):
    if length <= seq_length:
        return [0]
    starts = list(range(0, length - seq_length, stride))
    if not starts or starts[-1] + seq_length < length:
        starts.append(length - seq_length)
    return starts


def forecast_stitched_workout(model, batch, i, length, device, seq_length, stride):
    activity = torch.as_tensor(batch["activity"][i, :length]).float().to(device)
    times = torch.as_tensor(batch["time"][i, :length]).float().to(device)
    starts = window_starts(length, seq_length, stride)

    window_activity = torch.stack([activity[s : s + seq_length] for s in starts])
    window_times = torch.stack([times[s : s + seq_length] for s in starts])
    n_windows = len(starts)

    workout_id = torch.as_tensor(batch["workout_id"][i]).to(device).repeat(n_windows)
    subject_id = torch.as_tensor(batch["subject_id"][i]).to(device).repeat(n_windows)
    if batch["history"] is not None:
        history = torch.as_tensor(batch["history"][i : i + 1]).float().to(device).repeat(n_windows, 1, 1)
        history_length = torch.as_tensor(batch["history_length"][i]).to(device).repeat(n_windows)
    else:
        history = None
        history_length = None

    out = model.forecast_batch(
        activity=window_activity,
        times=window_times,
        workout_id=workout_id,
        subject_id=subject_id,
        history=history,
        history_length=history_length,
    )

    pred_sum = torch.zeros(length, device=device)
    pred_count = torch.zeros(length, device=device)
    for j, start in enumerate(starts):
        end = min(start + out.size(1), length)
        n = end - start
        pred_sum[start:end] += out[j, :n]
        pred_count[start:end] += 1

    if torch.any(pred_count == 0):
        raise RuntimeError("stitched evaluation left uncovered time steps")
    return pred_sum / pred_count


def masked_mean(values, mask):
    if not np.any(mask):
        return np.nan
    return float(np.mean(values[mask]))


def add_segment(segments, name, abs_err, sq_err, mask):
    if not np.any(mask):
        return
    segments.setdefault(name, {"abs_sum": 0.0, "sq_sum": 0.0, "n": 0})
    segments[name]["abs_sum"] += float(abs_err[mask].sum())
    segments[name]["sq_sum"] += float(sq_err[mask].sum())
    segments[name]["n"] += int(mask.sum())


def build_bins(per_workout):
    per_workout["duration_bin"] = pd.cut(
        per_workout["duration_min"],
        bins=[0, 20, 40, 60, np.inf],
        labels=["<=20m", "20-40m", "40-60m", ">60m"],
        include_lowest=True,
    )
    per_workout["history_bin"] = pd.cut(
        per_workout["history_length"],
        bins=[0, 1, 128, 512, np.inf],
        labels=["none_or_dummy", "2-128", "129-512", ">512"],
        include_lowest=True,
    )
    per_workout["avg_hr_bin"] = pd.cut(
        per_workout["true_mean_hr"],
        bins=[0, 120, 150, 170, np.inf],
        labels=["<120", "120-150", "150-170", ">=170"],
        include_lowest=True,
    )


def summarize_group(per_workout, column):
    rows = []
    for value, group in per_workout.groupby(column, dropna=False, observed=False):
        rows.append(
            {
                "grouping": column,
                "value": str(value),
                "n_workouts": len(group),
                "mean_mae": group["mae"].mean(),
                "median_mae": group["mae"].median(),
                "p75_mae": group["mae"].quantile(0.75),
                "p90_mae": group["mae"].quantile(0.90),
                "mean_bias": group["bias"].mean(),
                "mean_duration_min": group["duration_min"].mean(),
            }
        )
    return rows


def analyze(model, dataloader, df, args):
    meta = df.set_index("id")
    rows = []
    segments = {}
    all_abs, all_sq = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="stitch+analyze"):
            heart_rate = torch.as_tensor(batch["heart_rate"]).float().to(model.config.device)
            lengths = torch.as_tensor(batch["full_workout_length"]).long().tolist()
            for i, length in enumerate(lengths):
                pred = forecast_stitched_workout(
                    model,
                    batch,
                    i,
                    length,
                    model.config.device,
                    model.config.seq_length,
                    args.eval_stride,
                )
                true = heart_rate[i, :length]
                err = (pred - true).cpu().numpy()
                true_np = true.cpu().numpy()
                pred_np = pred.cpu().numpy()
                abs_err = np.abs(err)
                sq_err = err ** 2
                all_abs.append(abs_err)
                all_sq.append(sq_err)

                step = np.arange(length)
                first_2 = step < 12
                min_2_to_10 = (step >= 12) & (step < 60)
                after_10 = step >= 60
                add_segment(segments, "first_2min", abs_err, sq_err, first_2)
                add_segment(segments, "min_2_to_10", abs_err, sq_err, min_2_to_10)
                add_segment(segments, "after_10min", abs_err, sq_err, after_10)
                add_segment(segments, "hr_lt_120", abs_err, sq_err, true_np < 120)
                add_segment(segments, "hr_120_150", abs_err, sq_err, (true_np >= 120) & (true_np < 150))
                add_segment(segments, "hr_150_170", abs_err, sq_err, (true_np >= 150) & (true_np < 170))
                add_segment(segments, "hr_ge_170", abs_err, sq_err, true_np >= 170)

                workout_id = int(batch["workout_id"][i])
                subject_id = int(batch["subject_id"][i])
                sport = meta.at[workout_id, "sport"] if "sport" in meta.columns else "unknown"
                activity = np.asarray(batch["activity"][i, :length])
                rows.append(
                    {
                        "workout_id": workout_id,
                        "subject_id": subject_id,
                        "sport": sport,
                        "length_steps": length,
                        "duration_min": length / 6.0,
                        "history_length": int(batch["history_length"][i]),
                        "mae": float(abs_err.mean()),
                        "rmse": float(np.sqrt(sq_err.mean())),
                        "bias": float(err.mean()),
                        "true_mean_hr": float(true_np.mean()),
                        "pred_mean_hr": float(pred_np.mean()),
                        "true_max_hr": float(true_np.max()),
                        "mean_speed_h": float(activity[:, 0].mean()) if activity.shape[1] > 0 else np.nan,
                        "mean_speed_v": float(activity[:, 1].mean()) if activity.shape[1] > 1 else np.nan,
                        "first_2min_mae": masked_mean(abs_err, first_2),
                        "min_2_to_10_mae": masked_mean(abs_err, min_2_to_10),
                        "after_10min_mae": masked_mean(abs_err, after_10),
                        "hr_lt_120_mae": masked_mean(abs_err, true_np < 120),
                        "hr_120_150_mae": masked_mean(abs_err, (true_np >= 120) & (true_np < 150)),
                        "hr_150_170_mae": masked_mean(abs_err, (true_np >= 150) & (true_np < 170)),
                        "hr_ge_170_mae": masked_mean(abs_err, true_np >= 170),
                    }
                )

    per_workout = pd.DataFrame(rows)
    build_bins(per_workout)

    segment_rows = []
    for name, values in segments.items():
        segment_rows.append(
            {
                "segment": name,
                "n_steps": values["n"],
                "pooled_mae": values["abs_sum"] / values["n"],
                "pooled_rmse": np.sqrt(values["sq_sum"] / values["n"]),
            }
        )
    segment_summary = pd.DataFrame(segment_rows).sort_values("pooled_mae", ascending=False)

    cohort_rows = []
    for column in ["sport", "duration_bin", "history_bin", "avg_hr_bin"]:
        cohort_rows.extend(summarize_group(per_workout, column))
    cohort_summary = pd.DataFrame(cohort_rows).sort_values(["grouping", "mean_mae"], ascending=[True, False])

    all_abs = np.concatenate(all_abs)
    all_sq = np.concatenate(all_sq)
    headline = {
        "mean_workout_mae": float(per_workout["mae"].mean()),
        "median_workout_mae": float(per_workout["mae"].median()),
        "pooled_mae": float(all_abs.mean()),
        "pooled_rmse": float(np.sqrt(all_sq.mean())),
        "n_workouts": int(len(per_workout)),
        "n_steps": int(len(all_abs)),
    }
    return headline, per_workout, cohort_summary, segment_summary


def infer_seq_length(result_path):
    if not result_path.exists():
        return 64
    for token in result_path.read_text().split():
        if token.startswith("seq_length="):
            return int(token.split("=", 1)[1])
    return 64


def infer_feature_set(result_path):
    if not result_path.exists():
        return "basic"
    for token in result_path.read_text().split():
        if token.startswith("feature_set="):
            return token.split("=", 1)[1]
    return "basic"


def infer_bool(result_path, key, default=False):
    if not result_path.exists():
        return default
    prefix = f"{key}="
    for token in result_path.read_text().split():
        if token.startswith(prefix):
            return token.split("=", 1)[1] == "True"
    return default


def main():
    args = parse_args()
    run_dir = EXAMPLES_DIR / "reeval" / args.name
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"
    if args.seq_length is None:
        args.seq_length = infer_seq_length(run_dir / "result.txt")
    if args.feature_set is None:
        args.feature_set = infer_feature_set(run_dir / "result.txt")
    if not args.contextual_residual:
        args.contextual_residual = infer_bool(run_dir / "result.txt", "contextual_residual")
    if not args.physio_subject_stable:
        args.physio_subject_stable = infer_bool(run_dir / "result.txt", "physio_subject_stable")
    if not args.intensity_embedding:
        args.intensity_embedding = infer_bool(run_dir / "result.txt", "intensity_embedding")
    if args.physio_subject_stable or args.intensity_embedding:
        args.physiological = True
    sport_suffix = args.sport if args.sport is not None else "all"
    outdir = Path(args.outdir) if args.outdir else run_dir / f"analysis-{args.history_source}-{sport_suffix}"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_feather(args.data)
    # Cold-start population stats from train only — never held-out labels.
    train_population = df["in_train"]
    if args.sport is not None:
        train_population = train_population & df["sport"].eq(args.sport)
    df, activity_columns = add_activity_features(
        df,
        args.feature_set,
        population_mask=train_population,
        subject_id_column="userId",
        time_column="start_dt",
        sport=args.sport,
    )
    data_config = make_data_config(activity_columns)
    dataloader = make_heldout_dataloader(
        df, data_config, args.batch_size, args.history_source, args.limit, args.sport
    )
    model = build_model(df, data_config, args)
    model.load_state_dict(torch.load(checkpoint, map_location=model.config.device))

    headline, per_workout, cohort_summary, segment_summary = analyze(model, dataloader, df, args)

    per_workout.to_csv(outdir / "per_workout_errors.csv", index=False)
    per_workout.sort_values("mae", ascending=False).head(50).to_csv(outdir / "worst_workouts.csv", index=False)
    cohort_summary.to_csv(outdir / "cohort_summary.csv", index=False)
    segment_summary.to_csv(outdir / "segment_summary.csv", index=False)

    with open(outdir / "summary.txt", "w") as f:
        f.write(f"name={args.name}\n")
        f.write(f"checkpoint={checkpoint}\n")
        f.write(f"history_source={args.history_source}\n")
        f.write(f"sport={args.sport or 'all'}\n")
        f.write(f"seq_length={args.seq_length}\n")
        f.write(f"feature_set={args.feature_set}\n")
        f.write(f"contextual_residual={args.contextual_residual}\n")
        f.write(f"physio_subject_stable={args.physio_subject_stable}\n")
        f.write(f"intensity_embedding={args.intensity_embedding}\n")
        f.write(f"activity_columns={','.join(activity_columns)}\n")
        f.write(f"eval_stride={args.eval_stride}\n")
        for key, value in headline.items():
            f.write(f"{key}={value:.4f}\n" if isinstance(value, float) else f"{key}={value}\n")

    print("\nAnalysis written to", outdir)
    print(
        f"mean workout MAE={headline['mean_workout_mae']:.2f} BPM | "
        f"median={headline['median_workout_mae']:.2f} | pooled={headline['pooled_mae']:.2f}"
    )


if __name__ == "__main__":
    main()
