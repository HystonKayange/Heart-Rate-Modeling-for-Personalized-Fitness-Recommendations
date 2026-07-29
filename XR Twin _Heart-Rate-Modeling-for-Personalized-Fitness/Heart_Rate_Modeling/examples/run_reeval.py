"""
Clean-split re-evaluation.

Trains on a chronological subset of df[in_train], validates on the remaining
in_train workouts (per-user, by time), and reports only on df[~in_train].
Checkpoint selection and LR scheduling use validation only — never held-out.

    python run_reeval.py --name as-published
    python run_reeval.py --name as-described --adafs --physiological
    python run_reeval.py --name physiological-residual --physiological --residual
    python run_reeval.py --name run-p2-physio-val --physiological --residual \
        --personalized-physio --sport run --history-source all-prior \
        --feature-set run_intensity --seq-length 128 --loss huber \
        --huber-delta 12 --weight-decay 1e-4 --full-workout

Checkpoints are written under examples/reeval/<name>/ so the paper-period
best_model.pt artifact is never overwritten.
"""
import argparse
import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from Model.data import WorkoutDataset, WorkoutDatasetConfig, workout_dataset_collate_fn
from Model.dbn import DBNConfig, DBNModel, load_compatible_state_dict
from Model.trainer import Trainer
from Model.activity_features import add_activity_features, chronological_train_val_masks

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="run name; output goes to reeval/<name>/")
p.add_argument("--adafs", action="store_true", help="enable adaptive feature selection")
p.add_argument(
    "--adafs-variant",
    choices=["legacy", "paper"],
    default="legacy",
    help="legacy=flattened T*F controller; paper=§4.4 controller on latent z (per-step α)",
)
p.add_argument(
    "--paper-faithful",
    action="store_true",
    help=(
        "Paper-text stack: Eq. 9 physiological head + paper AdaFS (§4.4 on latent z). "
        "Residual off unless --residual."
    ),
)
p.add_argument("--physiological", action="store_true", help="enable the Eq. 9 head")
p.add_argument("--residual", action="store_true", help="add a linear residual correction on top of the Eq. 9 head")
p.add_argument(
    "--contextual-residual",
    action="store_true",
    help="feed state, embeddings, activity, and time into the physiological residual head",
)
p.add_argument(
    "--personalized-physio",
    action="store_true",
    help=(
        "P2: subject-stable A/B/HRmin/range (from embeddings only) and "
        "I(t)=f(activity, embeddings). Implies --physiological. Use with --residual."
    ),
)
p.add_argument(
    "--physio-subject-stable",
    action="store_true",
    help="A/B/HRmin/range from subject+history embeddings only (not per-step state)",
)
p.add_argument(
    "--intensity-embedding",
    action="store_true",
    help="intensity I(t) depends on activity and subject/history embeddings",
)
p.add_argument("--eval-only", action="store_true", help="skip training and evaluate reeval/<name>/best_model.pt")
p.add_argument("--full-workout", action="store_true", help="also score stitched predictions over full held-out workouts")
p.add_argument("--eval-stride", type=int, default=32, help="stride for stitched full-workout evaluation")
p.add_argument("--sport", default=None, help="optional sport filter, e.g. run or bike")
p.add_argument(
    "--history-source",
    choices=["split", "train-prior", "all-prior"],
    default="train-prior",
    help=(
        "split=held-out rows only; train-prior=train_fit rows only; "
        "all-prior=all chronological prior rows, including earlier val/held-out rows"
    ),
)
p.add_argument(
    "--val-fraction",
    type=float,
    default=0.15,
    help="fraction of each user's in_train workouts (by time) held out for validation/checkpointing",
)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--seq-length", type=int, default=64, help="training/prediction window length")
p.add_argument("--train-stride", type=int, default=None, help="training chunk stride; defaults to seq_length // 2")
p.add_argument(
    "--feature-set",
    choices=["basic", "run_intensity", "run_personal"],
    default="basic",
)
p.add_argument("--loss", choices=["mse_sum", "mse", "mae", "huber"], default="mse_sum")
p.add_argument("--huber-delta", type=float, default=10.0)
p.add_argument("--low-hr-weight", type=float, default=1.0)
p.add_argument("--high-hr-weight", type=float, default=1.0)
p.add_argument("--warmup-weight", type=float, default=1.0)
p.add_argument("--low-hr-threshold", type=float, default=120.0)
p.add_argument("--high-hr-threshold", type=float, default=170.0)
p.add_argument("--warmup-steps", type=int, default=12, help="10-second steps to reweight at workout/chunk start")
p.add_argument("--weight-decay", type=float, default=0.0)
p.add_argument(
    "--mean-bias-weight",
    type=float,
    default=0.0,
    help="P3: add lambda * |mean(pred)-mean(true)| per chunk to the step loss (0 disables)",
)
p.add_argument("--data", default="../output/endomondo_filtered.feather")
args = p.parse_args()

if args.paper_faithful:
    args.physiological = True
    args.adafs = True
    args.adafs_variant = "paper"
if args.personalized_physio:
    args.physiological = True
    args.physio_subject_stable = True
    args.intensity_embedding = True
if args.residual and not args.physiological:
    p.error("--residual requires --physiological")
if args.contextual_residual and not args.residual:
    p.error("--contextual-residual requires --residual")
if (args.physio_subject_stable or args.intensity_embedding) and not args.physiological:
    p.error("--physio-subject-stable / --intensity-embedding require --physiological "
            "(or pass --personalized-physio / --paper-faithful)")
if args.physio_subject_stable and not args.residual and not args.paper_faithful:
    # Without residual, the transition state is unused by the emission path when
    # A/B/bounds ignore state — dynamics would not train.
    print("WARNING: --physio-subject-stable without --residual leaves transition "
          "state unused by the head; residual is strongly recommended.")
if not 0.0 <= args.val_fraction < 1.0:
    p.error("--val-fraction must be in [0, 1)")
if args.train_stride is None:
    args.train_stride = max(1, args.seq_length // 2)

outdir = os.path.join("reeval", args.name)
os.makedirs(outdir, exist_ok=True)

df = pd.read_feather(args.data)
print(f"{len(df):,} workouts / {df['userId'].nunique():,} users")

# Base splits from the published in_train flag, optionally sport-filtered.
in_train_mask = df["in_train"].copy()
heldout_mask = ~df["in_train"]
if args.sport is not None:
    sport_mask = df["sport"].eq(args.sport)
    in_train_mask = in_train_mask & sport_mask
    heldout_mask = heldout_mask & sport_mask

# P0: chronological per-user train_fit / validation inside in_train only.
train_fit_mask, val_mask = chronological_train_val_masks(
    df,
    in_train_mask,
    subject_id_column="userId",
    time_column="start_dt",
    val_fraction=args.val_fraction,
)

# P1: features. Population cold-start stats from train_fit only (never held-out).
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

print(
    f"train_fit {train_fit_mask.sum():,}  |  val {val_mask.sum():,}  |  held-out {heldout_mask.sum():,}"
)
print(
    f"config: adafs={args.adafs}  adafs_variant={args.adafs_variant}  "
    f"paper_faithful={args.paper_faithful}  physiological={args.physiological}  "
    f"residual={args.residual}  contextual_residual={args.contextual_residual}  "
    f"physio_subject_stable={args.physio_subject_stable}  "
    f"intensity_embedding={args.intensity_embedding}  "
    f"epochs={args.epochs}  eval_only={args.eval_only}  "
    f"history_source={args.history_source}  sport={args.sport or 'all'}  "
    f"val_fraction={args.val_fraction}\n"
    f"train: seq_length={args.seq_length} stride={args.train_stride} loss={args.loss} "
    f"feature_set={args.feature_set} activity_dim={len(activity_columns)} "
    f"warmup_weight={args.warmup_weight} low_hr_weight={args.low_hr_weight} "
    f"high_hr_weight={args.high_hr_weight} weight_decay={args.weight_decay} "
    f"mean_bias_weight={args.mean_bias_weight}\n"
)

data_config_train = WorkoutDatasetConfig(
    subject_id_column="userId",
    workout_id_column="id",
    time_since_start_column="time_grid",
    time_of_start_column="start_dt",
    heart_rate_column="heart_rate",
    heart_rate_normalized_column="heart_rate_normalized",
    activity_columns=activity_columns,
    weather_columns=[],
    history_max_length=512,
    chunk_size=args.seq_length,
    stride=args.train_stride,
)
data_config_eval = dataclasses.replace(data_config_train, chunk_size=None, stride=None)
data_config_train_prior_eval = dataclasses.replace(
    data_config_eval,
    history_allowed_column=history_allowed_column,
)

train_dataset = WorkoutDataset(df[train_fit_mask], data_config_train)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    collate_fn=workout_dataset_collate_fn,
    shuffle=True,
    drop_last=True,
)

# Validation: full workouts from val split; history from train_fit priors only.
if val_mask.any():
    val_pool = WorkoutDataset(df[in_train_mask], data_config_train_prior_eval)
    in_train_positions = np.flatnonzero(in_train_mask.to_numpy())
    # Map full-df row positions that are val into positions within the in_train pool.
    val_global = set(np.flatnonzero(val_mask.to_numpy()).tolist())
    val_local = [i for i, g in enumerate(in_train_positions) if g in val_global]
    val_dataset = Subset(val_pool, val_local)
else:
    # val_fraction=0: fall back to a non-shuffled train eval (not ideal; avoid in real runs).
    val_dataset = WorkoutDataset(df[train_fit_mask], data_config_eval)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    collate_fn=workout_dataset_collate_fn,
    shuffle=False,
)

# Held-out test: used only for final reporting, never for checkpoint selection.
if args.history_source == "split":
    test_dataset = WorkoutDataset(df[heldout_mask], data_config_eval)
elif args.history_source == "train-prior":
    eval_pool_mask = train_fit_mask | heldout_mask
    eval_pool = WorkoutDataset(df[eval_pool_mask], data_config_train_prior_eval)
    eval_pool_positions = np.flatnonzero(eval_pool_mask.to_numpy())
    heldout_global = set(np.flatnonzero(heldout_mask.to_numpy()).tolist())
    heldout_local = [i for i, g in enumerate(eval_pool_positions) if g in heldout_global]
    test_dataset = Subset(eval_pool, heldout_local)
else:
    full_eval_dataset = WorkoutDataset(df, data_config_eval)
    heldout_indices = np.flatnonzero(heldout_mask.to_numpy()).tolist()
    test_dataset = Subset(full_eval_dataset, heldout_indices)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    collate_fn=workout_dataset_collate_fn,
    shuffle=False,
)

config = DBNConfig(
    data_config=data_config_train,
    seq_length=args.seq_length,
    learning_rate=1e-3,
    seed=0,
    n_epochs=args.epochs,
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
    adafs_variant=args.adafs_variant,
    paper_faithful=args.paper_faithful,
    physio_subject_stable_params=args.physio_subject_stable,
    intensity_use_embedding=args.intensity_embedding,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model = DBNModel(config=config, workouts_info=df[["userId", "id"]])

checkpoint_path = os.path.join(outdir, "best_model.pt")
if args.eval_only:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"--eval-only requested but checkpoint not found: {checkpoint_path}")
else:
    cwd = os.getcwd()
    os.chdir(outdir)  # Trainer writes best_model.pt relative to cwd
    try:
        trainer = Trainer(
            model,
            train_dataloader,
            val_dataloader,
            learning_rate=config.learning_rate,
            n_epochs=args.epochs,
            device=config.device,
            loss_type=args.loss,
            huber_delta=args.huber_delta,
            low_hr_weight=args.low_hr_weight,
            high_hr_weight=args.high_hr_weight,
            warmup_weight=args.warmup_weight,
            low_hr_threshold=args.low_hr_threshold,
            high_hr_threshold=args.high_hr_threshold,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            mean_bias_weight=args.mean_bias_weight,
        )
        trainer.train()
    finally:
        os.chdir(cwd)


def final_metrics(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    workout_count = 0
    truncated_workouts = 0
    max_predicted_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            # collate_fn returns workout_id/subject_id as numpy arrays, not tensors
            out = model.forecast_batch(
                activity=torch.as_tensor(batch["activity"]).float().to(device),
                times=torch.as_tensor(batch["time"]).float().to(device),
                workout_id=torch.as_tensor(batch["workout_id"]).to(device),
                subject_id=torch.as_tensor(batch["subject_id"]).to(device),
                history=torch.as_tensor(batch["history"]).float().to(device) if batch["history"] is not None else None,
                history_length=torch.as_tensor(batch["history_length"]).to(device) if batch["history_length"] is not None else None,
            )
            hr = torch.as_tensor(batch["heart_rate"]).float().to(device)
            n = min(out.size(1), hr.size(1))
            workout_count += hr.size(0)
            truncated_workouts += int((torch.as_tensor(batch["full_workout_length"]) > n).sum().item())
            max_predicted_steps = max(max_predicted_steps, n)
            preds.append(out[:, :n].cpu().numpy().ravel())
            trues.append(hr[:, :n].cpu().numpy().ravel())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    err = preds - trues
    return {
        "mae": np.abs(err).mean(),
        "rmse": np.sqrt((err ** 2).mean()),
        "n_steps": len(preds),
        "n_workouts": workout_count,
        "truncated_workouts": truncated_workouts,
        "max_predicted_steps": max_predicted_steps,
    }


def _window_starts(length, seq_length, stride):
    if length <= seq_length:
        return [0]
    starts = list(range(0, length - seq_length, stride))
    if not starts or starts[-1] + seq_length < length:
        starts.append(length - seq_length)
    return starts


def _forecast_stitched_workout(model, batch, i, length, device, seq_length, stride):
    activity = torch.as_tensor(batch["activity"][i, :length]).float().to(device)
    times = torch.as_tensor(batch["time"][i, :length]).float().to(device)
    starts = _window_starts(length, seq_length, stride)

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


def full_workout_metrics(model, dataloader, device, seq_length, stride):
    model.eval()
    workout_mae, workout_rmse = [], []
    all_abs_errors, all_sq_errors = [], []
    total_steps = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            heart_rate = torch.as_tensor(batch["heart_rate"]).float().to(device)
            lengths = torch.as_tensor(batch["full_workout_length"]).long().tolist()
            for i, length in enumerate(lengths):
                pred = _forecast_stitched_workout(model, batch, i, length, device, seq_length, stride)
                true = heart_rate[i, :length]
                err = pred - true
                abs_err = err.abs()
                sq_err = err.pow(2)
                workout_mae.append(abs_err.mean().item())
                workout_rmse.append(sq_err.mean().sqrt().item())
                all_abs_errors.append(abs_err.cpu().numpy())
                all_sq_errors.append(sq_err.cpu().numpy())
                total_steps += length

    all_abs_errors = np.concatenate(all_abs_errors)
    all_sq_errors = np.concatenate(all_sq_errors)
    return {
        "mean_workout_mae": float(np.mean(workout_mae)),
        "median_workout_mae": float(np.median(workout_mae)),
        "pooled_mae": float(all_abs_errors.mean()),
        "mean_workout_rmse": float(np.mean(workout_rmse)),
        "pooled_rmse": float(np.sqrt(all_sq_errors.mean())),
        "n_workouts": len(workout_mae),
        "n_steps": total_steps,
        "stride": stride,
    }


load_compatible_state_dict(model, torch.load(checkpoint_path, map_location=config.device))
# Final reporting: held-out only. Optionally also log val for transparency.
val_metrics = final_metrics(model, val_dataloader, config.device) if val_mask.any() else None
metrics = final_metrics(model, test_dataloader, config.device)
full_metrics = None
if args.full_workout:
    full_metrics = full_workout_metrics(model, test_dataloader, config.device, config.seq_length, args.eval_stride)

print("\n" + "=" * 58)
print(f"  HELD-OUT RESULT  [{args.name}]")
print(f"  MAE  {metrics['mae']:6.2f} BPM")
print(f"  RMSE {metrics['rmse']:6.2f} BPM")
print(
    f"  horizon: first {metrics['max_predicted_steps']} ten-second steps "
    f"({metrics['max_predicted_steps'] / 6:.1f} min), not full-workout"
)
print(f"  workouts: {metrics['n_workouts']:,} held-out; truncated: {metrics['truncated_workouts']:,}")
print(f"  points: {metrics['n_steps']:,}")
if val_metrics is not None:
    print("-" * 58)
    print(f"  VAL (checkpoint set) MAE {val_metrics['mae']:6.2f}  RMSE {val_metrics['rmse']:6.2f}  n={val_metrics['n_workouts']}")
if full_metrics is not None:
    print("-" * 58)
    print("  STITCHED FULL-WORKOUT RESULT")
    print(f"  mean workout MAE    {full_metrics['mean_workout_mae']:6.2f} BPM")
    print(f"  median workout MAE  {full_metrics['median_workout_mae']:6.2f} BPM")
    print(f"  pooled MAE          {full_metrics['pooled_mae']:6.2f} BPM")
    print(f"  pooled RMSE         {full_metrics['pooled_rmse']:6.2f} BPM")
    print(f"  workouts: {full_metrics['n_workouts']:,}; points: {full_metrics['n_steps']:,}; stride: {full_metrics['stride']}")
print("=" * 58)

with open(os.path.join(outdir, "result.txt"), "w") as f:
    f.write(
        f"{args.name}\n"
        f"adafs={args.adafs} adafs_variant={args.adafs_variant} paper_faithful={args.paper_faithful} "
        f"physiological={args.physiological} residual={args.residual} "
        f"contextual_residual={args.contextual_residual} "
        f"physio_subject_stable={args.physio_subject_stable} "
        f"intensity_embedding={args.intensity_embedding} "
        f"epochs={args.epochs} history_source={args.history_source} sport={args.sport or 'all'} "
        f"val_fraction={args.val_fraction}\n"
        f"train_fit_n={int(train_fit_mask.sum())} val_n={int(val_mask.sum())} heldout_n={int(heldout_mask.sum())}\n"
        f"seq_length={args.seq_length} train_stride={args.train_stride} feature_set={args.feature_set} "
        f"activity_columns={','.join(activity_columns)} loss={args.loss} "
        f"huber_delta={args.huber_delta} low_hr_weight={args.low_hr_weight} "
        f"high_hr_weight={args.high_hr_weight} warmup_weight={args.warmup_weight} "
        f"weight_decay={args.weight_decay} mean_bias_weight={args.mean_bias_weight}\n"
        f"MAE={metrics['mae']:.4f}\n"
        f"RMSE={metrics['rmse']:.4f}\n"
        f"horizon_steps={metrics['max_predicted_steps']}\n"
        f"n_workouts={metrics['n_workouts']}\n"
        f"truncated_workouts={metrics['truncated_workouts']}\n"
        f"n_steps={metrics['n_steps']}\n"
    )
    if val_metrics is not None:
        f.write(f"val_MAE={val_metrics['mae']:.4f}\nval_RMSE={val_metrics['rmse']:.4f}\n")
    if full_metrics is not None:
        f.write(
            f"full_mean_workout_MAE={full_metrics['mean_workout_mae']:.4f}\n"
            f"full_median_workout_MAE={full_metrics['median_workout_mae']:.4f}\n"
            f"full_pooled_MAE={full_metrics['pooled_mae']:.4f}\n"
            f"full_mean_workout_RMSE={full_metrics['mean_workout_rmse']:.4f}\n"
            f"full_pooled_RMSE={full_metrics['pooled_rmse']:.4f}\n"
            f"full_n_workouts={full_metrics['n_workouts']}\n"
            f"full_n_steps={full_metrics['n_steps']}\n"
            f"full_stride={full_metrics['stride']}\n"
        )
