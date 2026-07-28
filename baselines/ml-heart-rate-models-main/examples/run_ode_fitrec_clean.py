#!/usr/bin/env python3
"""
Clean FitRec reevaluation of the Apple / Nazaret hybrid ODE baseline.

The original notebook uses FitRec/Endomondo (endomondo.feather) but:
  - trains on a tiny user subset (subject_idx < 15) in the demo, and
  - builds test_dataset from the *full* frame (includes train workouts).

This script evaluates on FitRec under the same clean protocol as the DBN reeval:
  - optional sport filter (default: run)
  - train_fit / val split chronological per user from in_train
  - held-out = ~in_train only for final report
  - checkpoint on validation MAE (not held-out, not train)
  - primary metric: full-workout mean/median/pooled MAE
    (ODE predicts full sequences when chunk_size=None — no stitching needed)

Data:
  Prefer the already-preprocessed FitRec feather used by Heart_Rate_Modeling:
    .../Heart_Rate_Modeling/output/endomondo_filtered.feather

  Schema matches the ODE WorkoutDataset (userId, id, time_grid, start_dt,
  heart_rate, heart_rate_normalized, speed_h, speed_v, in_train, subject_idx).

Example:
  cd baselines/ml-heart-rate-models-main
  python examples/run_ode_fitrec_clean.py \\
    --name ode-run-clean-val \\
    --sport run --epochs 50 --full-workout
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse chronological val split helper from the DBN package when available.
HR_MODELING = ROOT.parents[1] / "XR Twin _Heart-Rate-Modeling-for-Personalized-Fitness" / "Heart_Rate_Modeling"
sys.path.insert(0, str(HR_MODELING))

from ode.data import WorkoutDataset, WorkoutDatasetConfig, workout_dataset_collate_fn
from ode.ode import ODEModel, OdeConfig

try:
    from Model.activity_features import chronological_train_val_masks
except ImportError:
    chronological_train_val_masks = None


DEFAULT_DATA = (
    HR_MODELING / "output" / "endomondo_filtered.feather"
)


def fallback_chronological_train_val_masks(df, base_mask, val_fraction=0.15):
    """Minimal copy if Model.activity_features is unavailable."""
    train_fit = pd.Series(False, index=df.index)
    val = pd.Series(False, index=df.index)
    if val_fraction <= 0 or not base_mask.any():
        train_fit.loc[base_mask] = True
        return train_fit, val
    subset = df.loc[base_mask]
    for _, group in subset.groupby("userId", sort=False):
        group = group.sort_values("start_dt")
        n = len(group)
        if n < 2:
            train_fit.loc[group.index] = True
            continue
        n_val = min(max(1, int(round(n * val_fraction))), n - 1)
        train_fit.loc[group.index[:-n_val]] = True
        val.loc[group.index[-n_val:]] = True
    return train_fit, val


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="ode-run-clean-val", help="output under examples/reeval_ode/<name>/")
    p.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="FitRec feather (default: Heart_Rate_Modeling filtered file)",
    )
    p.add_argument("--sport", default="run", help="sport filter; empty string for all")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-length", type=int, default=64, help="training chunk length")
    p.add_argument("--train-stride", type=int, default=32)
    p.add_argument("--history-max-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ode-step-size", type=float, default=1.0)
    p.add_argument("--full-workout", action="store_true", default=True)
    p.add_argument("--no-full-workout", action="store_false", dest="full_workout")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--history-source",
        choices=["split", "all-prior"],
        default="all-prior",
        help="held-out history: only held-out rows, or all chronological prior workouts",
    )
    return p.parse_args()


def make_dataloader(dataset, batch_size, shuffle, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=workout_dataset_collate_fn,
    )


def forecast_hr(model, batch, device, step_size):
    out = model.forecast_batch(
        activity=torch.as_tensor(batch["activity"]).float().to(device),
        times=torch.as_tensor(batch["time"]).float().to(device),
        workout_id=batch["workout_id"],
        subject_id=batch["subject_id"],
        history=torch.as_tensor(batch["history"]).float().to(device)
        if batch.get("history") is not None
        else None,
        history_length=torch.as_tensor(batch["history_length"]).to(device)
        if batch.get("history_length") is not None
        else None,
        weather=torch.as_tensor(batch["weather"]).float().to(device)
        if batch.get("weather") is not None
        else None,
        step_size=step_size,
    )
    return out["heart_rate"]


def pooled_mae(model, dataloader, device, step_size):
    model.eval()
    abs_errs = []
    with torch.no_grad():
        for batch in dataloader:
            pred = forecast_hr(model, batch, device, step_size)
            hr = torch.as_tensor(batch["heart_rate"]).float().to(device)
            lengths = torch.as_tensor(batch["full_workout_length"]).long()
            for i, length in enumerate(lengths.tolist()):
                n = min(int(length), pred.size(1), hr.size(1))
                abs_errs.append((pred[i, :n] - hr[i, :n]).abs().cpu().numpy())
    all_err = np.concatenate(abs_errs)
    return float(all_err.mean())


def full_workout_metrics(model, dataloader, device, step_size):
    model.eval()
    workout_mae, workout_rmse = [], []
    all_abs, all_sq = [], []
    n_steps = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="full-workout eval"):
            pred = forecast_hr(model, batch, device, step_size)
            hr = torch.as_tensor(batch["heart_rate"]).float().to(device)
            lengths = torch.as_tensor(batch["full_workout_length"]).long().tolist()
            for i, length in enumerate(lengths):
                n = min(int(length), pred.size(1), hr.size(1))
                err = pred[i, :n] - hr[i, :n]
                abs_err = err.abs()
                sq_err = err.pow(2)
                workout_mae.append(abs_err.mean().item())
                workout_rmse.append(sq_err.mean().sqrt().item())
                all_abs.append(abs_err.cpu().numpy())
                all_sq.append(sq_err.cpu().numpy())
                n_steps += n
    all_abs = np.concatenate(all_abs)
    all_sq = np.concatenate(all_sq)
    return {
        "mean_workout_mae": float(np.mean(workout_mae)),
        "median_workout_mae": float(np.median(workout_mae)),
        "pooled_mae": float(all_abs.mean()),
        "mean_workout_rmse": float(np.mean(workout_rmse)),
        "pooled_rmse": float(np.sqrt(all_sq.mean())),
        "n_workouts": len(workout_mae),
        "n_steps": n_steps,
    }


def train_loop(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs,
    lr,
    step_size,
    ckpt_path,
):
    from ode.trainer import l2_error, l2_reg, STD_HR, STD_EMBEDDING

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=4)
    best_val = float("inf")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm.tqdm(train_loader, desc=f"epoch {epoch} train"):
            # Move tensors used by model; forecast_batch expects CPU workout ids as list/array ok
            activity = torch.as_tensor(batch["activity"]).float().to(device)
            times = torch.as_tensor(batch["time"]).float().to(device)
            history = (
                torch.as_tensor(batch["history"]).float().to(device)
                if batch.get("history") is not None
                else None
            )
            history_length = (
                torch.as_tensor(batch["history_length"]).to(device)
                if batch.get("history_length") is not None
                else None
            )
            weather = (
                torch.as_tensor(batch["weather"]).float().to(device)
                if batch.get("weather") is not None
                else None
            )
            heart_rate = torch.as_tensor(batch["heart_rate"]).float().to(device)

            predictions = model.forecast_batch(
                activity=activity,
                times=times,
                workout_id=batch["workout_id"],
                subject_id=batch["subject_id"],
                history=history,
                history_length=history_length,
                weather=weather,
                step_size=step_size,
            )
            pred_hr = predictions["heart_rate"]
            n = min(pred_hr.size(1), heart_rate.size(1))
            recon = l2_error(pred_hr[:, :n], heart_rate[:, :n], std=STD_HR)
            emb = (predictions["workout_embedding"] / STD_EMBEDDING).pow(2).sum()
            emb = emb * model.config.embedding_reg_strength
            dec = l2_reg(model.fatigue_fn.parameters())
            if model.weather_fn is not None:
                dec = dec + l2_reg(model.weather_fn.parameters())
            dec = dec + l2_reg(model.activity_fn.parameters())
            dec = dec * model.config.decoder_reg_strength
            if model.embedding_store.encoder is not None:
                enc = l2_reg(model.embedding_store.encoder.parameters()) * model.config.encoder_reg_strength
            else:
                enc = 0.0
            loss = recon + emb + dec + enc
            optimizer.zero_grad()
            loss.backward()
            if model.config.clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.clip_gradient)
            optimizer.step()
            epoch_loss += float(loss.item())

        val_mae = pooled_mae(model, val_loader, device, step_size)
        scheduler.step(val_mae)
        print(
            f"Epoch {epoch}: train_loss={epoch_loss / max(len(train_loader), 1):.1f}  "
            f"val_MAE={val_mae:.3f}"
        )
        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved best checkpoint (val_MAE={best_val:.3f})")

    return best_val


def main():
    args = parse_args()
    if args.sport == "":
        args.sport = None
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(__file__).resolve().parent / "reeval_ode" / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "best_model.pt"

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"FitRec feather not found: {data_path}\n"
            "Build it with Heart_Rate_Modeling/examples/preprocess.py or the ODE preprocess.py."
        )

    print(f"Loading {data_path}")
    df = pd.read_feather(data_path)
    print(f"{len(df):,} workouts / {df['userId'].nunique():,} users")

    in_train_mask = df["in_train"].copy()
    heldout_mask = ~df["in_train"]
    if args.sport is not None:
        sport_mask = df["sport"].eq(args.sport)
        in_train_mask = in_train_mask & sport_mask
        heldout_mask = heldout_mask & sport_mask

    split_fn = chronological_train_val_masks or fallback_chronological_train_val_masks
    if chronological_train_val_masks is not None:
        train_fit_mask, val_mask = chronological_train_val_masks(
            df, in_train_mask, subject_id_column="userId", time_column="start_dt",
            val_fraction=args.val_fraction,
        )
    else:
        train_fit_mask, val_mask = split_fn(df, in_train_mask, args.val_fraction)

    print(
        f"train_fit {train_fit_mask.sum():,} | val {val_mask.sum():,} | held-out {heldout_mask.sum():,}"
    )

    data_config_train = WorkoutDatasetConfig(
        subject_id_column="userId",
        workout_id_column="id",
        time_since_start_column="time_grid",
        time_of_start_column="start_dt",
        heart_rate_column="heart_rate",
        heart_rate_normalized_column="heart_rate_normalized",
        activity_columns=["speed_h", "speed_v"],
        weather_columns=[],
        history_max_length=args.history_max_length,
        chunk_size=args.seq_length,
        stride=args.train_stride,
    )
    data_config_eval = dataclasses.replace(data_config_train, chunk_size=None, stride=None)

    train_dataset = WorkoutDataset(df[train_fit_mask], data_config_train)
    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True, drop_last=True)

    # Val: full workouts; history from in_train pool
    if val_mask.any():
        val_pool = WorkoutDataset(df[in_train_mask], data_config_eval)
        in_train_pos = np.flatnonzero(in_train_mask.to_numpy())
        val_global = set(np.flatnonzero(val_mask.to_numpy()).tolist())
        val_local = [i for i, g in enumerate(in_train_pos) if g in val_global]
        val_dataset = Subset(val_pool, val_local)
    else:
        val_dataset = WorkoutDataset(df[train_fit_mask], data_config_eval)
    val_loader = make_dataloader(val_dataset, args.batch_size, shuffle=False)

    if args.history_source == "split":
        test_dataset = WorkoutDataset(df[heldout_mask], data_config_eval)
    else:
        full_ds = WorkoutDataset(df, data_config_eval)
        heldout_idx = np.flatnonzero(heldout_mask.to_numpy()).tolist()
        test_dataset = Subset(full_ds, heldout_idx)
    test_loader = make_dataloader(test_dataset, args.batch_size, shuffle=False)

    ode_config = OdeConfig(
        data_config=data_config_train,
        learning_rate=args.lr,
        seed=0,
        n_epochs=args.epochs,
        ode_step_size=args.ode_step_size,
        encoder_embedding_dim=8,
        subject_embedding_dim=8,
    )
    # Embeddings over all subjects/workouts that may appear (train+val+held-out users)
    model = ODEModel(workouts_info=df[["userId", "id"]], config=ode_config)

    if args.eval_only:
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
    else:
        print(f"Training ODE on device={device} epochs={args.epochs}")
        best_val = train_loop(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            args.ode_step_size,
            ckpt_path,
        )
        print(f"Best val MAE during training: {best_val:.4f}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    val_mae = pooled_mae(model, val_loader, device, args.ode_step_size) if val_mask.any() else None
    full = full_workout_metrics(model, test_loader, device, args.ode_step_size) if args.full_workout else None
    # Also a simple pooled over eval loader (same as full for ODE full-seq)
    held_mae = full["pooled_mae"] if full else pooled_mae(model, test_loader, device, args.ode_step_size)

    print("\n" + "=" * 58)
    print(f"  ODE FITREC CLEAN HELD-OUT  [{args.name}]")
    if val_mae is not None:
        print(f"  VAL MAE (checkpoint set)  {val_mae:6.2f} BPM")
    if full is not None:
        print(f"  mean workout MAE          {full['mean_workout_mae']:6.2f} BPM")
        print(f"  median workout MAE        {full['median_workout_mae']:6.2f} BPM")
        print(f"  pooled MAE                {full['pooled_mae']:6.2f} BPM")
        print(f"  pooled RMSE               {full['pooled_rmse']:6.2f} BPM")
        print(f"  workouts: {full['n_workouts']:,}  steps: {full['n_steps']:,}")
    else:
        print(f"  held-out pooled MAE       {held_mae:6.2f} BPM")
    print("=" * 58)

    with open(outdir / "result.txt", "w") as f:
        f.write(
            f"{args.name}\n"
            f"model=hybrid_ode baseline=nazaret2023\n"
            f"data={data_path}\n"
            f"sport={args.sport or 'all'} history_source={args.history_source} "
            f"val_fraction={args.val_fraction} epochs={args.epochs}\n"
            f"train_fit_n={int(train_fit_mask.sum())} val_n={int(val_mask.sum())} "
            f"heldout_n={int(heldout_mask.sum())}\n"
            f"seq_length={args.seq_length} train_stride={args.train_stride} "
            f"ode_step_size={args.ode_step_size}\n"
        )
        if val_mae is not None:
            f.write(f"val_MAE={val_mae:.4f}\n")
        if full is not None:
            f.write(
                f"full_mean_workout_MAE={full['mean_workout_mae']:.4f}\n"
                f"full_median_workout_MAE={full['median_workout_mae']:.4f}\n"
                f"full_pooled_MAE={full['pooled_mae']:.4f}\n"
                f"full_mean_workout_RMSE={full['mean_workout_rmse']:.4f}\n"
                f"full_pooled_RMSE={full['pooled_rmse']:.4f}\n"
                f"full_n_workouts={full['n_workouts']}\n"
                f"full_n_steps={full['n_steps']}\n"
            )
        else:
            f.write(f"MAE={held_mae:.4f}\n")

    print(f"Wrote {outdir / 'result.txt'}")


if __name__ == "__main__":
    main()
