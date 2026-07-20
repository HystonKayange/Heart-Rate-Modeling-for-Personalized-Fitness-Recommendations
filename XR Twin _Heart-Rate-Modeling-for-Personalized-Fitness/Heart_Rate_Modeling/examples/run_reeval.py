"""
Clean-split re-evaluation.

Trains on df[in_train] and evaluates on df[~in_train] -- the held-out split the
paper describes but the notebook did not use. Reports MAE and RMSE in BPM.

    python run_reeval.py --name as-published
    python run_reeval.py --name as-described --adafs --physiological

Checkpoints are written under examples/reeval/<name>/ so the published
best_model.pt is never overwritten (it is the evidence for findings 1-2).
"""
import argparse
import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from Model.data import WorkoutDataset, WorkoutDatasetConfig, make_dataloaders
from Model.dbn import DBNConfig, DBNModel
from Model.trainer import Trainer

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="run name; output goes to reeval/<name>/")
p.add_argument("--adafs", action="store_true", help="enable adaptive feature selection")
p.add_argument("--physiological", action="store_true", help="enable the Eq. 9 head")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--data", default="../output/endomondo_filtered.feather")
args = p.parse_args()

outdir = os.path.join("reeval", args.name)
os.makedirs(outdir, exist_ok=True)

df = pd.read_feather(args.data)
print(f"{len(df):,} workouts / {df['userId'].nunique():,} users")
print(f"train {df['in_train'].sum():,}  |  held-out {(~df['in_train']).sum():,}")
print(f"config: adafs={args.adafs}  physiological={args.physiological}  epochs={args.epochs}\n")

data_config_train = WorkoutDatasetConfig(
    subject_id_column="userId",
    workout_id_column="id",
    time_since_start_column="time_grid",
    time_of_start_column="start_dt",
    heart_rate_column="heart_rate",
    heart_rate_normalized_column="heart_rate_normalized",
    activity_columns=["speed_h", "speed_v"],
    weather_columns=[],
    history_max_length=512,
)
data_config_test = dataclasses.replace(data_config_train, chunk_size=None, stride=None)

# The fix: evaluate on the complement of the training split, not the full frame.
train_dataset = WorkoutDataset(df[df["in_train"]], data_config_train)
test_dataset = WorkoutDataset(df[~df["in_train"]], data_config_test)
train_dataloader, test_dataloader = make_dataloaders(train_dataset, test_dataset, batch_size=args.batch_size)

config = DBNConfig(
    data_config=data_config_train,
    seq_length=64,
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
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model = DBNModel(config=config, workouts_info=df[["userId", "id"]])

cwd = os.getcwd()
os.chdir(outdir)  # Trainer writes best_model.pt relative to cwd
try:
    trainer = Trainer(model, train_dataloader, test_dataloader,
                      learning_rate=config.learning_rate, n_epochs=args.epochs, device=config.device)
    trainer.train()
finally:
    os.chdir(cwd)


def final_metrics(model, dataloader, device):
    model.eval()
    preds, trues = [], []
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
            preds.append(out[:, :n].cpu().numpy().ravel())
            trues.append(hr[:, :n].cpu().numpy().ravel())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    err = preds - trues
    return np.abs(err).mean(), np.sqrt((err ** 2).mean()), len(preds)


model.load_state_dict(torch.load(os.path.join(outdir, "best_model.pt"), map_location=config.device))
mae, rmse, n = final_metrics(model, test_dataloader, config.device)

print("\n" + "=" * 58)
print(f"  HELD-OUT RESULT  [{args.name}]")
print(f"  MAE  {mae:6.2f} BPM      (paper reported 5.2, in-sample)")
print(f"  RMSE {rmse:6.2f} BPM      (paper reported 8.1, in-sample)")
print(f"  over {n:,} predicted time steps")
print("=" * 58)

with open(os.path.join(outdir, "result.txt"), "w") as f:
    f.write(f"{args.name}\nadafs={args.adafs} physiological={args.physiological} epochs={args.epochs}\n"
            f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nn={n}\n")
