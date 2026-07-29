# Running the hybrid ODE baseline on FitRec (clean protocol)

Yes — **this codebase is built for FitRec / Endomondo**, not only Apple data.

## Evidence in the upstream repo

1. **`examples/preprocess.py`** loads Endomondo JSON, keeps **`"run"`** workouts, builds the feather schema.
2. **`examples/train_ode_model.ipynb`** trains on `endomondo.feather` with columns  
   `userId`, `id`, `time_grid`, `start_dt`, `heart_rate`, `heart_rate_normalized`, `speed_h`, `speed_v`, `in_train`.
3. Apple study data is **not public**; the README points users to FitRec for a runnable path.

Paper Table 3’s **Hybrid ODE MAE 6.1 BPM** is from **Apple Heart and Movement Study**, not FitRec.  
On FitRec you must measure separately under a **clean** protocol.

## Caveats in the demo notebook

| Notebook behavior | Issue |
|-------------------|--------|
| `df_tmp = df[df["subject_idx"] < 15]` | Only ~15 users (toy) |
| `test_dataset = WorkoutDataset(df_tmp, …)` | **Includes train workouts** (same optimism as DBN paper path) |
| Scheduler on train L1 | Not held-out selection |

## Clean FitRec runner (this repo)

```bash
cd baselines/ml-heart-rate-models-main
# needs: torch, torchdiffeq, pandas, pyarrow, tqdm, numpy
pip install -r requirements.txt

python examples/run_ode_fitrec_clean.py \
  --name ode-run-clean-val \
  --data "../../XR Twin _Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/output/endomondo_filtered.feather" \
  --sport run \
  --val-fraction 0.15 \
  --epochs 50 \
  --history-source all-prior \
  --full-workout
```

Defaults already point at `Heart_Rate_Modeling/output/endomondo_filtered.feather` if that path exists.

Outputs: `examples/reeval_ode/<name>/best_model.pt` and `result.txt`.

### Protocol (aligned with DBN clean reeval)

- Sport: run (optional)
- `train_fit` / `val` from chronological split of `in_train`
- Held-out: `~in_train` only for final metrics
- Checkpoint on **val MAE**
- Full-workout MAE (ODE integrates full sequences when `chunk_size=None`)

## Completed clean FitRec result

Run directory: `examples/reeval_ode/ode-run-clean-val/`

| Metric | Value |
|--------|------:|
| Validation MAE | 7.75 BPM |
| Mean workout MAE | 8.79 BPM |
| Median workout MAE | 7.12 BPM |
| Pooled MAE | 8.61 BPM |
| Pooled RMSE | 12.37 BPM |
| Held-out run workouts | 6,396 |
| Full-workout time steps | 2,191,034 |

Clean same-dataset comparison:

| Model | Mean workout MAE | Pooled MAE | Pooled RMSE |
|-------|-----------------:|-----------:|------------:|
| DBN strict train-prior | 8.12 BPM | 8.04 BPM | 11.19 BPM |
| Hybrid ODE FitRec baseline | 8.79 BPM | 8.61 BPM | 12.37 BPM |

Under this clean FitRec held-out full-workout protocol, the DBN model
outperforms the rerun Hybrid ODE baseline. This comparison is separate from
the paper Table 3 values, where the ODE 6.1 BPM number comes from the cited
Apple/Nazaret study.

## Alternative: build feather with Apple preprocess

```bash
python examples/preprocess.py \
  --input_path /path/to/endomondoHR_proper.json \
  --output_path /path/to/out_dir
# writes endomondo.feather
```

Your existing filtered feather is already compatible and preferred for a fair DBN vs ODE comparison on the **same** split flag (`in_train`).
