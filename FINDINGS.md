# Post-Publication Reproducibility Notes

Paper: Kayange, H.; Mun, J.; Park, Y.; Choi, J.; Choi, J.  
*A Hybrid Approach to Modeling Heart Rate Response for Personalized Fitness
Recommendations Using Wearable Data.* *Electronics* **2024**, *13*, 3888.  
https://doi.org/10.3390/electronics13193888

This document is a researcher-facing audit trail for the released code and the
post-publication reevaluation work completed in July 2026. It explains how the
published notebook result relates to the stricter reproducibility protocol now
provided in the repository.

For runnable commands, metrics, and figures, start with:

- [`README.md`](README.md)
- [`XR Twin _Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/README.md`](XR%20Twin%20_Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/README.md)
- [`baselines/ml-heart-rate-models-main/FITREC_CLEAN.md`](baselines/ml-heart-rate-models-main/FITREC_CLEAN.md)

---

## Executive Summary

The published **5.2 BPM MAE** is a real output from
`Heart_Rate_Modeling/examples/model_eval.ipynb`:

```text
MAE: 5.065673828125
```

The post-publication audit found that this value came from the paper-period
notebook procedure, not from the stricter held-out full-workout reevaluation now
implemented in the package.

Current clean FitRec held-out full-workout results:

| Model / protocol | Mean workout MAE | Pooled MAE | Pooled RMSE |
|------------------|-----------------:|-----------:|------------:|
| DBN strict train-prior | 8.12 BPM | 8.04 BPM | 11.19 BPM |
| DBN sequential-history | 7.37 BPM | 7.27 BPM | 10.54 BPM |
| Hybrid ODE rerun on FitRec | 8.79 BPM | 8.61 BPM | 12.37 BPM |

Protocol-specific conclusion:

> Under the clean FitRec held-out full-workout reevaluation, the DBN model
> outperforms the rerun Hybrid ODE baseline.

This should not be restated as "5.2 beats 6.1 on the same clean protocol,"
because those two numbers came from different evaluation contexts.

---

## Resolution Status

| Audit item | Original issue | Current repository status |
|------------|----------------|---------------------------|
| Published 5.2 BPM provenance | The origin of the number was unclear. | Documented as notebook output from `model_eval.ipynb`; not treated as clean held-out full-workout MAE. |
| Held-out evaluation | The notebook result was not a strict held-out full-workout metric. | `run_reeval.py` now supports validation checkpoints, held-out final scoring, full-workout stitching, and explicit history modes. |
| Equation 9 physiological head | The paper described Eq. 9 as central, but the original checkpoint used a linear emission path. | Implemented behind explicit flags; default remains compatible with the original checkpoint. |
| AdaFS | The original trained path did not execute AdaFS. | AdaFS variants are now explicit optional configurations; not used for the strict train-prior headline. |
| History handling | Padding could affect the history encoder state. | `LSTMEncoder` now supports sequence lengths and masks padded history. |
| Checkpoint selection | The old trainer selected against the reported test path. | Trainer now supports validation-loader checkpointing and learning-rate scheduling. |
| ODE comparison | Paper Table 3 used the cited ODE result, not a clean FitRec rerun. | ODE was rerun on FitRec; result summary is tracked in `baselines/.../reeval_ode/ode-run-clean-val/result.txt`. |
| Research figures | Earlier public figures did not show the clean DBN-vs-ODE comparison. | Public figures now include protocol comparison, prediction examples, error scatter, and cohort diagnostics. |

---

## Published Number vs Clean Protocol

The paper reported:

| Model | MAE | RMSE |
|-------|----:|-----:|
| Hybrid ODE Model | 6.1 BPM | - |
| FitRec (U/S/C) | 7.0 BPM | 17.1 BPM |
| Hybrid DBN Model | 5.2 BPM | 8.1 BPM |

The code audit found that the DBN 5.2 BPM value is best understood as the
paper-period notebook result. It is useful for provenance, but it is not directly
comparable to the clean held-out full-workout results.

The clean repository protocol now reports:

1. Train/validation split from `in_train` workouts.
2. Validation set used only for checkpointing and learning-rate scheduling.
3. Final metrics only on `~in_train` held-out workouts.
4. Full-workout stitched predictions as the primary metric.
5. Explicit history mode:
   - `train-prior`: held-out workouts use only `train_fit` history.
   - `all-prior`: sequential personalization from all earlier workouts.

---

## Clean Same-Dataset ODE Comparison

The Hybrid ODE code was rerun on the same FitRec / Endomondo filtered file used
by the DBN reevaluation:

```text
data=Heart_Rate_Modeling/output/endomondo_filtered.feather
sport=run
heldout_n=6396
full_n_steps=2191034
```

Result:

```text
Validation MAE        7.75 BPM
Mean workout MAE      8.79 BPM
Median workout MAE    7.12 BPM
Pooled MAE            8.61 BPM
Pooled RMSE          12.37 BPM
```

This supports the clean same-dataset statement that DBN outperforms the rerun
Hybrid ODE baseline on FitRec. The result is separate from the 6.1 BPM ODE value
reported by the cited Apple/Nazaret study.

---

## Remaining Scientific Caveats

The clean DBN result is not the end of the modeling problem. Diagnostics show a
regression-to-mean pattern:

| Cohort / segment | Error pattern |
|------------------|---------------|
| Average HR <120 BPM workouts | 13.98 BPM mean MAE; +11.16 BPM bias |
| Average HR >=170 BPM workouts | 10.13 BPM mean MAE; -8.12 BPM bias |
| HR <120 BPM segments | 16.13 BPM pooled MAE |
| First 2 minutes | 11.87 BPM pooled MAE |

Interpretation: the model handles mid-range heart-rate dynamics better than
extreme easy or hard sessions. Future work should focus on subject-relative
intensity, session context, and reducing workout-level bias.

---

## Implementation Notes

Important post-publication code additions:

- `DBNConfig.use_adafs`
- `DBNConfig.use_physiological_head`
- `DBNConfig.use_physiological_residual`
- `DBNConfig.paper_faithful`
- `DBNConfig.adafs_variant`
- `WorkoutDatasetConfig.history_allowed_column`
- `LSTMEncoder.forward(..., lengths=...)`
- `run_reeval.py --history-source train-prior|all-prior|split`
- `run_reeval.py --val-fraction`
- `analyze_errors.py`
- `plot_public_figures.py`

The default model path remains compatible with the original checkpoint. Optional
paper-style components are activated explicitly through flags rather than being
silently assumed.

---

## Reproducibility Checks

Verification used for the July 2026 reevaluation branch:

```text
python -m py_compile ...  # passed
pytest tests/ -q          # 29 passed
```

The public figures can be regenerated from:

```bash
cd "XR Twin _Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/examples"
python3 plot_public_figures.py
```

The clean ODE result can be recomputed from an existing checkpoint with:

```bash
cd baselines/ml-heart-rate-models-main/examples
/home/cyai/.venvs/xr-hr-p2/bin/python -u run_ode_fitrec_clean.py \
  --name ode-run-clean-val \
  --eval-only
```

---

## Data Provenance

The working filtered file is:

```text
Heart_Rate_Modeling/output/endomondo_filtered.feather
```

FitRec / Endomondo data are externally distributed; large raw data files and
generated model checkpoints are not committed to this repository.

The post-publication reevaluation uses the same `in_train` split flag in the
filtered file and reports the exact split counts in each `result.txt`.
