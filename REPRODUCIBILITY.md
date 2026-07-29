# Reproducibility Statement

This repository accompanies:

Kayange, H.; Mun, J.; Park, Y.; Choi, J.; Choi, J.  
*A Hybrid Approach to Modeling Heart Rate Response for Personalized Fitness
Recommendations Using Wearable Data.* *Electronics* **2024**, *13*, 3888.  
https://doi.org/10.3390/electronics13193888

This document defines the evaluation protocols and result provenance used by
the public repository. It is intended for researchers who want to reproduce,
extend, or compare against the released work.

---

## Summary

The paper reports an average MAE of **5.2 BPM** for the Hybrid DBN model. The
released notebook records the underlying value as approximately **5.07 BPM**:

```text
MAE: 5.065673828125
```

The repository now also provides stricter held-out full-workout reevaluation
scripts. These scripts use validation checkpoints, held-out final scoring, and
explicit history modes. The resulting numbers are protocol-specific and should
be cited with the protocol that produced them.

| Model / protocol | Mean workout MAE | Pooled MAE | Pooled RMSE |
|------------------|-----------------:|-----------:|------------:|
| DBN strict train-prior | 8.12 BPM | 8.04 BPM | 11.19 BPM |
| DBN sequential-history | 7.37 BPM | 7.27 BPM | 10.54 BPM |
| Hybrid ODE rerun on FitRec | 8.79 BPM | 8.61 BPM | 12.37 BPM |

Supported clean-protocol claim:

> Under the clean FitRec held-out full-workout reevaluation, the DBN model
> outperforms the rerun Hybrid ODE baseline.

The published 5.2 BPM DBN value and the cited 6.1 BPM Hybrid ODE value should
not be described as a same-protocol clean FitRec comparison.

---

## Protocol Definitions

The clean reevaluation protocol uses the filtered FitRec / Endomondo file:

```text
Heart_Rate_Modeling/output/endomondo_filtered.feather
```

Evaluation rules:

1. Use the existing `in_train` split flag.
2. Split `in_train` workouts into train-fit and validation subsets by user chronology.
3. Use validation only for checkpoint selection and learning-rate scheduling.
4. Score final metrics only on held-out `~in_train` workouts.
5. Report full-workout stitched predictions as the primary result.
6. State the history mode used by the run.

History modes:

| Mode | Meaning |
|------|---------|
| `train-prior` | Validation and held-out workouts use only train-fit workouts as history. This is the strictest protocol for new claims. |
| `all-prior` | Later workouts may use earlier chronological workouts from the same user as history. This represents sequential personalization. |
| `split` | Histories are built only inside the evaluated split; kept for ablation/debugging. |

---

## Published Result Provenance

The published 5.2 BPM value is retained as a paper-period notebook result. It is
useful for understanding the original report, but it is not the same metric as
the clean held-out full-workout reevaluation reported above.

Key differences:

| Aspect | Paper-period notebook result | Clean reevaluation |
|--------|------------------------------|--------------------|
| Headline DBN MAE | 5.2 BPM | 8.12 BPM strict train-prior; 7.37 BPM sequential-history |
| Checkpoint rule | Notebook/trainer path from paper period | Validation-only checkpointing |
| Final scoring | Notebook evaluation path | Held-out `~in_train` workouts |
| Prediction length | First window in the notebook path | Full workout, stitched |
| Reporting requirement | Paper result provenance | Protocol-specific reproducibility metric |

---

## Hybrid ODE Baseline on FitRec

The paper compares against a 6.1 BPM Hybrid ODE result from the cited
Apple/Nazaret study. For same-dataset reproducibility, this repository also
reruns the Hybrid ODE code on the FitRec file used by the DBN reevaluation.

Tracked result summary:

```text
baselines/ml-heart-rate-models-main/examples/reeval_ode/ode-run-clean-val/result.txt
```

Clean FitRec ODE result:

```text
Validation MAE        7.75 BPM
Mean workout MAE      8.79 BPM
Median workout MAE    7.12 BPM
Pooled MAE            8.61 BPM
Pooled RMSE          12.37 BPM
Held-out workouts     6,396
Full-workout steps    2,191,034
```

Direct FitRec comparison:

| Model | Mean workout MAE | Pooled MAE | Pooled RMSE |
|-------|-----------------:|-----------:|------------:|
| DBN strict train-prior | 8.12 BPM | 8.04 BPM | 11.19 BPM |
| Hybrid ODE FitRec rerun | 8.79 BPM | 8.61 BPM | 12.37 BPM |

---

## Model Configuration Notes

The repository exposes the paper-described components as explicit configuration
paths:

- physiological head / Equation 9
- physiological residual path
- AdaFS variants
- train-prior and sequential-history evaluation
- validation-only checkpointing
- full-workout stitched metrics

The original checkpoint remains loadable, and the compatibility path is kept so
researchers can distinguish paper-period artifacts from post-publication
reevaluation variants.

---

## Current Limitations and Research Directions

The clean DBN model is strongest in the mid-heart-rate range and shows a
regression-to-mean pattern at the extremes:

| Cohort / segment | Diagnostic result |
|------------------|------------------:|
| Average HR <120 BPM workouts | 13.98 BPM mean MAE; +11.16 BPM bias |
| Average HR >=170 BPM workouts | 10.13 BPM mean MAE; -8.12 BPM bias |
| HR <120 BPM segments | 16.13 BPM pooled MAE |
| First 2 minutes | 11.87 BPM pooled MAE |

These diagnostics motivate future work on subject-relative intensity, session
context, and workout-level bias reduction.

---

## Reproduction Commands

DBN strict train-prior run:

```bash
cd "XR Twin _Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/examples"

python3 run_reeval.py \
  --name run-huber-128-delta12-intensity-trainprior-val \
  --physiological --residual \
  --sport run --history-source train-prior \
  --seq-length 128 --train-stride 64 \
  --feature-set run_intensity \
  --loss huber --huber-delta 12 \
  --weight-decay 1e-4 \
  --val-fraction 0.15 \
  --epochs 100 --full-workout
```

Public figures:

```bash
cd "XR Twin _Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/examples"
python3 plot_public_figures.py
```

Hybrid ODE evaluation from an existing checkpoint:

```bash
cd baselines/ml-heart-rate-models-main/examples

/home/cyai/.venvs/xr-hr-p2/bin/python -u run_ode_fitrec_clean.py \
  --name ode-run-clean-val \
  --eval-only
```

---

## Validation

Validation used for the July 2026 reevaluation branch:

```text
python -m py_compile ...  # passed
pytest tests/ -q          # 29 passed
```

Generated logs, raw data, and model checkpoints are not committed by default.
The repository tracks lightweight result summaries and figure artifacts needed
to review the protocol-specific claims.
