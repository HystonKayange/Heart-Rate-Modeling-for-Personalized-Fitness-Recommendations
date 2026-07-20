# Post-publication code review — Electronics 2024, 13, 3888

Reviewed July 2026 against the released implementation and `best_model.pt`.
Paper: Kayange, H.; Mun, J.; Park, Y.; Choi, J.; Choi, J. "A Hybrid Approach to
Modeling Heart Rate Response for Personalized Fitness Recommendations Using
Wearable Data." *Electronics* **2024**, *13*, 3888. https://doi.org/10.3390/electronics13193888

Status: findings 1-4 verified. Finding 4's magnitude is pending a clean rerun.

---

## 1. Equation 9 was never evaluated

§4.3 presents `HR(t) = HRmin + (HRmax - HRmin)(1 - exp(-A(z) - B(z)I(t)))` as the
physiological core of the model. The released code instantiates `A`, `B`, `alpha`,
`beta`, `hr_min`, `hr_max` (`Model/dbn.py:171-176`) but never calls them. The
prediction path is `transition_model` (1-layer LSTM) -> `emission_model`
(`nn.Linear(8, 1)`).

**Evidence.** All 18 weight tensors of those six networks sit at their
initialization distribution in `best_model.pt`:

| tensor | fan_in | observed std | init std | ratio |
|---|---|---|---|---|
| `A.layers.0.weight` | 12 | 0.16817 | 0.16667 | 1.009 |
| `B.layers.0.weight` | 12 | 0.17041 | 0.16667 | 1.022 |
| `hr_min.layers.0.weight` | 12 | 0.16564 | 0.16667 | 0.994 |
| `hr_max.layers.0.weight` | 12 | 0.16681 | 0.16667 | 1.001 |
| ... (18 total, all ratios 0.94-1.27) | | | | |
| `emission_model.fc.weight` (trained, for contrast) | 8 | 0.58090 | 0.20412 | **2.85** |
| `transition_model.fc.weight` (trained, for contrast) | 256 | 0.65976 | 0.03608 | **18.28** |

Parameters not reached by the forward pass receive no gradient, so Adam leaves
them untouched. They are present in the checkpoint but were never trained.

**Independent corroboration from the original environment.** The repository tracks
Python 3.8 bytecode compiled on the original lab machine (`d:\Final_code\`,
July-August 2024). Decompiling `Model/__pycache__/dbn.cpython-38.pyc` yields a
`forecast_batch` identical to the committed source: `transition_model` ->
`emission_model`, with no call to `adafs_soft` and no use of the personalized
scalars.

The `.pyc` headers also record each source file's size at compile time, and all
five match the committed sources exactly:

| file | committed | recorded in 2024 `.pyc` |
|---|---|---|
| `dbn.py` | 11,677 | 11,677 |
| `data.py` | 17,336 | 17,336 |
| `trainer.py` | 5,558 | 5,558 |
| `modules_lstm.py` | 1,981 | 1,981 |
| `modules_dense_nn.py` | 1,762 | 1,762 |

The published repository is therefore the code that ran. An earlier hypothesis
that a wrong snapshot had been uploaded is not supported.

## 2. Adaptive feature selection was never executed

`AdaFSSoft` is called only from `DBNModel.forward()` (`dbn.py:185`). Training and
evaluation both call `forecast_batch()` (`trainer.py:41,98`), which does not
invoke it.

**Evidence.** `adafs_soft.controller.mlp.mlps.0.1.num_batches_tracked = 0` and
`running_var = 1.0` exactly — the BatchNorm inside AdaFS saw zero forward passes
across the entire training run. The same holds for the standalone bidirectional
`lstm_encoder` (`dbn.py:164`): `num_batches_tracked = 0`, `fc.weight` std 0.0365
versus 0.0361 at initialization.

`forward()` itself would raise if called: `dbn.py:182` unpacks the single tensor
returned by `LSTMEncoder.forward` as two values, which only succeeds at batch
size 2.

## 3. Table 4 (ablation study) reports experiments that did not run

Table 4 attributes MAE degradation to removing adaptive feature selection
(5.2 -> 12.5 BPM) and personalized parameters (5.2 -> 11.7 BPM). Per findings 1
and 2, neither component was active in the 5.2 BPM baseline. These rows cannot
have been produced by ablating the reported model.

## 4. Reported metrics are largely in-sample

`examples/model_eval.ipynb` built the evaluation set from the full dataframe
rather than the complement of the training split:

```python
train_dataset = WorkoutDataset(df_tmp[df_tmp["in_train"]], data_config_train)
test_dataset  = WorkoutDataset(df_tmp, data_config_test)   # all data
```

The 80% of workouts used for training were therefore also evaluated. The reported
MAE 5.2 BPM / RMSE 8.1 BPM (§5.3.1, Table 3) are not held-out figures.
`trainer.py:73-76` additionally selects the saved checkpoint against this same
set.

**Magnitude unknown pending rerun.** Mitigating factor: the split is within-subject
chronological, so every evaluated user's embedding was trained on their earlier
workouts. This is not a cold-start scenario, which limits how far performance can
fall.

---

## Secondary discrepancies

| Paper | Code |
|---|---|
| §3.1: "we filtered the dataset to include only running workout sessions"; Table 2 captioned "on running sport" | `preprocess.py:112` sets `target_activities = ["bike", "run"]`. In `endomondoHR_proper.json` that is 71,915 bike and 70,591 run workouts — cycling is the larger share. Table 2's statistics therefore describe a mixed cycling/running set. |
| Table 2: "Average workout speed 3.7 km/h" | Impossible under the code's own 5-40 km/h filter (`preprocess.py:172`). 3.7 m/s = 13.3 km/h is consistent with a mixed-sport set, so the unit label appears to be wrong. |
| §5.1: "workouts from the same user were not split across subsets" | `preprocess.py:184-187` splits each user's workouts chronologically 80/20, so every user appears in both. (The code's approach is appropriate for personalization; the description is inverted.) |
| §3.2.1, §5.1: duration filter 10 min - 2 h 20 min | 15 min - 2 h (`preprocess.py:129`) |
| §5.1: 665 users | `subject_embeddings` is (558, 8) |
| §3.2.3, Table 1: elevation gain, average speed, speed variability, max HR, gender as model inputs | Computed in `preprocess.py:156-161`, then dropped. Model input is `["speed_h", "speed_v"]` only |
| Abstract, §3.2.3, §4.1, Eq. 1: environmental factors / temperature | `weather_columns=[]`. No environmental inputs. (Note: temperature and humidity *are* available in `endomondoMeta.json.gz` — this was not a data limitation) |
| §5.1: "minimizing the MSE" | Sum of squared errors, unnormalized (`trainer.py:12`) |
| Eq. 1, 2: transition noise sigma^2, probabilistic state | Deterministic. No variance predicted, no likelihood optimized, no latent-state inference |
| Data Availability: "Restrictions apply... request from the authors" | Public download at https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html |

Evaluation also covers only the first 64 steps (~10.7 min at the 10 s grid) of each
workout; `forecast_batch` truncates and `trainer.py:107-110` truncates ground truth
to match. Figures 6 and 8 show 35-47 minute sessions, which this path cannot
produce.

`plotting.py` and `evaluation.py` were never uploaded, but both were recovered by
decompiling their `.pyc`. `plotting.py` (compiled 2024-08-05) calls
`forecast_single_workout` and then reads `predictions["hr_min"]` and
`predictions["hr_max"]`, with a docstring referring to "the ODE parameters for the
workout". The committed `forecast_single_workout` returns only `heart_rate`, so
this module raises `KeyError` against the model as released. It appears to be
leftover scaffolding from the hybrid-ODE codebase of ref. [4] and cannot have
produced Figures 6-9.

## Environment note (not a defect)

`preprocess.py` cannot run under pandas 3.x. `pd.date_range` now returns
`datetime64[s]` where every earlier pandas returned `datetime64[ns]`, so the
`/1e9` nanosecond conversion at line 170 is off by a factor of 1e9, yielding
speeds around 4.7e9 m/s and emptying the frame at the 5-40 km/h filter. Under
pandas 2.3.3 the same unmodified code computes `dt = 10.0 s` correctly. This is
environment drift, not an error in the preprocessing code.

---

## Retracted during review

An earlier draft claimed `preprocess.py` could not read `endomondoHR_proper.json`
because the heart rate field is Z-scored. **This was wrong.** The Z-scored schema
described on the FitRec project page documents the processed `.npy` files; the
JSON contains raw BPM (verified: `min=100.0, max=177.0`, no `tar_heart_rate`
field). `preprocess.py` reads the file it names, correctly.

---

## Work in progress

Branch `reeval-clean-split`:

- `DBNConfig.use_adafs` / `use_physiological_head`, both defaulting to `False`, so
  the default configuration still reproduces the model that actually trained.
- Equation 9 implemented in `DBNModel.physiological_head()`.
- `AdaFSSoft` corrected to weight feature channels within each time step. The
  previous softmax spanned the whole flattened sequence, normalizing 1216 weights
  to sum to 1.
- Test set changed to `~in_train`.
- `tests/test_forward_paths.py` asserts each component receives gradient in each
  configuration — the check that would have caught findings 1 and 2.

Two numbers are needed, both on a clean held-out split: the as-published
architecture, and the as-described architecture.

**Data provenance.** `endomondoHR_proper.json` was obtained from a Kaggle mirror
after the UCSD Google Drive link returned 404 and `deepyeti.ucsd.edu` stopped
responding. The mirror is byte-identical to the UCSD original: 4,929,126,138
bytes, matching the `Content-Length` served by
`mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/`. Schema verified as raw BPM
heart rate with no `tar_heart_rate` field.

Contents measured directly: 167,783 workouts / 1,059 users, against the 167,373 /
956 stated on the FitRec project page. Since the file is byte-identical to the
official artifact, the page's figures appear to be post-filtering counts rather
than a difference in the data.

`main` is unchanged and remains an accurate record of what produced the published
results. `best_model.pt` should be retained — it is the evidence for findings 1
and 2.
