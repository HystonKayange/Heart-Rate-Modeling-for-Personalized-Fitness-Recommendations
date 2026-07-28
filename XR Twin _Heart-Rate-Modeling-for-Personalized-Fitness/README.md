# Heart-Rate Modeling for Personalized Fitness Recommendations

This folder holds the implementation and reevaluation code for:

Kayange et al., *Electronics* **2024**, 13, 3888.

## Documentation

| Document | Content |
|----------|---------|
| [`Heart_Rate_Modeling/README.md`](Heart_Rate_Modeling/README.md) | Install, train, evaluate, protocol, results |
| [`../FINDINGS.md`](../FINDINGS.md) | Post-publication technical notes (repository root) |
| [`../baselines/ml-heart-rate-models-main/FITREC_CLEAN.md`](../baselines/ml-heart-rate-models-main/FITREC_CLEAN.md) | Hybrid ODE baseline on FitRec |
| [`Heart_Rate_Modeling/examples/figures/public/`](Heart_Rate_Modeling/examples/figures/public/) | Public figures for reevaluation |

## Results (summary)

| Source | MAE |
|--------|-----|
| Published paper | 5.2 BPM (notebook procedure; see package README) |
| Best open-loop under standard protocol | 7.37 BPM mean workout MAE |
| Paper-faithful stack under standard protocol | 7.42 BPM mean workout MAE |

Do not compare these numbers without the protocol notes in `Heart_Rate_Modeling/README.md`.

## Figures

See [`Heart_Rate_Modeling/examples/figures/public/FIGURES.md`](Heart_Rate_Modeling/examples/figures/public/FIGURES.md).
