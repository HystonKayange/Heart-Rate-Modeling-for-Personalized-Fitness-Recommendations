# XR Twin Heart Rate Prediction

This repository predicts heart rate from workout data for fitness and XR rehabilitation research.

## Reference paper

Kayange et al., *Electronics* 2024, 13, 3888  
https://doi.org/10.3390/electronics13193888

---

## 1. Start here

Main code:

[`XR Twin _Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/`](XR%20Twin%20_Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/)

Read the package guide:

[`Heart_Rate_Modeling/README.md`](XR%20Twin%20_Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/README.md)

Post-publication technical notes:

[`FINDINGS.md`](FINDINGS.md)

ODE baseline on FitRec:

[`baselines/ml-heart-rate-models-main/FITREC_CLEAN.md`](baselines/ml-heart-rate-models-main/FITREC_CLEAN.md)

Public figures:

[`Heart_Rate_Modeling/examples/figures/public/`](XR%20Twin%20_Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/examples/figures/public/)

---

## 2. Published MAE and package MAE

| Item | Published paper | Standard protocol in this repository |
|------|-----------------|--------------------------------------|
| MAE | 5.2 BPM | 7.37 BPM mean workout MAE (best open-loop so far) |
| Evaluation | Notebook procedure from the paper period | Held-out set only; validation checkpoints; full workout |
| Model path in the text | Hybrid model (AdaFS + Equation 9) | Original checkpoint: linear emission; other stacks optional |

The notebook recorded about 5.07 BPM MAE. The paper states 5.2 BPM as the average.

That figure is **not** the same as a held-out full-workout result under the standard protocol.

See the package README and `FINDINGS.md` for full detail.

---

## 3. Project context

- Funding: IITP / MSIT, Republic of Korea  
- Project No.: 2022-0-00218  
- Title: *XR Twin-based Rehabilitation Training Content Technology Development*

---

## 4. Citation

```bibtex
@article{kayange2024hybrid,
  title={A Hybrid Approach to Modeling Heart Rate Response for Personalized Fitness Recommendations Using Wearable Data},
  author={Kayange, Hyston and Mun, Jonghyeok and Park, Yohan and Choi, Jongsun and Choi, Jaeyoung},
  journal={Electronics},
  volume={13},
  number={19},
  pages={3888},
  year={2024},
  publisher={MDPI}
}
```
