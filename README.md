# Heart Rate Modeling for Personalized Fitness and XR Rehabilitation

This repository provides PyTorch code, evaluation protocols, and
reproducibility notes for **modeling heart-rate response from wearable workout
data**. The work supports personalized fitness, exercise physiology research,
and XR rehabilitation systems that adapt coaching to individual users.

## Keywords

heart-rate modeling, heart-rate prediction, wearable data, personalized
fitness, XR rehabilitation, exercise physiology, FitRec, Endomondo, time-series
modeling, physiological modeling, dynamic Bayesian networks, PyTorch,
digital health

## Reference paper

Kayange et al., *Electronics* 2024, 13, 3888  
https://doi.org/10.3390/electronics13193888

---

## 1. Start here

Main code:

[`XR Twin _Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/`](XR%20Twin%20_Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/)

Read the package guide:

[`Heart_Rate_Modeling/README.md`](XR%20Twin%20_Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/README.md)

Reproducibility notes:

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

ODE baseline on FitRec:

[`baselines/ml-heart-rate-models-main/FITREC_CLEAN.md`](baselines/ml-heart-rate-models-main/FITREC_CLEAN.md)

Third-party notices:

[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

Citation metadata:

[`CITATION.cff`](CITATION.cff)

Public figures:

[`Heart_Rate_Modeling/examples/figures/public/`](XR%20Twin%20_Heart-Rate-Modeling-for-Personalized-Fitness/Heart_Rate_Modeling/examples/figures/public/)

---

## 2. Reproducibility Position

This repository preserves the paper-period artifacts and adds reproducible
evaluation scripts. Published and reevaluated results should be cited with the
protocol that produced them.

| Item | Published paper | Clean reevaluation in this repository |
|------|-----------------|----------------------------------------|
| DBN headline MAE | 5.2 BPM | 8.12 BPM strict train-prior; 7.37 BPM sequential-history |
| Evaluation | Notebook procedure from the paper period | Held-out reporting; validation checkpoints; full workout |
| ODE comparison | Hybrid ODE 6.1 BPM from the cited Apple/Nazaret study | Hybrid ODE on FitRec: 8.79 BPM mean workout MAE |
| Model path in the text | Hybrid model (AdaFS + Equation 9) | Released-checkpoint compatibility path; paper-style components available as explicit flags |

The notebook recorded about 5.07 BPM MAE, which the paper reports as 5.2 BPM.
That figure is a real notebook output, but it is **not** the same metric as the
held-out full-workout results produced by the reevaluation scripts.

Under the clean FitRec held-out full-workout evaluation setting, the DBN model
remains better than the rerun Hybrid ODE baseline:

| Model | Mean workout MAE | Pooled MAE | Pooled RMSE |
|-------|-----------------:|-----------:|------------:|
| DBN strict train-prior | 8.12 BPM | 8.04 BPM | 11.19 BPM |
| Hybrid ODE FitRec baseline | 8.79 BPM | 8.61 BPM | 12.37 BPM |

See the package README and `REPRODUCIBILITY.md` for full detail.

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
