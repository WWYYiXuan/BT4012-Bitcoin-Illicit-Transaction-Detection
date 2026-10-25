# Bitcoin Transaction Classification

This project classifies Bitcoin transactions by illicit probability using 165 anonymized numeric features and a directed transaction graph. The analysis and results are presented in [e1156133.ipynb](e1156133.ipynb).

## Project files

```text
Kaggle/
├── README.md
├── e1156133.ipynb
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── txs_edgelist.csv
│   └── sample_submission.csv
├── submission_16.csv
└── submission_19.csv
```

This file list covers the report, competition data and two selected submissions.

## Data

- **Training:** 141,772 transactions: 27,591 confirmed licit, 3,644 confirmed illicit and 110,537 unknown.
- **Test:** 15,329 transactions with withheld labels.
- **Features:** all 165 original numeric features are retained. Transaction identifiers and `time_step` are excluded from classifier inputs.
- **Labels:** `0` means licit, `1` means illicit, and missing labels mean unknown. Only the 31,235 confirmed transactions provide supervised targets.
- **Graph:** 203,769 nodes and 234,355 directed edges. Each edge links an earlier transaction to a transaction that spends one of its outputs.

Training covers time steps 1–35; test covers 36–49. Unknown transactions contribute label-free feature-distribution and graph information without receiving invented labels.

## Notebook structure

1. **Data and working environment:** source files and analysis libraries.
2. **Exploratory data analysis:** dimensions, missing values, duplicates, labels, time coverage and graph coverage.
3. **Data preparation:** numeric validation, confirmed-label cohorts and training-fitted imputation.
4. **Validation design:** matched stratified random folds and evaluation limitations.
5. **Logistic regression baseline:** a linear reference model.
6. **Model development:** HGB and CatBoost, probability blending, subtype weighting, graph descriptors and stronger rare-subtype weighting.
7. **Final model selection:** comparison of the two selected strategies, residual errors and prediction-file validation.

## Validation

Models are compared using three-fold stratified random validation repeated with seeds **47, 48 and 49**. Imputers, scalers and subtype clusters are fitted without held-out transactions. Mean and weakest-fold ROC-AUC summarize ranking performance; precision, recall and F1 use a probability threshold of **0.5**.

Random folds mix periods and may share graph neighborhoods. They do not establish performance on unseen future periods. Repeated shuffles overlap, and reuse of the same folds introduces selection optimism. Local validation scores and Kaggle public scores are therefore reported separately.

## Selected models

| Model | Prediction file |
|---|---|
| HGB–CatBoost Graph Ensemble | `submission_16.csv` |
| HGB–CatBoost Graph Ensemble with Stronger Subtype Weighting | `submission_19.csv` |

Both strategies use **177 classifier inputs**: all 165 original features plus 12 graph descriptors covering degree, source/sink position, graph depth and shared-neighbor structure.

Each prediction combines four learners:

| Component | Probability weight |
|---|---:|
| HGB with feature-subtype weights | 37.5% |
| CatBoost with feature-subtype weights | 37.5% |
| HGB with graph-informed subtype weights | 12.5% |
| CatBoost with graph-informed subtype weights | 12.5% |

HGB uses 300 iterations, learning rate 0.05, 15 leaves, minimum leaf size 100 and L2 regularization 5, with early stopping disabled. Its fitting weights also include a 20-step recency half-life. CatBoost uses 800 iterations, depth 6, learning rate 0.05 and L2 leaf regularization 3, without recency weighting.

Subtype weights increase the influence of smaller clusters of confirmed illicit transactions while preserving total illicit weight. HGB–CatBoost Graph Ensemble with Stronger Subtype Weighting strengthens the existing positive subtype weights by squaring, clipping to 0.5–3 and renormalizing. The learner settings, inputs and blend fractions remain the same.

| Metric | HGB–CatBoost Graph Ensemble | HGB–CatBoost Graph Ensemble with Stronger Subtype Weighting |
|---|---:|---:|
| Mean validation ROC-AUC, nine folds | 0.998787 | 0.998798 |
| Weakest-fold ROC-AUC | 0.998084 | 0.998111 |
| Mean precision | 0.990680 | 0.989753 |
| Mean recall | 0.952524 | 0.953439 |
| Mean F1 | 0.971222 | 0.971247 |
| Confirmed Kaggle public ROC-AUC | **0.95132** | Not reported |

**HGB–CatBoost Graph Ensemble** is the strongest model supported by a confirmed Kaggle result. **HGB–CatBoost Graph Ensemble with Stronger Subtype Weighting** is a targeted local candidate: it wins six of nine matched folds, although its average AUC gain is small and seed 49 worsens on average. A higher Kaggle score has not been established.

In the seed-47 out-of-fold error analysis, rare-subtype misses decrease from **115 to 107**, with rare-subtype recall increasing from **64.9% to 67.4%**. Overall misses decrease from 174 to 167, while false positives increase from 35 to 37. These single-pass diagnostics are separate from the nine-fold averages above.

## Execution and prediction files

The notebook uses NumPy, pandas, SciPy, scikit-learn, CatBoost and threadpoolctl in a Jupyter-compatible Python environment. Recorded results are displayed by default through `USE_RECORDED_RESULTS = True`; `REBUILD_SUBMISSIONS = False` preserves the existing prediction files. Changing these controls enables evaluation or full-data refitting and can require substantial runtime.

The final models were fitted on all confirmed training labels. [HGB–CatBoost Graph Ensemble predictions](submission_16.csv) and [HGB–CatBoost Graph Ensemble with Stronger Subtype Weighting predictions](submission_19.csv) each contain 15,329 rows of illicit probabilities, with columns and row order matching `dataset/sample_submission.csv`. The notebook validates both files and protects existing files from replacement by differing regenerated output.
