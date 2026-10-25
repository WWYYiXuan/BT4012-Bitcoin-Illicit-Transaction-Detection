# Bitcoin Transaction Classification

This BT4012 project ranks Bitcoin transactions by illicit probability using all 165 anonymized numeric features and the directed transaction graph. The report, model implementations, validation results and prediction-generation code are in [e1156133.ipynb](e1156133.ipynb).

The two selected strategies are the **HGB?CatBoost Graph Ensemble** and the **HGB?CatBoost Graph Ensemble with Stronger Subtype Weighting**. Their prediction files are `submission_16.csv` and `submission_19.csv`, respectively.

## 1. Repository and data layout

The competition data are not included in the repository. Obtain the four original files from the competition and place them in a folder named **`dataset/` directly beside the notebook**:

```text
repository/
|-- README.md
|-- requirements.txt
|-- e1156133.ipynb
|-- submission_16.csv
|-- submission_19.csv
`-- dataset/                       # Created locally; data supplied separately
    |-- train.csv
    |-- test.csv
    |-- txs_edgelist.csv
    `-- sample_submission.csv
```

Use the original filenames and column names. `sample_submission.csv` is required because it defines the output schema and identifier order. The notebook reads the data through `PROJECT_DIR / 'dataset'`; there are no hardcoded user-specific data paths.

The notebook must run with the **repository root as its working directory**. Its configuration uses `Path.cwd()` to identify that directory. Opening the repository folder in VS Code and running its notebook provides the intended layout. A missing-data error reports the folder searched and the filenames that are absent.

The uploaded prediction files are in the repository root. The notebook's default export directory is `results/`; the reproduction configuration below explicitly changes it to the root to match the uploaded layout.

## 2. Python environment

The reference environment is **64-bit Python 3.12 on Windows**. The models use CPU; a GPU is not required. Dependencies are listed in [requirements.txt](requirements.txt).

From the repository root, create an isolated environment and install the requirements.

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux/macOS with Python 3.12 installed:**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

In VS Code, use the Python and Jupyter extensions, open `e1156133.ipynb`, and select this `.venv` environment as the notebook kernel. The requirements include `ipykernel`; they do not install the VS Code extensions or a standalone Jupyter Notebook server.

### Reference package versions

The following versions were observed in the model-development environment:

| Package | Reference version |
|---|---:|
| Python | 3.12.14 |
| NumPy | 2.5.4 |
| pandas | 3.0.6 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| CatBoost | 1.2.10 |
| threadpoolctl | 3.7.0 |

The current `requirements.txt` is mostly unpinned, so a normal installation may select different versions. For closer numerical reproduction, install the reference model-package versions into the same environment after installing the requirements:

```bash
python -m pip install numpy==2.5.4 pandas==3.0.6 scipy==1.18.1 scikit-learn==1.9.1 catboost==1.2.10 threadpoolctl==3.7.0
```

Run this command with the environment's Python executable, or from a terminal in which that environment is activated. Software versions, CPU libraries and floating-point behavior can affect exact numerical or byte-for-byte reproduction across machines.

## 3. Regenerate both selected submission files

The notebook is the reproduction entry point. Existing submission CSVs and previously fitted model objects are not needed to generate predictions; the models are refitted from the four competition data files.

### Configuration

1. Open the notebook with the environment described above.
2. In the setup cell under **Section 1: Data and working environment**, replace the existing configuration assignments with:

```python
PROJECT_DIR = Path.cwd().resolve()
DATA_DIR = PROJECT_DIR / 'dataset'
RESULTS_DIR = PROJECT_DIR

USE_RECORDED_RESULTS = True
REBUILD_SUBMISSIONS = True
OVERWRITE_SUBMISSIONS = True
SHOW_EXISTING_SUBMISSIONS = True
```

The setup cell already imports `Path`. Replace its corresponding assignments rather than adding conflicting values that are overwritten later in that cell.

3. Restart the kernel and select **Run All**. Restarting clears any variables retained from earlier interactive runs.
4. The final export cell in **Section 7.2: Final prediction files** fits and exports both selected strategies.

This configuration writes to the repository root:

| Model | Generated file |
|---|---|
| HGB&CatBoost Graph Ensemble | `submission_16.csv` |
| HGB&CatBoost Graph Ensemble with Stronger Subtype Weighting | `submission_19.csv` |

**`OVERWRITE_SUBMISSIONS = True` replaces existing files with these names.** To preserve the uploaded CSVs and generate a comparison copy, use:

```python
RESULTS_DIR = PROJECT_DIR / 'regenerated'
OVERWRITE_SUBMISSIONS = False
```

The export routine creates the output directory when needed. With overwriting disabled, an existing destination raises `FileExistsError` before fitting that model; use an empty output directory for a new comparison run.

### What runs during regeneration

For each selected model, the notebook:

1. Validates the input schema and separates confirmed labels from unknown transactions.
2. Constructs 12 label-free structural descriptors from the directed graph.
3. Uses all 31,235 confirmed transactions as supervised training data.
4. Fits the illicit-profile clustering and derives the selected subtype weights using the full training reference population.
5. Fits four HGB/CatBoost learners with fixed settings and seed **47**.
6. Predicts illicit probabilities for the test transactions and combines them using the fixed ensemble weights.
7. Maps predictions to the identifiers and order in `sample_submission.csv` and writes the CSV.

The two recipes fit eight learners in total. CPU execution can take several minutes or longer depending on hardware. Unknown transactions contribute label-free covariates and topology; they are never assigned licit targets.

### Recorded results versus new predictions

`USE_RECORDED_RESULTS = True` displays the completed validation experiments without retraining every historical comparison. It **does not** load saved test predictions into the inference routine. `REBUILD_SUBMISSIONS = True` independently triggers fresh full-data fitting and prediction generation.

For report inspection only, leave `REBUILD_SUBMISSIONS = False`. Existing files in `RESULTS_DIR` are validated; absent files are reported as `not generated`. This mode does not recreate CSVs.

To repeat the nine-fold validation comparisons as well, set `USE_RECORDED_RESULTS = False` and run all cells. This is substantially more expensive and is unnecessary for regenerating the two final prediction files.

## 4. Data preprocessing and feature engineering

Training contains 141,772 transactions: 27,591 confirmed licit, 3,644 confirmed illicit and 110,537 unknown. Test contains 15,329 transactions with withheld labels. Labels are encoded as `0` for licit, `1` for illicit and missing for unknown.

The preprocessing checks numeric types, transaction identifiers and time steps, removes identical duplicate rows, and replaces nonfinite feature values with missing values. Median imputers are fitted on fitting rows only. Unknown labels are excluded from supervised targets.

Both final strategies retain **all 165 original numeric features** and append **12 graph descriptors**, giving **177 classifier inputs**. Transaction identifiers and `time_step` are excluded from classifier inputs. The graph descriptors summarize degree, source/sink position, depth and shared-neighbor structure. Graph degree counts observed neighboring transactions rather than exact Bitcoin input/output counts.

Unknown source transactions participate in the label-free, within-period percentile reference used to construct subtype weights. Complete supplied snapshot topology is available at prediction time. No test labels or invented labels are used.

## 5. Final model recipes

Each selected strategy combines four learners:

| Component | Probability weight |
|---|---:|
| HGB with feature-subtype weights | 37.5% |
| CatBoost with feature-subtype weights | 37.5% |
| HGB with graph-informed subtype weights | 12.5% |
| CatBoost with graph-informed subtype weights | 12.5% |

Equivalently, a 50:50 HGB/CatBoost pair with feature-based subtype weights contributes 75%, and a second 50:50 pair with graph-informed subtype weights contributes 25%.

| Setting | HGB | CatBoost |
|---|---|---|
| Iterations | 300 | 800 |
| Learning rate | 0.05 | 0.05 |
| Tree complexity | 15 maximum leaves | Depth 6 |
| Minimum leaf size | 100 | Default |
| L2 regularization | 5 | 3 |
| Early stopping | Disabled | Fixed iteration budget; no validation set supplied |
| Recency weighting | 20-step half-life | None |
| Final fitting seed | 47 | 47 |

The **HGB?CatBoost Graph Ensemble** uses eight confirmed-illicit profile clusters for each weighting scheme. Initial positive weights increase with inverse square-root cluster frequency, are clipped to 0.5?3, and are normalized to preserve total illicit weight. Licit examples retain unit subtype weight. HGB additionally receives the recency weights.

The **HGB?CatBoost Graph Ensemble with Stronger Subtype Weighting** squares the normalized illicit subtype weights, clips them to 0.5?3 again, and renormalizes within illicit examples. Its features, learner settings and probability fractions are otherwise unchanged. The notebook preserves the original final-fit CatBoost thread settings: one thread for the first ensemble and four for the stronger-weighting ensemble.

## 6. Validation and recorded results

The modelling sequence is logistic regression ? standalone HGB and CatBoost ? probability blending ? subtype weighting ? graph structural features ? stronger rare-subtype weighting.

Comparison uses **three-fold stratified random validation repeated with seeds 47, 48 and 49**. Preprocessing and subtype clustering exclude held-out transactions. Mean and weakest-fold ROC-AUC summarize ranking performance; precision, recall and F1 use threshold **0.5**.

| Metric | HGB?CatBoost Graph Ensemble | Stronger-subtype-weighted ensemble |
|---|---:|---:|
| Mean validation ROC-AUC, nine folds | 0.998787 | 0.998798 |
| Weakest-fold ROC-AUC | 0.998084 | 0.998111 |
| Mean precision | 0.990680 | 0.989753 |
| Mean recall | 0.952524 | 0.953439 |
| Mean F1 | 0.971222 | 0.971247 |
| Confirmed Kaggle public ROC-AUC | **0.95132** | Not reported |

The graph ensemble has the strongest confirmed Kaggle result. The stronger-weighting candidate wins six of nine local folds, but its average AUC gain is small and seed 49 worsens on average. A higher Kaggle score has not been established.

In the seed-47 out-of-fold diagnostic, rare-subtype misses fall from 115 to 107, and rare-subtype recall increases from 64.9% to 67.4%. Overall misses decrease from 174 to 167 while false positives increase from 35 to 37. These diagnostics are distinct from the nine-fold averages above.

Random validation mixes periods and can share graph neighborhoods. Training covers steps 1?35 while test covers 36?49, so these scores do not establish performance on unseen periods. Repeated shuffles overlap, and repeated tuning on the same folds introduces selection optimism. Public leaderboard scores also do not reveal final private-leaderboard performance.

## 7. Output validation and reproducibility limits

Each output contains **15,329 rows** with the exact columns and identifier order specified by `sample_submission.csv` (`index` and `target` for the supplied data). `target` is the illicit probability, not a thresholded class label.

The notebook checks row count, columns, identifier order, missing predictions and probability bounds `[0, 1]`. It also calculates a SHA-256 digest for each exported file. These checks establish submission format, not predictive accuracy on withheld labels.

The source includes the full fitting and inference paths for both selected models. A prior partial reproduction check matched the graph ensemble's saved probability values exactly, although the CSV bytes differed. Full reproduction of the stronger-weighting ensemble has not been verified to completion. No additional training or generation was performed for this documentation update.

For comparing regenerated files, check numerical probability values and identifier alignment as well as file hashes: formatting or line-ending differences can change a hash without changing predictions.

## 8. Troubleshooting

| Symptom | Check |
|---|---|
| Missing competition files | Confirm all four CSVs are directly inside `<repository>/dataset/`, with their original names. |
| Files exist but cannot be found | Confirm the notebook kernel's working directory is the repository root; `PROJECT_DIR` uses `Path.cwd()`. |
| `ModuleNotFoundError` | Install `requirements.txt` with the same Python executable selected as the notebook kernel. |
| Output is shown as `not generated` | Confirm `REBUILD_SUBMISSIONS = True` and inspect `RESULTS_DIR`. The default is `results/`. |
| `FileExistsError` during generation | Use `OVERWRITE_SUBMISSIONS = True` intentionally, or select a new output directory. |
| CSVs appear under `results/` rather than beside the notebook | Set `RESULTS_DIR = PROJECT_DIR` to match the uploaded root-level files. |
| Run takes substantially longer than expected | Keep `USE_RECORDED_RESULTS = True` when only regenerating the final predictions. |
| Regenerated probabilities differ | Check the original data, package versions, seed, model settings and numerical environment. |
