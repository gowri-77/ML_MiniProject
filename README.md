# ML_MiniProject

**Nepal Earthquake Damage Prediction**

- **Course:** UE24CS352A Machine Learning, Mini-Project

- **Team H05:** Gowri T N (PES1UG24AM431) and Rishi Manjunath (PES1UG24AM445)


Predicting the damage grade (1 = low, 2 = medium, 3 = near-total destruction) of buildings hit by the 2015 Gorkha earthquake, using the DrivenData *Richter's Predictor: Modeling Earthquake Damage* dataset. We compare KNN, a neural network, Random Forest and LightGBM, and reproduce and fix the overfitting reported in the Stanford CS229 reference study.

## Results (validation set, 52,121 buildings)

| Model | Train micro-F1 | Val micro-F1 | Val macro-F1 |
|---|---|---|---|
| KNN (k=15) | 0.6339 | 0.6304 | 0.5539 |
| KNN (k=51) | 0.6463 | 0.6461 | 0.5490 |
| MLP (10 units) | 0.6782 | 0.6847 | 0.6075 |
| MLP (64-32, alpha=1e-4) | 0.6950 | 0.6940 | 0.6183 |
| MLP (64-32, alpha=1e-3) | 0.6956 | 0.6950 | 0.6315 |
| MLP (64-32, alpha=1e-2) | 0.6959 | 0.6941 | 0.6255 |
| Random Forest (baseline) | 0.9868 | 0.7230 | 0.6662 |
| Random Forest (tuned) | 0.8362 | 0.7428 | 0.6835 |
| Random Forest (balanced) | 0.8016 | 0.7029 | 0.6763 |
| **LightGBM (tuned)** | 0.8337 | **0.7468** | **0.6958** |

The full table, including training times, is in `results/results.csv`. The write-up explains the results in detail.

## Project structure

```
ML_MiniProject/
  README.md
  requirements.txt
  data/                      dataset CSVs (train_values, train_labels, test_values)
  notebooks/
    01_eda.ipynb             exploratory data analysis
    02_knn.ipynb             KNN baseline and k search
    03_nn.ipynb              neural network (MLP) experiments
    04_random_forest.ipynb   RF baseline, tuning, class weights
    05_boosting.ipynb        LightGBM, cross-validation, feature importance
  src/
    preprocess.py            shared data split and preprocessors
    evaluate.py              logs each model's scores to results.csv
    predict_demo.py          live demo: loads the saved model and predicts
  models/                    saved models (knn, nn, lgbm)
  results/
    results.csv              scores of every model
    figures/                 all plots used in the report
```

## Setup

You need **Python 3.12** and Git.

### 1. Clone the repository

```
git clone https://github.com/gowri-77/ML_MiniProject.git
cd ML_MiniProject
```

### 2. Create a virtual environment and install packages

Windows (PowerShell):

```
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows (Command Prompt):

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Mac / Linux:

```
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

`(venv)` should now appear at the start of your prompt.

> **Library versions matter.** The saved models in `models/` were created with numpy 2.2.6, scikit-learn 1.7.2 and LightGBM 4.5.0, which are pinned in `requirements.txt`. Other versions may print warnings when the models load, and LightGBM 4.5.0 does not work with scikit-learn 1.8 or newer.

### 3. Data

The dataset is already included in the `data/` folder, so no download is needed:

```
data/train_values.csv
data/train_labels.csv
data/test_values.csv
```

The data comes from the DrivenData competition **Richter's Predictor: Modeling Earthquake Damage**. If any file is missing or damaged, create a free account on [DrivenData](https://www.drivendata.org), download the three files from that competition and put them directly inside `data/`.

The test labels are hidden, so all scores come from a stratified 80/20 split of the training data (`random_state=42`).

### 4. Check that everything works

```
python -m src.preprocess
```

Expected output: the split shapes `(208480, 38) (52121, 38) ...`, the class proportions (about 0.569, 0.335, 0.096), and the transformed shapes `linear -> (208480, 96)` and `tree -> (208480, 38)`.

## Run the project

### Live demo (no retraining)

```
python -m src.predict_demo
```

This loads the saved LightGBM pipeline from `models/lgbm.joblib`, predicts the damage grade for a few validation buildings and prints the true grade next to the predicted grade. It takes a few seconds. If scikit-learn prints version warnings, run `python -W ignore -m src.predict_demo`.

### Reproduce the experiments

Start Jupyter **from the `notebooks` folder**, because the notebooks use relative paths such as `../data` and `../results`:

```
cd notebooks
python -m notebook
```

If the browser does not open, copy the `http://localhost:8888/...` link from the terminal into your browser. Run the notebooks in this order, and run each cell once:

| Order | Notebook | What it does |
|---|---|---|
| 1 | `01_eda.ipynb` | Class balance and damage by foundation, floors and region |
| 2 | `02_knn.ipynb` | KNN with k search (3 to 51) |
| 3 | `03_nn.ipynb` | MLP with 4 configurations and loss curves |
| 4 | `04_random_forest.ipynb` | Baseline, tuning, depth plot, class weights (the search takes about 10 minutes) |
| 5 | `05_boosting.ipynb` | LightGBM tuning, 5-fold CV, feature importance |

Notes:

- Each model notebook **appends** a row to `results/results.csv`. To start fresh, delete that file before running the notebooks. Notebook 05 replaces its own `lightgbm` row.
- Notebook 05 overwrites `models/lgbm.joblib` with a scikit-learn `Pipeline` (preprocessing and model together).
- The tuned Random Forest model is not stored in the repository because of its size. Run `04_random_forest.ipynb` to rebuild it.

## Method summary

- **Two preprocessing pipelines.** KNN and the MLP use one-hot encoding plus standardization (96 features, the two high-cardinality location IDs dropped). The tree models use ordinal encoding and keep all 38 features, including every location level.
- **Metrics.** Micro-F1 and macro-F1, because the classes are imbalanced (about 57% Grade 2, 34% Grade 3, 10% Grade 1).
- **Findings.** Tree ensembles beat KNN and the MLP by about five points. An unconstrained Random Forest overfits (train 0.9868, validation 0.7230), and larger leaves with feature subsampling shrink the gap from 26.38% to 9.34%. Tuned LightGBM is the best model.

## Contributions

| Part | Owner | Work |
|---|---|---|
| Part 1 | Gowri T N | Repository setup, shared preprocessing and results logger, EDA, KNN, neural network |
| Part 2 | Rishi Manjunath | Random Forest and tuning, class weighting, LightGBM, cross-validation, feature importance, demo script |
| Shared | Both | Write-up, slides, final review |

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'src'` | Run commands from the repository root, or start Jupyter from the `notebooks` folder |
| `FileNotFoundError` for a CSV | Check that the three data files are in `data/` with their exact names, and that they were not shortened by a failed download |
| `pip.exe` blocked or not found | Use `python -m pip install -r requirements.txt` instead |
| Version warnings when loading a model | Install the pinned versions from `requirements.txt` |
