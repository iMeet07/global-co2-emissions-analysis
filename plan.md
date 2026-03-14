# CO2 Emissions Modeling Plan

## Data

- **Source:** Our World in Data CO₂/GHG emissions ([dataset.md](dataset.md)).
- **Cleaning:** [data_clean.Rmd](data_clean.Rmd) — years ≥ 1990, valid `iso_code`, 20 variables, complete cases only.
- **File:** [cleaned_co2_data_20vars.csv](cleaned_co2_data_20vars.csv) — ~2,641 country-year rows, 20 columns.
- **Variables:** 3 IDs (`country`, `year`, `iso_code`); 17 numeric (economics, energy, emissions). **Target:** `co2` (total CO₂; avoid predicting `co2_per_capita` from `co2`/`population` to prevent leakage).

---

## 1. Preliminaries

- **Splits:** Time-based — train 1990–2015, validation 2016–2019, test 2020–2022.
- **Scaling:** Standardize predictors (and target if needed) using training set only; apply to val/test.
- **Evaluation:** RMSE, MAE, R² on validation and test.

---

## 2. PCA

- Standardize features (train fit); run PCA on numeric predictors (exclude target or all emission vars for interpretation).
- Choose number of components by cumulative variance (e.g. ≥85–90%) and scree plot.
- Outputs: scree plot, loadings table, component-vs-target plot; optional PCA-score dataset.

---

## 3. Regression

- **Linear regression** — all predictors (+ optional `year`); check VIF and residual plots.
- **Ridge / Lasso / Elastic Net** — regularized; use validation or CV for penalty; Lasso for variable selection.
- Optional: regression on PCA scores; compare R² and RMSE to raw-predictor model.
- Deliverables: coefficients, RMSE/MAE/R², residual diagnostics, coefficient plot.

---

## 4. Clustering

- K-means on standardized features or PCA scores (first 2–5 components).
- Choose K by elbow or silhouette score.
- Summarize clusters (mean emissions, energy mix); optionally use cluster as feature.

---

## 5. Feedforward NN

- Standardized inputs; 1–2 hidden layers (e.g. 64–128 units), ReLU; MSE loss.
- Early stopping on validation loss; same train/val/test split.
- Deliverables: val/test RMSE/MAE/R², learning curves.

---

## 6. LSTM

- Per-country sequences: (X_{t-k}, …, X_t) → y_t or y_{t+1}; sequence length 5–10 years.
- Standardize by train; LSTM + dense to scalar; time-ordered splits.
- Deliverables: test RMSE/MAE/R²; optional predicted vs actual plot.

---

## 7. GNN (optional)

- Build graph: nodes = countries (or country-years); edges by similarity (e.g. correlation or PCA distance).
- GCN: node features = numeric vars; predict `co2`. Optional if time permits.

---

## Order of implementation

| Step | Content |
|------|--------|
| 1 | Load data, time splits, scaling, evaluation helper |
| 2 | PCA: scree, loadings, component plots |
| 3 | Regression: linear, Ridge, Lasso, Elastic Net; PCA regression |
| 4 | Clustering: K-means on PCA scores |
| 5 | Feedforward NN |
| 6 | LSTM |
| 7 | GNN (optional) |

All steps are implemented in a single Jupyter notebook: **`co2_modeling.ipynb`**.
