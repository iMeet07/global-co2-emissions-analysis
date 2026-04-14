# Global CO₂ Emissions Analysis — AMS 597 Group 9

Statistical and machine learning analysis of global CO₂ emissions using economic, demographic,
and energy indicators across a multi-country, multi-year panel dataset.

The project addresses three research questions (R1–R3), implemented across two reproducible files:
an R Markdown report and a Python Jupyter notebook.

---

## Research Questions

| RQ | Question | Method | File |
|----|----------|--------|------|
| **R1** | Which economic and energy-related variables are most strongly associated with CO₂ emissions per capita? | OLS, Ridge, Lasso regression | `R1-R2_AMS-597-Project.Rmd` |
| **R2** | Can country-year observations be grouped into meaningful emission clusters? | K-Means, Hierarchical, DBSCAN, GMM | `R1-R2_AMS-597-Project.Rmd` |
| **R3** | Can future CO₂ emissions be accurately forecast, and do deep learning models outperform classical baselines? | Linear, Ridge, Lasso, RF, XGBoost, MLP, LSTM, PatchTST, TimesFM, Ensemble | `r3_forecasting.ipynb` |

---

## Repository Structure

```
.
├── R1-R2_AMS-597-Project.Rmd     # R1 regression + R2 clustering (R)
├── r3_forecasting.ipynb          # R3 forecasting — all ML/DL models (Python)
├── co2_modeling.ipynb            # Exploratory modeling notebook (Python)
├── eda_co2.Rmd                   # Standalone EDA report (R)
├── cleaned_co2_data_20vars.csv   # Cleaned dataset (2,640 country-year rows)
├── requirements.txt              # Python dependencies
├── output_r3/
│   ├── figures/                  # R3 model prediction plots
│   └── tables/                   # R3 metrics (CSV)
├── Fix.md                        # Project planning notes
└── olds/                         # Archived earlier versions
```

---

## Data

- **Source:** [Our World in Data — CO₂ and Greenhouse Gas Emissions](https://github.com/owid/co2-data)
- **Cleaned file:** `cleaned_co2_data_20vars.csv` — years ≥ 1990, valid ISO codes, 20 variables, complete cases only
- **Dimensions:** ~2,640 country-year observations × 20 columns
- **ID columns:** `country`, `year`, `iso_code`
- **Target:** `co2_per_capita` (R1/R2), `co2` (R3)
- **Key predictors:** `gdp`, `population`, `energy_per_capita`, `total_ghg`, `coal_co2`, `oil_co2`, `gas_co2`, `methane`, and others

---

## Setup

### Python (R3 + co2_modeling.ipynb)

```bash
pip install -r requirements.txt
```

For **RTX 50-series (Blackwell / sm_120)** GPU support, install PyTorch with CUDA 12.8:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### R (R1/R2 + EDA)

Required packages (installed automatically on first knit):

```r
install.packages(c(
  "tidyverse", "glmnet", "caret", "factoextra", "cluster",
  "corrplot", "car", "knitr", "dbscan", "mclust", "gridExtra"
))
```

Knit the report to PDF:

```bash
Rscript -e "rmarkdown::render('R1-R2_AMS-597-Project.Rmd')"
```

---

## R1 — Regression Analysis

**File:** `R1-R2_AMS-597-Project.Rmd`

**Variables:** `co2_per_capita` ~ `gdp` + `population` + `energy_per_capita`

**Split:** 80% train / 20% test (random, stratified)

| Model | Description |
|-------|-------------|
| OLS | Baseline linear regression; coefficients + VIF diagnostics |
| Ridge | L2 regularization; CV-tuned `lambda` |
| Lasso | L1 regularization; CV-tuned `lambda`; implicit variable selection |

**Evaluation:** RMSE, R², MAE on held-out test set. Side-by-side bar chart and actual vs. predicted
plots for all three models. Residual diagnostics for OLS.

**Key finding:** `energy_per_capita` is the dominant predictor across all models (R² > 0.85).
Lasso confirms this through variable selection. GDP has a secondary positive effect; population
is non-significant after controlling for energy use.

---

## R2 — Clustering Analysis

**File:** `R1-R2_AMS-597-Project.Rmd`

**Features:** `co2_per_capita`, `total_ghg`, `methane`, `coal_co2`, `oil_co2`, `gas_co2`

**Unit:** Country-year observations (preserves temporal variation)

| Method | Description |
|--------|-------------|
| K-Means (k=3) | Elbow + silhouette method to select k; hard cluster assignment |
| Hierarchical (Ward's D2) | Dendrogram with `rect.hclust`; full-data `cutree(k=3)` |
| DBSCAN | kNN distance plot for `eps` tuning; noise point detection |
| GMM | `mclust`; probabilistic soft assignments; BIC model selection |

**Comparison:** Adjusted Rand Index (ARI) across all method pairs; 2×2 PCA grid; silhouette plots;
cluster profile table (mean feature values per group).

**Key finding:** K-Means, Hierarchical, and GMM converge on three coherent emission profiles
(Low / Medium / High) with high ARI (~0.7–0.9). DBSCAN isolates anomalous country-years
that resist classification.

---

## R3 — Forecasting Analysis

**File:** `r3_forecasting.ipynb`

**Target:** `co2_per_capita` | **Split:** Time-based (train ≤ 2015, val 2016–2019, test 2020+)

**Feature engineering:** Lag features (`co2_lag1`, `co2_lag3`, `gdp_lag1`, `energy_lag1`),
rolling averages, year trend.

| Category | Models |
|----------|--------|
| Baseline | Linear Regression, Ridge, Lasso |
| Tree-based | Random Forest, XGBoost |
| Neural network | MLP (feedforward NN) |
| Sequence model | LSTM |
| Pretrained (Hugging Face) | PatchTST, TimesFM |
| Ensemble | Average ensemble (XGBoost + LSTM) / 2 |

**Evaluation:** RMSE, MAE, R² on test set. Learning curves for MLP and LSTM. Per-model
actual vs. predicted plots saved to `output_r3/figures/`.

**Key finding:** Tree-based models (XGBoost, Random Forest) and the ensemble achieve the
best test-set performance. Pretrained Hugging Face models (PatchTST, TimesFM) provide
competitive zero-shot forecasts without task-specific training.

---

## EDA Report

**File:** `eda_co2.Rmd`

```bash
Rscript -e "rmarkdown::render('eda_co2.Rmd', output_format = 'html_document')"
```

Covers: data quality checks, distribution diagnostics, time trends, emission concentration,
GDP/energy relationships, source composition shifts (coal/gas/oil shares), and
multicollinearity analysis.

---

## Output Files (R3)

| Path | Description |
|------|-------------|
| `output_r3/figures/r3_model_comparison_rmse.png` | RMSE comparison across all R3 models |
| `output_r3/figures/r3_actual_vs_predicted.png` | Predicted vs actual (best model) |
| `output_r3/figures/r3_lstm_learning_curve.png` | LSTM training loss curve |
| `output_r3/figures/r3_mlp_learning_curve.png` | MLP training loss curve |
| `output_r3/figures/r3_patchtst_test_pred_vs_actual.png` | PatchTST predictions |
| `output_r3/figures/r3_timesfm_test_pred_vs_actual.png` | TimesFM predictions |
| `output_r3/figures/r3_xgboost_test_pred_vs_actual.png` | XGBoost predictions |
| `output_r3/tables/r3_model_comparison.csv` | Full metrics table (RMSE, MAE, R²) |
| `output_r3/tables/r3_hf_status.csv` | Hugging Face model availability log |

---

## Authors

Group 9 — AMS 597 Statistical Computing, Spring 2026
