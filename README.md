# Global CO₂ Emissions Analysis — AMS 597 Group 9

Statistical and machine learning analysis of global CO₂ emissions using economic, demographic,
and energy indicators across a multi-country, multi-year panel dataset.

The project addresses three research questions (R1–R3), implemented across two reproducible files:
an R Markdown report (`R1-R2_AMS-597-Project.Rmd`) and a Python Jupyter notebook (`r3_forecasting.ipynb`).

---

## Research Questions

| RQ | Question | Method | File |
|----|----------|--------|------|
| **R1** | Which economic and energy-related variables are most strongly associated with CO₂ emissions per capita? | OLS, Ridge, Lasso | `R1-R2_AMS-597-Project.Rmd` |
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
├── cleaned_co2_data_20vars.csv   # Cleaned dataset (~2,640 country-year rows)
├── requirements.txt              # Python dependencies
├── figures/                      # R1/R2 output figures (from knitted Rmd)
├── output_r3/
│   ├── figures/                  # R3 model prediction and comparison plots
│   └── tables/                   # R3 metrics CSV files
├── output/figures/               # co2_modeling.ipynb output figures
├── Fix.md                        # Project planning notes
└── olds/                         # Archived earlier versions
```

---

## Data

- **Source:** [Our World in Data — CO₂ and Greenhouse Gas Emissions](https://github.com/owid/co2-data)
- **Cleaned file:** `cleaned_co2_data_20vars.csv` — years ≥ 1990, valid ISO codes, complete cases only
- **Dimensions:** ~2,640 country-year observations × 20 columns
- **Target (R1/R2):** `co2_per_capita` | **Target (R3):** `co2_per_capita`
- **Key predictors:** `gdp`, `population`, `energy_per_capita`, `total_ghg`, `coal_co2`, `oil_co2`, `gas_co2`, `methane`, and others

---

## Setup

### Python (R3)

```bash
pip install -r requirements.txt
```

For **RTX 50-series (Blackwell / sm_120)** GPU support:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### R (R1/R2 + EDA)

```r
install.packages(c(
  "tidyverse", "glmnet", "caret", "factoextra", "cluster",
  "corrplot", "car", "knitr", "dbscan", "mclust", "gridExtra"
))
```

```bash
Rscript -e "rmarkdown::render('R1-R2_AMS-597-Project.Rmd')"
```

---

## R1 — Regression Analysis

**Research Question:** *Which economic and energy-related variables are most strongly associated with CO₂ emissions per capita?*

**Variables:** `co2_per_capita` ~ `gdp` + `population` + `energy_per_capita` | **Split:** 80/20 train-test

### Correlation Matrix

![Correlation Matrix](figures/r1-correlation-1.png)

### Model Comparison (RMSE / R² / MAE)

![R1 Model Comparison Bar Chart](figures/r1-comparison-barchart-1.png)

### Actual vs Predicted — OLS, Ridge, Lasso

![R1 Actual vs Predicted](figures/r1-actual-vs-predicted-1.png)

### OLS Residual Diagnostics

![R1 Residual Plots](figures/r1-residuals-1.png)

### Key Finding

`energy_per_capita` is the dominant predictor across all three models (R² > 0.85). Lasso confirms
this through variable selection — GDP has a secondary positive effect while population is
non-significant after controlling for energy use.

---

## R2 — Clustering Analysis

**Research Question:** *Can country-year observations be grouped into meaningful emission clusters?*

**Features:** `co2_per_capita`, `total_ghg`, `methane`, `coal_co2`, `oil_co2`, `gas_co2` | **Unit:** Country-year observations

### EDA: GDP vs CO₂

![EDA Plot](figures/eda-1.png)

### PCA Scree Plot

![R2 PCA Scree Plot](figures/r2-pca-1.png)

### Optimal K — Elbow and Silhouette

![Elbow Method](figures/r2-optimal-k-1.png)

![Silhouette Method](figures/r2-optimal-k-2.png)

### K-Means Clustering (k = 3)

![K-Means PCA Plot](figures/r2-kmeans-1.png)

### Hierarchical Clustering — Dendrogram

![Hierarchical Dendrogram](figures/r2-hierarchical-1.png)

![Hierarchical PCA Plot](figures/r2-hierarchical-2.png)

### DBSCAN Clustering

![DBSCAN kNN Distance Plot](figures/r2-dbscan-1.png)

![DBSCAN PCA Plot](figures/r2-dbscan-2.png)

### GMM — Gaussian Mixture Model

![GMM PCA Plot](figures/r2-gmm-1.png)

### All Methods Side-by-Side

![All Clustering Methods](figures/r2-side-by-side-1.png)

### Silhouette Comparison: K-Means vs GMM

![Silhouette Plots](figures/r2-silhouette-plots-1.png)

### Key Finding

K-Means, Hierarchical, and GMM converge on three coherent emission profiles (Low / Medium / High)
with high Adjusted Rand Index (~0.7–0.9), confirming robustness across methods.
DBSCAN uniquely identifies anomalous country-year outliers that resist clean classification.

---

## R3 — Forecasting Analysis

**Research Question:** *Can future CO₂ emissions be accurately forecast, and do deep learning models outperform classical baselines?*

**Target:** `co2_per_capita` | **Split:** Time-based (train ≤ 2015, val 2016–2019, test 2020–2022)

**Feature engineering:** Lag features (`co2_lag1`, `co2_lag3`, `gdp_lag1`, `energy_lag1`), 3- and 5-year rolling means, year trend.

### Models implemented

| Category | Models |
|----------|--------|
| **Linear baselines** | Linear Regression, Ridge, Lasso, ElasticNet |
| **Tree-based** | Random Forest, XGBoost (CUDA if available) |
| **Neural networks** | MLP (sklearn), LSTM (TensorFlow / GPU) |
| **Pretrained (HF)** | PatchTST (`ibm/patchtst-etth1-pretrain`), TimesFM (`google/timesfm-1.0-200m-pytorch`) — per-country, zero-shot |
| **Ensembles** | Simple Average (top-2 by RMSE), Weighted (val-RMSE optimized weights via L-BFGS-B), **Stacking (Ridge meta-learner, 5-fold OOF)** |

### Model Comparison (Test RMSE)

![R3 Model Comparison RMSE](output_r3/figures/r3_model_comparison_rmse.png)

### Test Set Results

| Model | Test RMSE | Test MAE | Test R² | Test MAPE |
|-------|-----------|----------|---------|-----------|
| 🏆 **Ensemble_Stacking** | **0.357** | **0.208** | **0.9961** | **4.95%** |
| Lasso | 0.376 | 0.207 | 0.9957 | 4.85% |
| Ensemble_Avg (Lasso+ElasticNet) | 0.386 | 0.209 | 0.9955 | 5.08% |
| Ensemble_Weighted | 0.396 | 0.216 | 0.9952 | 4.75% |
| ElasticNet | 0.400 | 0.213 | 0.9951 | 5.37% |
| Linear | 0.406 | 0.224 | 0.9950 | 5.19% |
| Ridge | 0.408 | 0.213 | 0.9950 | 5.27% |
| XGBoost | 0.502 | 0.283 | 0.9924 | 5.37% |
| Random Forest | 0.553 | 0.299 | 0.9907 | 5.35% |
| MLP | 0.584 | 0.347 | 0.9897 | 8.87% |
| PatchTST | 0.901 | 0.529 | 0.9754 | 10.16% |
| TimesFM | 0.947 | 0.541 | 0.9728 | 9.93% |
| LSTM | 1.181 | 0.658 | 0.9581 | 13.45% |

### Best Model — Stacking Ensemble (Predicted vs Actual)

![Stacking Ensemble Predictions](output_r3/figures/r3_ensemble_stacking_test_pred_vs_actual.png)

### Lasso Regression

![Lasso Predictions](output_r3/figures/r3_lasso_test_pred_vs_actual.png)

### ElasticNet Regression

![ElasticNet Predictions](output_r3/figures/r3_elasticnet_test_pred_vs_actual.png)

### Ridge Regression

![Ridge Predictions](output_r3/figures/r3_ridge_test_pred_vs_actual.png)

### Linear Regression

![Linear Predictions](output_r3/figures/r3_linear_test_pred_vs_actual.png)

### Random Forest

![Random Forest Predictions](output_r3/figures/r3_randomforest_test_pred_vs_actual.png)

### XGBoost (CUDA)

![XGBoost Predictions](output_r3/figures/r3_xgboost_test_pred_vs_actual.png)

### MLP — Learning Curve & Predictions

![MLP Learning Curve](output_r3/figures/r3_mlp_learning_curve.png)

![MLP Predictions](output_r3/figures/r3_mlp_test_pred_vs_actual.png)

### LSTM — Learning Curve & Predictions

![LSTM Learning Curve](output_r3/figures/r3_lstm_learning_curve.png)

![LSTM Predictions](output_r3/figures/r3_lstm_test_seq_pred_vs_actual.png)

### PatchTST (Hugging Face) — Predictions & Yearly Forecast

![PatchTST Predictions](output_r3/figures/r3_patchtst_test_pred_vs_actual.png)

![PatchTST Yearly Forecast](output_r3/figures/r3_patchtst_yearly_forecast.png)

### TimesFM (Google / Hugging Face) — Predictions & Yearly Forecast

![TimesFM Predictions](output_r3/figures/r3_timesfm_test_pred_vs_actual.png)

![TimesFM Yearly Forecast](output_r3/figures/r3_timesfm_yearly_forecast.png)

### Ensemble Comparison

| Ensemble type | Strategy | Test RMSE |
|---------------|----------|-----------|
| Simple Average | Average of top-2 models by RMSE | 0.386 |
| Weighted | L-BFGS-B optimized weights on val RMSE | 0.396 |
| **Stacking** | Ridge meta-learner on 5-fold OOF predictions | **0.357** |

![Stacking Predictions](output_r3/figures/r3_ensemble_stacking_test_pred_vs_actual.png)

### Key Findings

- **Stacking ensemble achieves the best test RMSE (0.357, R² = 0.9961)**, outperforming all individual models and simpler ensembles. The Ridge meta-learner (tuned via 3-fold CV) learns to weight linear regularized models most heavily, and appropriately down-weights tree-based models on this dataset.
- **Lasso is the best single model** (RMSE = 0.376, R² = 0.9957), confirming that strong L1 regularization fits the co2 per capita regression well given collinear economic/energy predictors.
- **ElasticNet** (L1+L2 combined, RMSE = 0.400) provides a robust alternative with better generalization than Ridge when predictors are moderately correlated.
- **Pretrained HF models** (PatchTST, TimesFM) deliver competitive zero-shot performance (R² ≈ 0.973–0.975) per country without any task-specific fine-tuning, demonstrating the value of large-scale time-series pretraining.
- **Deep learning** (LSTM, MLP) underperforms linear methods on this relatively small panel dataset, where regularized regression captures the dominant linear signal more efficiently.

---

## Authors

Group 9 — AMS 597 Statistical Computing, Spring 2026
