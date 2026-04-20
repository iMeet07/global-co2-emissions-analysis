# Global CO₂ Emissions Analysis — AMS 597 Group 9

Statistical and machine learning analysis of global CO₂ emissions using economic, demographic,
and energy indicators across a multi-country, multi-year panel dataset.

The project addresses four research questions (R1–R4), implemented across R Markdown reports and a Python Jupyter notebook.

---

## Research Questions

| RQ | Question | Method | File |
|----|----------|--------|------|
| **R1** | Which economic and energy-related variables are most strongly associated with CO₂ emissions per capita? | OLS, Ridge, Lasso | `R1-R2_AMS-597-Project.Rmd` |
| **R2** | Can country-year observations be grouped into meaningful emission clusters? | K-Means, Hierarchical, DBSCAN, GMM | `R1-R2_AMS-597-Project.Rmd` |
| **R3** | Can future CO₂ emissions be accurately forecast, and do deep learning models outperform classical baselines? | Linear, Ridge, Lasso, ElasticNet, RF, XGBoost, MLP, LSTM, PatchTST, TimesFM, Ensemble (Weighted/Avg/Stacking) | `r3_forecasting.ipynb` |
| **R4** | Do countries become more energy-efficient as they develop, and how does this affect CO₂ emissions over time? | OLS variants, interaction model, fixed effects panel, EKC test | `R4_Addition.Rmd` |

---

## Repository Structure

```
.
├── R1-R2_AMS-597-Project.Rmd     # R1 regression + R2 clustering (R)
├── R4_Addition.Rmd               # R4 development-efficiency-emissions analysis (R)
├── r3_forecasting.ipynb          # R3 forecasting — all ML/DL models (Python)
├── co2_modeling.ipynb            # Exploratory modeling notebook (Python)
├── eda_co2.Rmd                   # Standalone EDA report (R)
├── cleaned_co2_data_20vars.csv   # Cleaned dataset (~2,640 country-year rows)
├── requirements.txt              # Python dependencies
├── figures/                      # R1/R2/R4 output figures (r1-*/r2-*/r4-* naming)
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

**Safe covariates (no leakage):** `population`, `gdp`, `primary_energy_consumption`, `energy_per_capita`, `energy_per_gdp`

**Engineered features (all shifted):** `co2_lag1`, `co2_lag3`, `gdp_lag1`, `energy_lag1`, `co2_roll3_mean` (shift 1), `co2_roll5_mean` (shift 1)

**Excluded from features (leakage):** `co2`, `total_ghg`, `consumption_co2`, `coal_co2`, `oil_co2`, `gas_co2` — these are components or near-duplicates of the target (`co2_per_capita = co2 / population`) and would trivially reveal the answer.

> **Leakage fixes applied (Apr 2026):** (1) Removed all CO₂-component columns (`co2`, `total_ghg`, `coal_co2`, `oil_co2`, `gas_co2`, `consumption_co2`) from features — using `df.columns` wholesale was pulling these in silently. (2) Rolling means use `.shift(1)` so year *t*'s feature only sees years *t-1* and earlier. (3) LSTM uses a separate scaler fitted on train-only data. (4) Ensemble members are selected by **Val RMSE**, not Test RMSE. (5) Stacking uses `TimeSeriesSplit` instead of shuffled KFold. These corrections raise reported RMSE values; the new numbers are more trustworthy.

### Models implemented

| Category | Models |
|----------|--------|
| **Linear baselines** | Linear Regression, Ridge, Lasso, ElasticNet |
| **Tree-based** | Random Forest, XGBoost (CUDA if available) |
| **Neural networks** | MLP (sklearn), LSTM (TensorFlow / GPU) |
| **Pretrained (HF)** | PatchTST (`ibm/patchtst-etth1-pretrain`), TimesFM (`google/timesfm-1.0-200m-pytorch`) — per-country, zero-shot |
| **Ensembles** | Simple Average (top-2 by **Val** RMSE), Weighted (combo grid-search over all 2–4 subsets of top-6 candidates, SLSQP simplex weights on val only), Stacking (XGBoost or Ridge meta-learner auto-selected by OOF-val RMSE, `TimeSeriesSplit` 5-fold OOF) |

### Model Comparison (Test RMSE)

![R3 Model Comparison RMSE](output_r3/figures/r3_model_comparison_rmse.png)

### Test Set Results

| Model | Test RMSE | Test MAE | Test R² | Test MAPE |
|-------|-----------|----------|---------|-----------|
| 🏆 **Ridge** | **0.5989** | **0.3126** | **0.9891** | **8.07%** |
| ElasticNet | 0.5996 | 0.3172 | 0.9891 | 8.21% |
| Linear | 0.6001 | 0.3072 | 0.9891 | 7.77% |
| Ensemble_Avg(ElasticNet+Lasso) | 0.6004 | 0.3169 | 0.9891 | 8.13% |
| Lasso | 0.6015 | 0.3170 | 0.9890 | 8.07% |
| Ensemble_Weighted | 0.6108 | 0.3175 | 0.9887 | 6.97% |
| Ensemble_Stacking | 0.6315 | 0.3415 | 0.9879 | 10.91% |
| Ensemble_Avg(Ensemble_Weighted+XGBoost) | 0.6573 | 0.3490 | 0.9869 | 6.98% |
| Random Forest | 0.6823 | 0.4008 | 0.9859 | 7.52% |
| XGBoost | 0.7247 | 0.3933 | 0.9841 | 7.46% |
| MLP | 0.7825 | 0.4662 | 0.9814 | 14.05% |
| LSTM | 0.8878 | 0.5293 | 0.9764 | 14.46% |
| PatchTST | 0.9014 | 0.5293 | 0.9754 | 10.16% |
| TimesFM | 0.9473 | 0.5406 | 0.9728 | 9.93% |

### Best Model — Ridge (Predicted vs Actual)

![Ridge Predictions — Best Model](output_r3/figures/r3_ridge_test_pred_vs_actual.png)

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

| Ensemble type | Strategy | Val RMSE | Test RMSE |
|---------------|----------|----------|-----------|
| Simple Average | Average of ElasticNet+Lasso (top-2 by val RMSE) | 0.5169 | 0.6004 |
| **Weighted** | Combo grid-search (2–4 subsets of top-6), SLSQP on val | **0.4723** | 0.6108 |
| Stacking | XGBoost meta-learner (OOF-val selected), `TimeSeriesSplit` 5-fold | 0.4232 | 0.6315 |

### Key Findings

- **Ridge is the best overall model** (Test RMSE = 0.5989, R² = 0.9891). All four linear baselines (Ridge 0.5989, ElasticNet 0.5996, Linear 0.6001, Lasso 0.6015) are tightly clustered, confirming that the dominant signal in this panel dataset is well captured by regularized linear regression.
- **Best ensemble trails Ridge by only 0.0015 RMSE**: Ensemble_Avg(ElasticNet+Lasso) scores 0.6004, demonstrating that averaging two near-identical linear models provides minimal variance reduction.
- **Weighted ensemble and stacking achieve excellent val RMSE** (0.4723 and 0.4232 respectively) but do not generalize as well to test (0.6108 and 0.6315). The combo grid-search selects an XGBoost+Linear combination that captures val-period structure but overfits to pre-COVID trends, which differ from the 2020–2022 COVID shock in the test set. This illustrates that low val RMSE alone does not guarantee test-set gains when distribution shift is present.
- **Pretrained HF models** (PatchTST, TimesFM) deliver competitive zero-shot performance (R² ≈ 0.973–0.975) without any task-specific fine-tuning, demonstrating the value of large-scale time-series pretraining.
- **Deep learning** (LSTM, MLP) underperforms linear methods on this small panel dataset, where regularized regression captures the dominant linear signal more efficiently than sequence models.
- **RMSE values are higher than pre-fix runs** (e.g. Ridge 0.5989 vs previously reported ~0.38). This is expected and correct — earlier results were inflated by rolling features that included the target, test-set-driven ensemble selection, and shuffled KFold CV. The current numbers reflect a genuinely out-of-sample evaluation.

---

## R4 — Energy Efficiency and Development Analysis

**Research Question:** *Do countries become more energy-efficient as they develop, and how does this affect CO₂ emissions over time?*

**File:** `R4_Addition.Rmd` | **Output:** knitted PDF report + figures in `figures/r4-*.png`

**Variables:** `gdp`, `energy_per_gdp` (energy intensity), `co2_per_capita`, `population`, `energy_per_capita` | **Unit:** Country-year panel observations

### Motivation

R4 extends the project beyond prediction and clustering by examining whether economic development changes the *way* energy is used, and whether those efficiency gains translate into lower emissions. Three specific questions are addressed:

1. As GDP rises, do countries use less energy per unit of output (falling `energy_per_gdp`)?
2. Is higher energy intensity (`energy_per_gdp`) associated with higher `co2_per_capita`?
3. Does GDP's effect on emissions depend on energy intensity (interaction effect)?

### Models

| Model | Specification | Purpose |
|-------|--------------|---------|
| **Model 1** | `co2 ~ gdp + energy_per_gdp` | Baseline: development + efficiency |
| **Model 2** | `+ population + energy_per_capita` | Add demographic and energy-use controls |
| **Model 3** | `gdp × energy_per_gdp` interaction | Test whether efficiency moderates GDP effect |
| **Model 4** | Fixed effects panel: `co2 ~ log(gdp) + energy_per_gdp` | Within-country variation over time (controls for unobserved country traits) |
| **Model 5 (EKC)** | `co2 ~ gdp + gdp² + energy_per_gdp` | Environmental Kuznets Curve — nonlinear GDP-emissions relationship |

### Exploratory Plots

#### GDP vs Energy Intensity

![GDP vs Energy per GDP](figures/r4-plot-gdp-energyeff-1.png)

**X-axis:** GDP per capita (log scale) | **Y-axis:** Energy intensity (`energy_per_gdp`, TWh per $M GDP) | **Each point:** one country-year observation.

A clear negative slope confirms the energy efficiency hypothesis — as countries grow wealthier, they produce each unit of GDP using progressively less energy. The scatter is wide at low GDP levels (developing economies use energy very differently) and tightens at higher incomes, where industrial and infrastructure patterns converge.

#### Energy Intensity vs CO₂ per Capita

![Energy Efficiency vs CO2](figures/r4-plot-energyeff-co2-1.png)

**X-axis:** Energy intensity (`energy_per_gdp`) | **Y-axis:** CO₂ emissions per capita.

Near-linear positive relationship: countries that consume more energy per dollar of GDP systematically produce more CO₂ per person. This is the direct channel linking inefficient energy use to emissions. Notable outliers at high intensity tend to be fossil-fuel-dependent economies in early industrialisation; outliers at low intensity include high-income service economies.

#### GDP vs CO₂ per Capita (Coloured by Energy Intensity)

![GDP vs CO2 coloured](figures/r4-plot-gdp-co2-colored-1.png)

**X-axis:** GDP per capita | **Y-axis:** CO₂ per capita | **Colour:** tertile of `energy_per_gdp` (light = low intensity / efficient, dark = high intensity / inefficient).

The colour gradient reveals that energy efficiency is a confounding variable: at the same GDP level, high-intensity countries (dark) consistently emit more CO₂. This motivates adding `energy_per_gdp` as a control variable in Models 2–5 — GDP alone is an incomplete predictor of emissions.

### Aggregate Trends Over Time

#### Average GDP per Capita (1990–2022)

![Avg GDP over time](figures/r4-plot-year-gdp-1.png)

**X-axis:** Year | **Y-axis:** Cross-country average GDP per capita (constant USD).

Steady upward trend throughout the period, with a visible dip around 2008–2009 (Global Financial Crisis) and a sharper drop in 2020 (COVID-19) followed by a partial recovery. The long-run growth trend is uninterrupted across both shocks.

#### Average Energy Intensity Over Time

![Avg Energy per GDP over time](figures/r4-plot-year-energyeff-1.png)

**X-axis:** Year | **Y-axis:** Cross-country average `energy_per_gdp`.

A consistent downward trend — the world economy has become steadily more energy-efficient since 1990. The decline accelerates slightly post-2000 as China, India, and other large emerging economies modernised industrial processes. This is one of the most encouraging structural trends in global climate data.

#### Average CO₂ per Capita Over Time

![Avg CO2 per capita over time](figures/r4-plot-year-co2-1.png)

**X-axis:** Year | **Y-axis:** Cross-country average CO₂ emissions per capita (metric tonnes).

Broadly flat from 1990 to ~2000, then rising through 2010 driven by rapid industrialisation in Asia, followed by a plateau and a slight dip post-2015. The 2020 drop reflects COVID-19 mobility restrictions. The flatness in per-capita terms despite rising GDP is partly explained by improving energy intensity (see plot above) — efficiency gains have offset some of the emissions pressure from growth.

### Interaction Model Visualization

![Interaction plot](figures/r4-interaction-plot-1.png)

**X-axis:** GDP per capita | **Y-axis:** CO₂ per capita | **Lines:** predicted emissions for three groups defined by energy intensity tertile (low / medium / high `energy_per_gdp`).

The GDP–emissions slope is steepest for the high-energy-intensity group and flattest for the low-intensity group. Visually, high-intensity countries gain more CO₂ per additional unit of GDP growth. However, the interaction coefficient in Model 3 (`gdp × energy_per_gdp`) is not statistically significant — the divergence between groups is visible but is better explained by additive effects than by a true multiplicative interaction in a linear framework.

### Residual Diagnostics (Model 2)

![Diagnostics](figures/r4-diagnostics-1.png)

Four-panel standard regression diagnostic plot for Model 2 (`co2 ~ gdp + population + energy_per_capita + energy_per_gdp`):

- **Residuals vs Fitted:** slight heteroscedasticity at large fitted values — wealthier, high-emission countries have more variable residuals. A log transformation of the target or predictors would reduce this.
- **Q-Q plot:** residuals follow the normal line closely in the centre but show heavy right-tail deviation — a small number of extreme emitters (e.g. Gulf petrostate country-years) pull the distribution away from normality.
- **Scale-Location:** confirms heteroscedasticity; variance increases with fitted values.
- **Residuals vs Leverage:** a handful of influential points with both high leverage and moderate residuals (petrostate outliers). These are not data errors but genuine structural outliers that Model 4 (fixed effects) partially addresses by absorbing country-specific intercepts.

### Model Comparison

| Model | Adj. R² | AIC | Notes |
|-------|---------|-----|-------|
| Model 1: GDP + Energy/GDP | low–moderate | highest | Baseline only |
| Model 2: + Controls | substantially higher | lower | `energy_per_capita` dominates |
| Model 3: Interaction | ≈ Model 2 | similar | Interaction not significant |
| Model 5: EKC | marginal gain | lower | Nonlinear term significant |

### Key Findings

- **Energy intensity (`energy_per_gdp`) is a robust predictor** of CO₂ emissions per capita across all model specifications — more so than GDP level alone.
- **Countries do become more energy-efficient as they develop** (negative GDP–energy intensity relationship), but the efficiency gains are gradual and do not automatically translate into falling per-capita emissions in the short run.
- **GDP effect is positive but secondary**: once `energy_per_capita` is controlled for in Model 2, GDP adds relatively little additional explanatory power. Emissions are driven more by *how much* energy is consumed than by economic size.
- **Interaction (Model 3) is not significant**: the effect of GDP on emissions does not meaningfully change across energy intensity levels in a linear framework.
- **EKC evidence (Model 5)**: the positive GDP coefficient and negative GDP² coefficient are consistent with an inverted-U pattern — emissions may rise in early development and peak then decline at higher income levels, though the turning point is far above most countries' current GDP.
- **Fixed effects (Model 4)** confirm that the GDP–energy intensity–emissions relationships also hold *within* countries over time, ruling out pure cross-sectional confounding.
- **Policy implication**: interventions targeting energy efficiency and decarbonising energy supply are likely to reduce emissions more effectively than policies focused solely on limiting economic growth.

---

## Authors

Group 9 — AMS 597 Statistical Computing, Spring 2026
