# CO₂ Emissions Project — Final Plan (A-Level Version)

## 🎯 Goal
This document outlines the final refined research questions (R1, R2, R3) and what needs to be done for each to achieve an A/A+ level project.

---

# 🔥 R1 — Regression Analysis

## Research Question
> Which economic and energy-related variables are most strongly associated with CO₂ emissions per capita?

## What to Do

### 1. Data Preparation
- Select variables: `co2_per_capita`, `gdp`, `population`, `energy_per_capita`
- Handle missing values (drop or impute)
- Train-test split (80/20)

### 2. Models
- OLS Regression
- Ridge Regression
- Lasso Regression

### 3. Evaluation
- RMSE
- MAE
- R²

### 4. Diagnostics
- Residual plots
- VIF (multicollinearity)

### 5. Key Insight
- Identify most important variables
- Interpret coefficients

---

# 🔥 R2 — Clustering Analysis

## Research Question
> Can observations be clustered into meaningful groups based on emission-related characteristics?

## What to Do

### 1. Data Preparation
- Select features: emissions + energy variables
- Standardize data

### 2. Dimensionality Reduction
- Apply PCA
- Scree plot

### 3. Clustering
- K-means clustering
- Determine K using:
  - Elbow method
  - Silhouette score

### 4. Interpretation
- Label clusters (Low / Medium / High)
- Visualize clusters using PCA

### 5. Important Clarification
- Clustering is done on **country-year observations**
- Explain why (temporal variation)

### 6. Bonus (A+)
- Mention alternative methods:
  - Hierarchical clustering
  - DBSCAN
  - GMM

---

# 🔥 R3 — Machine Learning / Forecasting

## Research Question
> How accurately can future CO₂ emissions per capita be forecast using historical economic, demographic, and energy-related variables, and do advanced deep learning models outperform traditional machine learning baselines?

## What to Do

### 1. Problem Setup
- Treat as **time-series / panel forecasting**
- Target: `co2_per_capita`
- Use **time-based split** (NOT random split)

### 2. Feature Engineering
- Lag features:
  - `co2_lag1`, `co2_lag3`
  - `gdp_lag1`
  - `energy_lag1`
- Rolling averages / trends

---

## 🔹 Models

### Baseline Models
- Linear Regression
- Ridge/Lasso

### Tree-Based Models
- Random Forest
- XGBoost

### Neural Network
- MLP (feedforward NN)

### Sequence Model
- LSTM (for temporal modeling)

---

## 🔹 Hugging Face Model (REQUIRED)

Use at least one pretrained time-series model:

Options:
- Time Series Transformer
- PatchTST
- TimesFM

### Purpose:
- Test whether pretrained models improve performance
- Compare against LSTM and XGBoost

---

## 🔹 Ensemble Model (HIGHLY RECOMMENDED)

Combine predictions from multiple models:

Examples:
- Average ensemble:
  - (XGBoost + LSTM) / 2
- Weighted ensemble
- Stacking (advanced)

### Why:
- Improves stability and performance
- Shows deeper understanding

---

## 🔹 Evaluation

- RMSE
- MAE
- R²
- (Optional) MAPE

---

## 🔹 Key Comparison

Compare:
- Linear vs Tree vs Neural vs Pretrained
- Single model vs Ensemble

---

## 🔹 Expected Insight

- Which model performs best?
- Do deep learning models outperform classical methods?
- Does pretrained model help?

---

# 🔥 Final Notes

## To Get A / A+

You MUST:
- Clearly justify each method
- Use correct data split (time-aware)
- Provide strong interpretation (not just results)
- Show awareness of limitations

## Extra Points
- Ensemble model
- Hugging Face model
- Clean visualizations
- Strong discussion section

---

# 🚀 Final Outcome

Following this plan ensures:
- Strong methodology
- Clear research alignment
- High-quality analysis

👉 Target: **A / A+ grade**
