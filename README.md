# Venice Surge Prediction

XGBoost models for meteorological surge and PAW (Planetary Atmospheric Wave) surge prediction at Venice, using mean sea-level pressure data provided by **CNR ISMAR** (Istituto di Scienze Marine, Consiglio Nazionale delle Ricerche).

---

## Overview

Sea level at Venice is affected by two distinct surge components:

- **Meteorological surge** — short-term sea level anomalies driven by local atmospheric pressure and wind forcing over the Adriatic Sea (3-hourly resolution).
- **PAW surge** — low-frequency oscillations of the entire Adriatic water mass driven by large-scale planetary atmospheric waves over the European domain (daily resolution).

This project trains and evaluates XGBoost regression models for each component, using gridded mean sea-level pressure (MSL) and its temporal lags as input features. SHAP (SHapley Additive exPlanations) is used to interpret model predictions and identify the most relevant spatial and temporal pressure patterns.

---

## Repository Structure

```
venice-surge-prediction/
│
├── notebooks/
│   ├── surge.ipynb              # Meteorological surge model
│   ├── paw.ipynb                # PAW surge model
│   │
│   ├── models/
│   │   ├── surge/
│   │   │   ├── xgb_surge_tuned.pkl       # Tuned XGBoost model
│   │   │   └── xgb_surge_search.json     # Hyperparameter search results
│   │   └── paw/
│   │       ├── xgb_paw_tuned.pkl         # Tuned XGBoost model
│   │       ├── xgb_paw_baseline.pkl      # Baseline XGBoost model
│   │       ├── paw_pca.pkl               # PCA transformer (95% variance)
│   │       ├── paw_scaler.pkl            # StandardScaler
│   │       └── xgb_paw_search.json       # Hyperparameter search results
│   │
│   └── plots/
│       ├── surge/                        # Surge model figures
│       └── paw/                          # PAW model figures
│
├── src/
│   ├── preprocessing.py         # Feature construction, PCA, train/test split
│   ├── models.py                # Training, hyperparameter search, evaluation
│   ├── metrics.py               # MAE, RMSE, R² computation and printing
│   └── visualization.py         # All plotting functions
│
├── requirements.txt
└── .gitignore
```

> **Note:** raw data files (`data/*.parquet`) are not included in this repository due to size constraints (~117 MB and ~280 MB). Contact CNR ISMAR for data access.

---

## Models

### Meteorological Surge

| Property | Value |
|---|---|
| Target | Surge at Venice (45.75°N, 12.25°E) |
| Resolution | 3-hourly |
| Domain | Adriatic sub-grid (38–47°N, 7–22°E) |
| Grid points | 133 |
| Features | MSL + 5 daily lags × 133 points = **798 features** |
| Train period | 2019-01-01 → 2024-08-07 (4,674 samples) |
| Test period | 2024-08-07 → 2024-12-31 (1,169 samples) |

**Results:**

| Model | MAE (m) | RMSE (m) | R² |
|---|---|---|---|
| Baseline — Train | 0.00132 | 0.00176 | 0.9985 |
| Baseline — Test | 0.02780 | 0.03689 | 0.3150 |
| Tuned — Train | 0.01349 | 0.01748 | 0.8539 |
| **Tuned — Test** | **0.02712** | **0.03563** | **0.3611** |

Best hyperparameters: `n_estimators=500`, `max_depth=6`, `learning_rate=0.01`, `reg_lambda=10.0`, `subsample=0.6`.

---

### PAW Surge

| Property | Value |
|---|---|
| Target | PAW surge at Venice (45.75°N, 12.25°E) |
| Resolution | Daily |
| Domain | Full European grid (30–59.75°N, −19.75–29.25°E) |
| Grid points | 1,420 |
| Raw features | MSL + 30 daily lags × 1,420 points = **44,020 features** |
| PCA components | 115 (PCA-95, 382:1 compression) |
| Train period | 2019-01-31 → 2024-08-12 (560 samples) |
| Test period | 2024-08-13 → 2024-12-31 (141 samples) |

**Results:**

| Model | MAE (m) | RMSE (m) | R² |
|---|---|---|---|
| Baseline — Train | 0.00028 | 0.00038 | 1.0000 |
| Baseline — Test | 0.06388 | 0.07977 | 0.2868 |
| Tuned — Train | 0.01344 | 0.01796 | 0.9711 |
| **Tuned — Test** | **0.05534** | **0.06635** | **0.5066** |

Best hyperparameters: `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `min_child_weight=20`, `colsample_bytree=0.5`.

---

## SHAP Analysis

SHAP (TreeExplainer) is used to interpret both models:

- **Bar chart** — mean absolute SHAP value per feature/component.
- **Beeswarm plot** — distribution of SHAP values across all test observations.
- **Waterfall plots** — per-observation decomposition for extreme and median events.
- **Spatial importance map** — aggregated SHAP importance back-projected onto the geographic grid.

For the PAW model, SHAP values are computed in PCA space (115 components). The spatial map is obtained via back-projection through the PCA loadings:

```python
feature_importance_original = mean_abs_shap @ abs(pca.components_[:n_sel, :])
```

This operation is valid for spatial aggregates but does not preserve per-observation additivity.

**Key findings — Surge model:** the most important feature is `msl_lag1d` at 44.75°N, 11.25°E (Po Delta), with importance nearly double that of the second feature. Spatial importance is concentrated in the northern Adriatic.

**Key findings — PAW model:** PC5 is the most predictive component despite explaining only 5.0% of variance, ahead of PC3 (7.2%) and PC1 (10.0%). Back-projected spatial importance is nearly uniform across the full European domain (range: 0.021–0.027), consistent with the planetary scale of PAW forcing.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Leoooo22/venice-surge-prediction.git
cd venice-surge-prediction

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data

Pressure and surge data were provided by **CNR ISMAR** (Istituto di Scienze Marine, Consiglio Nazionale delle Ricerche). The dataset covers 2019 - 2024.

Data files are **not included** in this repository due to size constraints (`surge_pressures.parquet` ~117 MB, `paw_pressures.parquet` ~280 MB). 

---

## Thesis

This project is part of a Bachelor's thesis on machine learning approaches for sea level prediction at Venice.
