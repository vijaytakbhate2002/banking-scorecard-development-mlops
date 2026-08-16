# Credit Risk Scorecard — MLOps

End-to-end credit risk scorecard development pipeline for Lending Club loan data. This project combines data preparation, leakage-safe feature engineering, model training, artifact packaging, and API deployment for a production-style credit risk scorecard.

---

## Repository Structure

```text
credit_risk_scorecard_mlops/
├── app.py                              # FastAPI scoring service
├── requirements.txt                   # Python dependencies
├── model_development/
│   ├── pd_model_training.ipynb        # Full model training pipeline
│   ├── model_artifact_loader.py       # Loader for saved model artifacts
│   ├── target_encoder.py              # Reusable target encoder implementation
│   ├── data_processing/
│   │   ├── data_collection.ipynb      # Download raw Lending Club data
│   │   ├── feature_selection.ipynb    # Target creation, split logic, feature filtering
│   │   └── data/
│   │       ├── raw_data/              # Raw Kaggle files (gitignored)
│   │       └── imp_features/          # Processed feature-selected datasets
│   └── artifacts/                     # Saved trained model bundles
├── test_api_results/
│   └── test_api.py                    # API validation checks
├── README.md
└── .gitignore
```

---

## Data Pipeline Overview

### 1. Data Collection

Notebook: `model_development/data_processing/data_collection.ipynb`

This stage downloads accepted and rejected Lending Club datasets from Kaggle and stores them in `model_development/data_processing/data/raw_data/`.

### 2. Feature Selection and Splitting

Notebook: `model_development/data_processing/feature_selection.ipynb`

The feature selection pipeline uses accepted loans only and creates the binary target `default_status`.

- Good: `0`
- Bad: `1`
- Derived from `loan_status`
- Post-target columns removed: `loan_status`, `id`
- Temporal split by `issue_d` using a chronological train / validation / OOT design

#### Final processed datasets in `imp_features/`

| File             | Purpose                       |
| ---------------- | ----------------------------- |
| `full_df_fi.csv` | Full feature-selected dataset |
| `train_fi.csv`   | 70% training set              |
| `test_fi.csv`    | 15% validation set            |
| `oot_fi.csv`     | 15% out-of-time holdout set   |

---

## Major Model Development Fixes

### 1. Leakage cleanup

The first major issue was a target leakage problem caused by post-disbursal information leaking into the training data. After ranking features by SHAP and coefficient gain, several suspicious post-disbursal variables were identified and removed from the training set.

This corrected the feature set and produced a more trustworthy model signal. The original leakage-heavy version had an inflated, unrealistic AUC/Gini pattern before the cleanup.

### 2. Target encoder block

After removing the leakage-causing variables, the model Gini dropped sharply, indicating the remaining feature set was still weak for a direct logistic scorecard. To fix that, a target encoding block was added using:

- mean
- std
- count
- skew

This was applied to the categorical features before the WoE transformation stage. The encoder produces encoded categorical features that help preserve signal without leaking target information into the validation/test pipeline.

### 3. AUC validation fix

The original pytest logic used the first 200 rows of the validation dataset, which did not represent the overall behaviour of the real validation population. That created unstable and overly pessimistic AUC checks. The final validation check uses a representative sampled validation set (1000 rows) for AUC testing.

### 4. PCA visualization

A 3D PCA view was added to visualize default and non-default observations across the transformed feature space. This helps confirm that the scorecard features carry separable structure and supports model explainability.

---

## Model Development Notebook

Notebook: `model_development/pd_model_training.ipynb`

This notebook implements the full production-style pipeline:

1. Data read from `imp_features/`
2. Date preprocessing and missing-value imputation
3. Target leakage cleanup
4. Categorical target encoding using the custom encoder
5. WoE binning with `optbinning`
6. Optuna search for logistic regression hyperparameters
7. Final model training
8. SHAP + coefficient gain ranking
9. Top-5 feature selection
10. AUC / KS analysis on train, validation, and OOT splits
11. PCA visualization of defaulters vs non-defaulters
12. Artifact export

---

## Current Model Metrics

The final trained model artifacts were saved under `model_development/artifacts/` and evaluated on the real validation and OOT splits.

| Split      | AUC    | Gini   | KS     |
| ---------- | ------ | ------ | ------ |
| Train      | 0.6909 | 0.3817 | 0.2769 |
| Validation | 0.7109 | 0.4218 | 0.3126 |
| OOT        | 0.7012 | 0.4024 | 0.3046 |

These values are the current benchmark from the notebook artifact export and reflect the leakage-safe, target-encoded model configuration.

> The model is not tuned to chase a near-perfect benchmark; the objective was to build a trustworthy, explainable scorecard with realistic, stable out-of-sample performance.

---

## Artifact Bundle

Each trained run is saved under a timestamped folder inside `model_development/artifacts/`.

Files typically include:

- `model_top5.joblib` — final trained logistic regression model
- `target_encoder.joblib` — fitted target encoder used for categorical encoding
- `impute_values.joblib` — train-fit imputation values
- `top5_features.json` — selected top-5 feature list
- `woe_features.json` — accepted WoE features for scoring
- `metrics.json` — train / validation / OOT model performance
- `metadata.json` — experiment metadata
- `binners/` — persisted WoE binning objects

---

## API / Prediction Service

The FastAPI app in `app.py` loads the latest artifact directory at startup and applies the same pipeline to new raw records:

- date preprocessing
- imputation using saved training statistics
- target encoding using the saved encoder
- WoE transformation using the saved binners
- probability prediction using the trained scorecard model

Run the API with:

```bash
uvicorn app:app --reload
```

Endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`

---

## Test Coverage

The repository includes API validation checks in `test_api_results/test_api.py` to confirm:

- app health is ok
- metadata loads correctly
- model artifacts load successfully
- predictions return valid probabilities and scores
- AUC validation matches the real validation distribution

---

## Setup

### Prerequisites

- Python 3.x
- `pip`
- Access to Lending Club data source

### Install dependencies

```bash
pip install -r requirements.txt
```

This project also depends on libraries such as `optbinning`, `shap`, `optuna`, and `fastapi` as part of the training and serving flow.

---

## Current Status

| Phase                               | Status      |
| ----------------------------------- | ----------- |
| Data collection                     | ✅ Complete |
| Feature selection                   | ✅ Complete |
| Leakage cleanup                     | ✅ Complete |
| Target encoder feature engineering  | ✅ Complete |
| Model training and evaluation       | ✅ Complete |
| PCA visualization                   | ✅ Complete |
| Model artifact saving               | ✅ Complete |
| API deployment / prediction service | ✅ Complete |
| Automated validation checks         | ✅ Complete |
