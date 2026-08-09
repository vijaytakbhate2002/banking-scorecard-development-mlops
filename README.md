# Credit Risk Scorecard — MLOps

End-to-end credit risk scorecard development pipeline for Lending Club loan data. This repository documents each stage of model development, from raw data ingestion through deployment-ready artifacts.

---

## Repository Structure

```
credit_risk_scorecard_mlops/
├── model_development/
│   └── data_processing/
│       ├── data_collection.ipynb      # Download and store raw Lending Club data
│       ├── feature_selection.ipynb    # Target definition, splits, and feature filtering
│       └── data/
│           ├── raw_data/              # Downloaded Kaggle dataset (gitignored)
│           └── imp_features/          # Processed feature-selected datasets (gitignored)
├── requirements.txt
└── README.md
```

---

## Phase 1: Data Collection & Feature Selection

### 1. Data Collection

**Notebook:** `model_development/data_processing/data_collection.ipynb`

| Item | Detail |
|------|--------|
| **Source** | [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (`wordsforthewise/lending-club`) via `kagglehub` |
| **Output path** | `model_development/data_processing/data/raw_data/` |

**Files downloaded:**

| File | Description |
|------|-------------|
| `accepted_2007_to_2018q4.csv` | Accepted loan applications (2007–2018 Q4) |
| `accepted_2007_to_2018Q4.csv.gz` | Compressed accepted loans |
| `rejected_2007_to_2018q4.csv` | Rejected loan applications (2007–2018 Q4) |
| `rejected_2007_to_2018Q4.csv.gz` | Compressed rejected loans |

The notebook downloads the dataset to a temporary Kaggle cache path and copies all files into the local `data/raw_data/` directory.

---

### 2. Feature Selection

**Notebook:** `model_development/data_processing/feature_selection.ipynb`

#### Dataset Overview

| Dataset | Rows | Columns |
|---------|------|---------|
| Accepted loans (`acc_df`) | 2,260,701 | 151 |
| Rejected loans (`rej_df`) | 27,648,741 | 9 |
| **Combined** | **29,909,442** | — |

Feature selection is performed on **accepted loans only**.

#### Target Variable

Binary default flag derived from `loan_status`:

| Label | `default_status` | Statuses |
|-------|------------------|----------|
| Good | `0` | Fully Paid; Does not meet the credit policy. Status:Fully Paid |
| Bad | `1` | Charged Off; Default; Does not meet the credit policy. Status:Charged Off |

Columns dropped after target creation: `loan_status`, `id`.

**Overall bad rate:** ~52.28%

#### Temporal Train / Validation / OOT Split

Records are sorted by `issue_d` (ascending) and split chronologically to avoid look-ahead bias:

| Split | Share | Rows | Purpose |
|-------|-------|------|---------|
| **Train** | 70% | 1,582,467 | Model training |
| **Validation (Test)** | 15% | 339,100 | Risk segment calculation |
| **OOT (Out-of-Time)** | 15% | 339,101 | Final holdout evaluation |

Rows with null `issue_d` are removed before splitting (33 rows dropped).

#### Feature Selection Pipeline

Features are filtered in four sequential stages:

| Stage | Method | Threshold | Features In | Features Out | Dropped |
|-------|--------|-----------|-------------|--------------|---------|
| Initial | — | — | 149 | 149 | 0 |
| 1. Fill Rate | Drop columns below fill-rate threshold | `< 5%` | 149 | 115 | 34 |
| 2. Near-Zero Variance | Drop constant or near-constant columns | Top frequency ≥ 99% | 115 | 109 | 6 |
| 3. Information Value (IV) | Optimal binning + WoE/IV via `optbinning` | IV ≥ 0.2 | 109 | 36 | 73 |

**Final survival rate:** 24.16% of original features (36 of 149).

**Near-zero variance features dropped:** `policy_code`, `pymnt_plan`, `hardship_flag`, `delinq_amnt`, `acc_now_delinq`, `chargeoff_within_12_mths`

**Top features by IV (selected):**

| Feature | IV |
|---------|-----|
| `next_pymnt_d` | 11.56 |
| `last_pymnt_amnt` | 8.26 |
| `emp_title` | 6.19 |
| `last_pymnt_d` | 5.88 |
| `total_rec_prncp` | 5.26 |
| `issue_d` | 3.08 |
| `last_credit_pull_d` | 2.41 |
| `total_pymnt` | 2.25 |
| `total_pymnt_inv` | 2.21 |
| `inq_last_12m` | 1.77 |

IV computation uses a stratified sample of up to **100,000** rows from the full accepted dataset.

#### Output Artifacts

Saved to `model_development/data_processing/data/imp_features/`:

| File | Description |
|------|-------------|
| `full_df_fi.csv` | Full accepted dataset with 36 selected features + target |
| `train_fi.csv` | Training split (70%) |
| `test_fi.csv` | Validation split (15%) |
| `oot_fi.csv` | Out-of-time holdout split (15%) |

---

## Setup

### Prerequisites

- Python 3.x
- Kaggle account (for `kagglehub` dataset download)

### Install Dependencies

```bash
pip install -r requirements.txt
pip install optbinning
```

> **Note:** `optbinning` is used in feature selection but is not yet listed in `requirements.txt`.

### Run Notebooks

Execute notebooks in order from the `model_development/data_processing/` directory:

1. `data_collection.ipynb` — downloads and copies raw data
2. `feature_selection.ipynb` — builds target, splits data, and exports feature-selected datasets

---

## Upcoming Phases

<!-- Add future pipeline stages below as they are completed -->

| Phase | Status |
|-------|--------|
| Data Collection & Feature Selection | ✅ Complete |
| Model Development | 🔲 Pending |
| Scorecard Scaling & Calibration | 🔲 Pending |
| MLOps / Deployment | 🔲 Pending |
