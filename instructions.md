# Network Intrusion Detection System using ML

## Project Description

This project is an end-to-end **Network Intrusion Detection System (NIDS)** built using Machine Learning, developed as part of a cybersecurity and network automation internship aligned with DRDO's focus on securing defence-grade networks.

The system monitors network traffic in real time, extracts statistical features from each connection, and classifies it as either normal or one of four attack categories — DoS, Probe, R2L, or U2R — using an ensemble of XGBoost, LightGBM, and Random Forest models. Alerts are logged to a SQLite database and displayed on a live Flask dashboard with confidence scores and source IP tracking.

The project trains on two modern, large-scale datasets — **CICIDS2017** (2.8M rows) and **UNSW-NB15** (2.5M rows) — and validates cross-dataset generalisation, making it research-grade. SHAP is used for model explainability, showing analysts exactly which features triggered each alert.

**Domain:** Cybersecurity / Network Automation  
**Tech Stack:** Python, XGBoost, LightGBM, scikit-learn, Scapy, Flask, SQLite, SHAP, pandas, Google Colab  
**Hardware:** Apple M1 Air (local dev) + Google Colab (heavy training)  
**Target Duration:** 6 weeks  

---

## Project Structure

```
ids_project/
├── data/                  # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── notebooks/             # EDA and experiment notebooks
├── src/
│   ├── preprocess.py      # Data cleaning and feature engineering
│   ├── train_model.py     # Model training and evaluation
│   ├── capture.py         # Live packet capture via Scapy
│   ├── detector.py        # End-to-end inference pipeline
│   └── app.py             # Flask dashboard
├── models/                # Saved model files (.joblib)
├── reports/               # Final report and presentation
├── requirements.txt
└── README.md
```

---

## Todo List

### Phase 1 — Environment Setup
- [ ] Install Miniforge (M1-native conda)
- [ ] Create conda environment `ids_project` with Python 3.11
- [ ] Install core packages via conda-forge (pandas, numpy, scikit-learn, matplotlib, seaborn, pyarrow)
- [ ] Install pip packages (xgboost, lightgbm, imbalanced-learn, shap, flask, scapy)
- [ ] Set up Google Colab with Google Drive for heavy training runs
- [ ] Create GitHub repository and push initial structure

### Phase 2 — Data Pipeline (Week 1)
- [ ] Download CICIDS2017 dataset (all CSV files)
- [ ] Download UNSW-NB15 dataset
- [ ] Write chunked CSV loader (`chunksize=100000`) to handle large files
- [ ] Standardise column names across both datasets
- [ ] Handle missing values and infinite values
- [ ] Encode categorical features (protocol, service, flag)
- [ ] Normalise numerical features with MinMaxScaler
- [ ] Save cleaned data as compressed `.parquet` files
- [ ] Write EDA notebook — class distribution, correlation heatmap, feature stats

### Phase 3 — Feature Engineering (Week 2)
- [ ] Run quick Random Forest on 10% sample to get feature importances
- [ ] Select top 20–25 features, drop the rest
- [ ] Engineer derived features (bytes/sec ratio, forward/backward packet ratio, flag anomaly score)
- [ ] Apply SMOTE to minority attack classes
- [ ] Create final train/test split (80/20, stratified)

### Phase 4 — Model Training (Week 3)
- [ ] Train Random Forest baseline — log accuracy, F1, confusion matrix
- [ ] Train XGBoost on full CICIDS2017 via Colab
- [ ] Train LightGBM on full CICIDS2017 via Colab
- [ ] Hyperparameter tune XGBoost and LightGBM with RandomizedSearchCV
- [ ] Run SHAP analysis — generate feature importance and force plots
- [ ] Save all three models with joblib

### Phase 5 — Ensemble & Cross-Dataset Validation (Week 4)
- [ ] Build soft-voting ensemble (XGBoost + LightGBM + Random Forest)
- [ ] Evaluate ensemble on CICIDS2017 test set
- [ ] Cross-dataset validation — train on CICIDS2017, test on UNSW-NB15
- [ ] Document generalisation gap and analysis in notebook
- [ ] Achieve target: overall weighted F1 > 95%

### Phase 6 — Live Detection Engine (Week 5 — Part 1)
- [ ] Write Scapy packet capture module (`capture.py`)
- [ ] Group packets into connections by (src IP, dst IP, port, protocol)
- [ ] Extract top 15–20 features matching training data format
- [ ] Load saved ensemble model and run inference loop
- [ ] Log predictions to SQLite database (timestamp, src IP, label, confidence)
- [ ] Test on local network traffic

### Phase 7 — Dashboard (Week 5 — Part 2)
- [ ] Set up Flask app (`app.py`)
- [ ] Build live connection feed (auto-refresh every 5 seconds)
- [ ] Build alert panel — flagged connections only
- [ ] Add summary stats (total today, attack breakdown, top source IPs)
- [ ] Colour-code table by attack type
- [ ] Test full pipeline end-to-end: capture → classify → display

### Phase 8 — Documentation & Report (Week 6)
- [ ] Write `README.md` with setup instructions and architecture diagram
- [ ] Write internship-style project report (PDF):
  - [ ] Problem statement
  - [ ] Literature review (3–4 IEEE/Springer IDS papers)
  - [ ] Dataset description
  - [ ] Methodology
  - [ ] Results with charts
  - [ ] Limitations and future scope
- [ ] Create presentation slide deck (10–12 slides)
- [ ] Clean and comment all code
- [ ] Final GitHub push — tidy commits, proper `.gitignore`

---

## Key Milestones

| Milestone | Target Date |
|-----------|-------------|
| Environment ready + datasets downloaded | End of Week 1 |
| Clean preprocessed data as parquet | End of Week 1 |
| Trained ensemble model with F1 > 95% | End of Week 4 |
| Cross-dataset validation complete | End of Week 4 |
| Live detector running on local network | End of Week 5 |
| Dashboard live at localhost:5000 | End of Week 5 |
| Report + slides + GitHub finalised | End of Week 6 |

---

## Step-by-Step Build Guide

### Step 1 — Set Up Your Environment

**1.1 Install Miniforge**
```bash
brew install miniforge
conda init zsh
# Restart terminal after this
```

**1.2 Create and activate project environment**
```bash
conda create -n ids_project python=3.11
conda activate ids_project
```

**1.3 Install packages**
```bash
# Always use conda-forge for these on M1
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn jupyter pyarrow

# Use pip for these
pip install xgboost lightgbm imbalanced-learn shap flask scapy
```

**1.4 Create project folder structure**
```bash
mkdir ids_project && cd ids_project
mkdir -p data/raw data/processed notebooks src models reports
touch src/preprocess.py src/train_model.py src/capture.py src/detector.py src/app.py
touch requirements.txt README.md
```

**1.5 Set up GitHub repo**
```bash
git init
echo "data/raw/\n__pycache__/\n*.joblib\n.DS_Store\n.env" > .gitignore
git add .
git commit -m "initial project structure"
# Create repo on GitHub and push
```

---

### Step 2 — Download the Datasets

**CICIDS2017**
- Go to: https://www.unb.ca/cic/datasets/ids-2017.html
- Download all CSV files (Monday through Friday)
- Place them in `data/raw/cicids2017/`

**UNSW-NB15**
- Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`
- Place them in `data/raw/unsw_nb15/`

---

### Step 3 — Preprocess the Data (`src/preprocess.py`)

This script cleans and prepares both datasets for training.

**What it does, in order:**

1. Load CICIDS2017 CSVs in chunks of 100,000 rows at a time
2. Strip whitespace from column names (CICIDS2017 has leading spaces)
3. Replace infinite values with NaN, then drop or fill NaN rows
4. Map the `Label` column to 5 classes: Normal, DoS, Probe, R2L, U2R
5. Encode categorical columns with LabelEncoder
6. Drop constant and near-zero-variance columns
7. Normalise all numerical features with MinMaxScaler
8. Save final cleaned data to `data/processed/cicids_clean.parquet`
9. Repeat steps 1–8 for UNSW-NB15

**Run it:**
```bash
python src/preprocess.py
```

Expected output: two `.parquet` files in `data/processed/`, total size ~300–400MB.

---

### Step 4 — Exploratory Data Analysis (`notebooks/01_eda.ipynb`)

Open Jupyter and create `notebooks/01_eda.ipynb`. Cover these in order:

1. Load the parquet file and inspect shape, dtypes, sample rows
2. Plot class distribution — bar chart of Normal vs each attack type
3. Plot correlation heatmap of top 30 features
4. Box plots of the 5 most important numerical features by class
5. Note any observations about class imbalance — this informs SMOTE strategy

This notebook is a deliverable. Keep it clean and add markdown cells explaining each finding.

---

### Step 5 — Feature Engineering (`src/preprocess.py` — add to existing)

After basic cleaning, add these derived features before saving:

```python
# Bytes per second
df['bytes_per_sec'] = df['total_fwd_packets'] / (df['flow_duration'] + 1e-9)

# Forward to backward packet ratio
df['fwd_bwd_ratio'] = df['total_fwd_packets'] / (df['total_bwd_packets'] + 1e-9)

# Flag anomaly score — count of unusual TCP flag combinations
df['flag_anomaly'] = (df['fin_flag_count'] + df['rst_flag_count']) / (df['syn_flag_count'] + 1e-9)
```

Then run a quick Random Forest on a 10% sample to rank feature importances. Keep top 25 features, drop the rest. Save the feature list to `models/selected_features.json` — you'll need this later in the capture module.

---

### Step 6 — Handle Class Imbalance

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
```

Apply SMOTE only to the training set, never the test set. Check class counts before and after to confirm minority classes are balanced.

---

### Step 7 — Train Models on Google Colab

**7.1 Push your preprocessed parquet to Google Drive**
Upload `data/processed/cicids_clean.parquet` to a folder in your Google Drive.

**7.2 In Colab, mount Drive and load data**
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_parquet('/content/drive/MyDrive/ids_project/cicids_clean.parquet')
```

**7.3 Train XGBoost**
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                    use_label_encoder=False, eval_metric='mlogloss',
                    tree_method='hist', random_state=42)
xgb.fit(X_train, y_train)
```

**7.4 Train LightGBM**
```python
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                      class_weight='balanced', random_state=42)
lgbm.fit(X_train, y_train)
```

**7.5 Train Random Forest baseline**
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                            n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
```

**7.6 Evaluate each model**
```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, xgb.predict(X_test)))
```

**7.7 Save models to Drive**
```python
import joblib
joblib.dump(xgb, '/content/drive/MyDrive/ids_project/models/xgb_model.joblib')
joblib.dump(lgbm, '/content/drive/MyDrive/ids_project/models/lgbm_model.joblib')
joblib.dump(rf, '/content/drive/MyDrive/ids_project/models/rf_model.joblib')
```

Download all three to your local `models/` folder.

---

### Step 8 — SHAP Explainability (`notebooks/02_shap_analysis.ipynb`)

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test[:500])  # sample for speed

# Summary plot — which features matter most overall
shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names)

# Force plot — why was this specific connection flagged?
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])
```

Save these plots as PNGs for your report. These are the most visually impressive outputs of the entire project.

---

### Step 9 — Build the Ensemble (`src/train_model.py`)

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[('xgb', xgb_model), ('lgbm', lgbm_model), ('rf', rf_model)],
    voting='soft'  # uses predicted probabilities, not just votes
)
```

Evaluate on CICIDS2017 test set, then run it against the UNSW-NB15 test set without retraining. Document the accuracy drop — this is your cross-dataset generalisation result.

---

### Step 10 — Live Packet Capture (`src/capture.py`)

This module uses Scapy to capture live packets and extract features.

**What it does:**
1. Sniff packets on your network interface (usually `en0` on Mac)
2. Group packets into flows by (src_ip, dst_ip, dst_port, protocol)
3. After a 5-second window, compute flow-level statistics
4. Return a dictionary matching your 25 selected features
5. Apply the same MinMaxScaler used during training

**Run with sudo** (Scapy needs root for raw packet access):
```bash
sudo python src/capture.py
```

Test it by opening a browser or running a ping — you should see connections being printed to the terminal.

---

### Step 11 — Inference Pipeline (`src/detector.py`)

This ties everything together:

1. Load ensemble model and scaler from `models/`
2. Load selected feature list from `models/selected_features.json`
3. Start capture loop (calls `capture.py` in a thread)
4. For each new flow, extract features → scale → predict
5. If prediction is not Normal, log to SQLite:
   ```
   timestamp | src_ip | dst_ip | label | confidence | top_shap_feature
   ```
6. Print alert to terminal in real time

**Run:**
```bash
sudo python src/detector.py
```

---

### Step 12 — Flask Dashboard (`src/app.py`)

**Routes to build:**
- `GET /` — main dashboard page
- `GET /api/recent` — returns last 50 connections as JSON
- `GET /api/alerts` — returns flagged connections only
- `GET /api/stats` — returns summary counts by label

**Dashboard UI (plain HTML + JS):**
- Table of recent connections, colour-coded by label
- Alert panel on the right showing only attacks
- Summary cards at the top: total today, DoS count, Probe count, other attacks
- Auto-refresh every 5 seconds using `setInterval` + `fetch`

**Run:**
```bash
python src/app.py
# Open http://localhost:5000
```

Note: run `detector.py` in one terminal and `app.py` in another — they share the same SQLite database.

---

### Step 13 — Write the Report (`reports/report.pdf`)

Structure your report exactly like this:

1. **Abstract** — 150 words, summarise the whole project
2. **Introduction** — why IDS matters, especially in defence contexts
3. **Literature Review** — summarise 3–4 papers from IEEE Xplore on ML-based IDS
4. **Dataset Description** — CICIDS2017 and UNSW-NB15, class distributions
5. **Methodology** — preprocessing pipeline, feature engineering, model architecture
6. **Results** — classification report tables, confusion matrices, SHAP plots
7. **Cross-Dataset Validation** — generalisation gap analysis
8. **Limitations** — encrypted traffic, zero-days, feature extraction gaps
9. **Future Scope** — deep learning (LSTM for sequential traffic), deployment on Raspberry Pi
10. **Conclusion**
11. **References**

Target length: 15–20 pages. Use IEEE conference paper format for maximum credibility.

---

### Step 14 — Final Cleanup and Submission

**Code cleanup:**
- Add docstrings to every function
- Add inline comments on non-obvious logic
- Remove all debug print statements
- Run `flake8 src/` to catch style issues

**GitHub:**
```bash
git add .
git commit -m "final submission — all modules complete"
git push origin main
```

Add a `requirements.txt`:
```bash
conda activate ids_project
pip freeze > requirements.txt
```

**Final folder check — make sure you have:**
- [ ] `src/` — all 5 Python files, clean and commented
- [ ] `notebooks/` — EDA notebook + SHAP notebook
- [ ] `models/` — 3 saved model files + selected_features.json
- [ ] `reports/` — PDF report + slide deck
- [ ] `README.md` — setup instructions + architecture diagram
- [ ] `requirements.txt`

---

## Notes for Claude

- Stack: M1 Mac for local dev, Google Colab for training. Always use conda-forge for numpy/pandas/sklearn installs, pip for xgboost/lightgbm.
- Dataset: CICIDS2017 primary, UNSW-NB15 for cross-validation.
- Always use chunked loading for datasets — never load full CSV into memory at once.
- Model explainability via SHAP is a required deliverable, not optional.
- Target overall weighted F1 > 95% on CICIDS2017 test set.
- When writing code, prefer clarity over cleverness — this is a documented internship project.