# RAKSHAK — ML-Based Network Intrusion Detection System

RAKSHAK detects and classifies network intrusions (DoS, Probe, R2L, U2R) in real time, using a
soft-voting ensemble of XGBoost, LightGBM, and Random Forest trained on CICIDS2017 and
cross-validated against UNSW-NB15. Built as a 6-week internship project.

**Final result** (untouched test set): weighted F1 **0.9985**, macro F1 **0.9259**. Full
per-class breakdown, methodology, and honest discussion of limitations (U2R's precision/recall
trade-off, the cross-dataset generalization gap) are in `reports/` — this README covers setup
and usage only.

## Architecture

```
CICIDS2017 / UNSW-NB15 CSVs
          |
          v
  src/preprocess.py            cleans, engineers features, saves parquet
          |
          v
  src/train_model.py           feature selection, scaling, SMOTE, trains
  (+ Colab hyperparameter          RF / XGBoost / LightGBM, builds the
    tuning notebook)                soft-voting ensemble
          |
          v
    models/*.joblib            scaler, selected features, trained models
          |
          v
  src/capture.py    ---->      sniffs live traffic, reconstructs flows,
  (Scapy, needs sudo)           extracts the same 25 features
          |
          v
  src/detector.py   ---->      loads the ensemble, classifies each
  (needs sudo)                  finished flow, logs to SQLite
          |
          v
    detections.db              timestamp, src/dst IP+port, protocol,
                                label, confidence
          |
          v
  src/app.py        ---->      Flask dashboard reading detections.db,
  (no sudo needed)              auto-refreshes every 5s in the browser
```

## Setup

**Hardware note**: developed on Apple Silicon (M1, 8GB RAM). `numpy`/`pandas`/`scikit-learn`
must go through `conda-forge`, not `pip` — pip's generic build loses Apple's Accelerate-optimized
BLAS backend, which matters a lot on 8GB of RAM.

```bash
# 1. Create and activate the environment
conda create -n ids_project python=3.11
conda activate ids_project

# 2. Core scientific packages via conda-forge (NOT pip - see hardware note above)
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn jupyter

# 3. Everything else via pip
pip install -r requirements.txt
```

**Datasets** (not included in the repo — see `instructions.md` for download links):
- CICIDS2017 CSVs → `data/raw/cicids2017/`
- UNSW-NB15 CSVs → `data/raw/unsw_nb15/`

## Usage

```bash
# Clean the raw CSVs into processed parquet files
python src/preprocess.py

# Train all three models + build the ensemble
# (run this, not train_model.py directly - see run_training.py's docstring)
python src/run_training.py

# Live detection - needs sudo for raw packet capture
sudo python src/detector.py

# Dashboard - separate terminal, no sudo needed
python src/app.py
# then open http://127.0.0.1:5000
```

Then, either browse normally to watch ordinary traffic get classified as `Normal`, or run a
self-scan against your own router (`sudo nmap -sS <router-ip>`) to see how the dashboard responds
to something unusual - give it 2-3 minutes after the scan for the flow timeout to elapse.

### Optional: reproducing the Colab-tuned models

XGBoost and LightGBM were separately hyperparameter-tuned on Google Colab
(`notebooks/03_colab_hyperparameter_tuning.ipynb`, per the hardware note above - `RandomizedSearchCV`
on 2M+ rows isn't practical on 8GB of local RAM). To reassemble tuned models downloaded from
Colab into `models/`:

```bash
python src/reassemble_tuned_models.py   # wraps + swaps in the tuned XGBoost/LightGBM
python src/rebuild_ensemble.py          # rebuilds the ensemble, re-sweeps the U2R threshold
python src/final_evaluation.py          # one-time confirmation on the test set
```

## Project Structure

```
RAKSHAK/
├── data/{raw,processed}/       # datasets (gitignored except structure)
├── models/                     # trained models, scaler, selected features (gitignored)
├── notebooks/                  # EDA, SHAP analysis, Colab tuning, cross-dataset validation
├── reports/                    # internship report
├── src/
│   ├── preprocess.py           # Phase 2-3: cleaning, feature engineering
│   ├── train_model.py          # Phase 4-5: training, ensemble, threshold tuning
│   ├── run_training.py         # entry point for train_model.py (see its docstring)
│   ├── reassemble_tuned_models.py / rebuild_ensemble.py / final_evaluation.py
│   │                           # bring Colab-tuned models back into models/
│   ├── capture.py              # Phase 6: packet capture, flow reconstruction
│   ├── detector.py             # Phase 6: live classification + SQLite logging
│   ├── app.py                  # Phase 7: Flask dashboard
│   └── templates/dashboard.html
├── CLAUDE.md                   # project rules used during development
├── PROGRESS.md                 # working notes on what's left / was found along the way
└── instructions.md             # the original step-by-step build guide
```

## Known Limitations

- **U2R detection** trades recall for precision (56.8% precision / 78.3% recall at the tuned
  decision threshold) — a direct consequence of only ~1,587 real U2R training examples in
  CICIDS2017, not a fixable bug. See `reports/` for the full investigation (SHAP analysis,
  threshold sweep, why more tuning wouldn't close the gap).
- **Cross-dataset generalization is weak** without adaptation: a model trained on CICIDS2017 and
  evaluated zero-shot on UNSW-NB15 collapses from 97.7% to 39.9% weighted F1 (see
  `notebooks/05_cross_dataset_validation.ipynb`) — traced to a genuine distribution shift in the
  underlying feature values between the two datasets' capture tools, not a modeling mistake.
- **Live flow reconstruction** approximates a few CICFlowMeter conventions it doesn't exactly
  replicate (documented in `capture.py`'s docstrings) - `Subflow Fwd/Bwd Bytes` and standard
  deviation's `ddof` convention specifically.
