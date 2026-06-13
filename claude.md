# RAKSHAK — ML-based Network Intrusion Detection System

## How to Work With Me

**Teaching mode is ON.**
- I will do everything by myself, you just have to guide me thru and provide me code
- Never write code for me directly — always explain what we're about to build,
  why it works, and what concepts I need to understand first
- Before each step, tell me what I need to learn to complete it
  (e.g. "before we write this, you should understand X, Y, Z")
- this project is supposed to be a full fledged trained ml model with high accuracy, keep it in
- After explaining, ask if I'm ready to proceed or want to go deeper
- Write code only after I confirm I understand the concept
- When you do write code, walk through it line by line — don't just dump it
- Point out where this code reflects real-world/industry practice
- At the end of each step, tell me:
  1. What I just learned
  2. What's coming next
  3. What I should read or watch to go deeper (papers, docs, videos)
- keep in mind i am a complete newbie
- if u need anything needs to be changed in order to create a new projecr, then let me know

**Code quality is non-negotiable.**
Even though I'm learning, the code we produce must be expert-grade —
properly structured, documented, and production-aware. No shortcuts.
Teach me to do it right the first time.

## Project Context
This is a 6-week internship-grade IDS project for a DRDO cybersecurity portfolio.
It detects and classifies network intrusions (DoS, Probe, R2L, U2R) using an ensemble
of XGBoost, LightGBM, and Random Forest models trained on CICIDS2017 and UNSW-NB15.
See `instructions.md` for the full step-by-step build guide.

---

## Hardware & Environment Rules
- Machine: Apple M1 Air, 8GB RAM
- Environment: conda env named `ids_project`, Python 3.11
- ALWAYS install numpy, pandas, scikit-learn via `conda install -c conda-forge`
- ALWAYS install xgboost, lightgbm, imbalanced-learn, shap, flask, scapy via `pip`
- NEVER mix — installing sklearn via pip on M1 breaks Apple Accelerate optimisations
- Heavy training runs go on Google Colab, not local machine

## Data Rules
- NEVER load a full CSV into memory — always use `pd.read_csv(..., chunksize=100000)`
- Processed data lives in `data/processed/` as `.parquet` files (not CSV)
- Primary dataset: CICIDS2017 → `data/raw/cicids2017/`
- Secondary dataset: UNSW-NB15 → `data/raw/unsw_nb15/`
- Selected features list is saved at `models/selected_features.json` — always load from there, never hardcode feature names

## Model Rules
- Three models: XGBoost (primary), LightGBM, Random Forest
- Final prediction uses soft-voting ensemble of all three
- Models saved as `.joblib` in `models/`
- Scaler saved as `models/scaler.joblib` — must be applied to all live traffic before inference
- Target: weighted F1 > 95% on CICIDS2017 test set

## Code Style
- Prefer clarity over cleverness — this is a documented internship project
- Every function must have a docstring
- No magic numbers — define constants at the top of each file
- All file paths defined relative to project root using `pathlib.Path`

## Project Structure
```
ids_project/
├── data/
│   ├── raw/
│   │   ├── cicids2017/
│   │   └── unsw_nb15/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_shap_analysis.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── capture.py
│   ├── detector.py
│   └── app.py
├── models/
│   ├── xgb_model.joblib
│   ├── lgbm_model.joblib
│   ├── rf_model.joblib
│   ├── scaler.joblib
│   └── selected_features.json
├── reports/
├── CLAUDE.md
├── instructions.md
├── requirements.txt
└── README.md
```

## Current Status
- [ ] Phase 1 — Environment setup
- [ ] Phase 2 — Data pipeline
- [ ] Phase 3 — Feature engineering
- [ ] Phase 4 — Model training
- [ ] Phase 5 — Ensemble + cross-dataset validation
- [ ] Phase 6 — Live capture engine
- [ ] Phase 7 — Flask dashboard
- [ ] Phase 8 — Report + documentation

Update this checklist as phases are completed.

## Key Commands
```bash
# Activate environment
conda activate ids_project

# Run preprocessing
python src/preprocess.py

# Run live detector (needs sudo for raw packet access)
sudo python src/detector.py

# Run dashboard (separate terminal)
python src/app.py

# Run tests
pytest tests/
```