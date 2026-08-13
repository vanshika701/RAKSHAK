# RAKSHAK — Remaining Work

Detailed breakdown of what's left in each phase, against instructions.md's build guide.
See CLAUDE.md's "Current Status" for the coarse phase-level checklist — this is the
zoomed-in version. Update this as items get done; it's a working note, not a deliverable.

---

## Phase 1 — Environment Setup
- [ ] Set up Google Colab + Google Drive for the heavy hyperparameter-tuning runs (everything
      else in this phase is done and verified)

## Phase 2 — Data Pipeline
- [ ] Write UNSW-NB15's own EDA notebook — only CICIDS2017's (`01_eda.ipynb`) exists so far

## Phase 3 — Feature Engineering
Nothing left — fully complete and verified (derived features, top-25 selection, SMOTE,
stratified split, the row-duplication leak fix).

## Phase 4 — Model Training
- [ ] Move XGBoost + LightGBM training to Colab with `RandomizedSearchCV` hyperparameter
      tuning — current versions (`models/*.joblib`) are local, **untuned** baseline runs, done
      as a quick diagnostic, not the final models
- [ ] Decide whether Random Forest also gets a tuned pass, or stays as-is
- [ ] Run SHAP analysis (`notebooks/02_shap_analysis.ipynb`) — feature importance + force
      plots, specifically to investigate *why* U2R has such weak precision (~0.22-0.24 on both
      RF and XGBoost)
- [ ] Pull the tuned models back down from Colab into local `models/`

## Phase 5 — Ensemble & Cross-Dataset Validation
- [ ] Build the soft-voting ensemble (XGBoost + LightGBM + Random Forest)
- [ ] Evaluate the ensemble on the CICIDS2017 test set
- [ ] Train/evaluate a model on the common 6-feature schema for genuine cross-dataset
      validation — infrastructure already built and verified (`cicids_common.parquet` /
      `unsw_common.parquet`), just needs a model trained on it
- [ ] Document the generalization gap in a notebook (expected to be real and worth reporting,
      not a bug to hide)
- [x] Target metric (weighted F1 > 95%) — already cleared by individual models ahead of schedule
- [ ] (Optional, flagged but not investigated) Check whether any duplicate-feature rows in
      CICIDS2017 carry conflicting labels — a data-quality note worth a line in the report
      either way

## Phase 6 — Live Detection Engine
- [ ] Write `src/capture.py` — Scapy packet capture, flow grouping by (src IP, dst IP, port,
      protocol), feature extraction matching `selected_features.json`
- [ ] Write `src/detector.py` — load ensemble + scaler + selected features, inference loop,
      SQLite logging (timestamp, src IP, label, confidence)
- [ ] Test against real local network traffic

## Phase 7 — Dashboard
- [ ] Write `src/app.py` — Flask routes (`/`, `/api/recent`, `/api/alerts`, `/api/stats`)
- [ ] Build the frontend — connection table, alert panel, summary cards, 5s auto-refresh
- [ ] Test the full pipeline end-to-end: capture → classify → display

## Phase 8 — Documentation & Report
- [ ] `README.md` — setup instructions + architecture diagram
- [ ] Internship report (problem statement, literature review, dataset description,
      methodology, results, cross-dataset generalization analysis, limitations, future scope,
      conclusion, references)
- [ ] Presentation slide deck (10-12 slides)
- [ ] Final code cleanup pass (docstrings, remove debug prints, `flake8 src/`)
- [ ] Final tidy GitHub push
