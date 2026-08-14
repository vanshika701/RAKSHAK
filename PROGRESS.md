# RAKSHAK — Remaining Work

Only unfinished items - completed work has been stripped out to keep this file focused and
short. See CLAUDE.md's "Current Status" for the coarse phase-level checklist.

---

## Deferred — revisit later if time allows

**U2R further improvement** (as of 2026-08-14): current state is a defensible, evidence-based
result (F1 0.35 → 0.65 via SHAP investigation + validated threshold tuning), not a bug with an
obvious fix left on the table. Root cause is only ~1,587 real training examples - not fixable
without more real data. Ideas on record if revisited: Borderline-SMOTE instead of plain SMOTE,
feature selection targeted specifically at the U2R-vs-Normal boundary (`PSH Flag Count` lead from
the SHAP notebook).

---

## Phase 4 — Model Training (in progress)
- [ ] Colab hyperparameter tuning (`notebooks/03_colab_hyperparameter_tuning.ipynb`) - currently
      running (on a fresh Google account, after the original one hit free-tier usage limits).
      The notebook already has both required fixes baked in: Section 4 loads the canonical
      `selected_features.json` instead of recomputing it (avoids the cross-environment
      sklearn-version feature-selection mismatch found earlier), and the "9b" cell after Section
      9 has the manually-regularized LightGBM candidates (avoids the SMOTE-before-CV leakage that
      broke the naive `RandomizedSearchCV` result). Just needs to finish running top to bottom.
- [ ] Decide whether Random Forest also gets a Colab-tuned pass, or stays as-is
- [ ] Once Colab finishes: download `xgb_model_raw.joblib`, `xgb_label_encoder.joblib`,
      `lgbm_model.joblib` into `models/tuned_from_colab/`, run
      `python src/reassemble_tuned_models.py`, then `python src/rebuild_ensemble.py` to rebuild
      the ensemble and re-sweep the U2R threshold (current `U2R_DECISION_THRESHOLD=0.80` was
      tuned for the untuned models and isn't assumed to still be optimal)
- [ ] After the threshold re-sweep: one-time final confirmation on the untouched test set

## Phase 5 — Ensemble & Cross-Dataset Validation
- [ ] (Optional) Check whether any duplicate-feature rows in CICIDS2017 carry conflicting labels
      — a data-quality note worth a line in the report either way

## Phase 7 — Dashboard
- [x] `src/app.py` and `src/templates/dashboard.html` written (Flask routes `/`, `/api/recent`,
      `/api/alerts`, `/api/stats`; connection table, alerts panel, summary cards, 5s auto-refresh)
- [ ] Confirm end-to-end in the browser: `detector.py` running + `app.py` running + dashboard
      actually showing live data - not yet confirmed working (last attempt hit a conda-env issue,
      since fixed, but success hasn't been confirmed back)

## Phase 8 — Documentation & Report
- [ ] `README.md` — setup instructions + architecture diagram
- [ ] Internship report (problem statement, literature review, dataset description, methodology,
      results, cross-dataset generalization analysis, limitations, future scope, conclusion,
      references)
- [ ] Presentation slide deck (10-12 slides)
- [ ] Final code cleanup pass (docstrings, remove debug prints, `flake8 src/`)
- [ ] Final tidy GitHub push
