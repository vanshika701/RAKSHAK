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

## Phase 4/5 — Model Training + Ensemble: DONE
Colab-tuned XGBoost + LightGBM reassembled into `models/`, ensemble rebuilt, U2R threshold
re-swept on validation (confirmed 0.80 still optimal - same value as before), one-time final
test-set check run via `src/final_evaluation.py`. Final result: weighted F1 **0.9985**, macro F1
**0.9259**, U2R F1 **0.6582** (precision 0.5678, recall 0.7828), R2L F1 0.9774 - a real, if
modest, improvement over the old untuned ensemble's final test result (U2R F1 0.6526, macro F1
0.9245), and test numbers closely tracked what the validation sweep predicted, confirming the
threshold genuinely generalizes rather than overfitting to validation noise. Random Forest was
left untuned (diminishing-returns call, not revisited).

## Phase 5 — Ensemble & Cross-Dataset Validation
- [ ] (Optional) Check whether any duplicate-feature rows in CICIDS2017 carry conflicting labels
      — a data-quality note worth a line in the report either way

## Phase 8 — Documentation & Report
- [ ] `README.md` — setup instructions + architecture diagram
- [ ] Internship report (problem statement, literature review, dataset description, methodology,
      results, cross-dataset generalization analysis, limitations, future scope, conclusion,
      references)
- [ ] Final code cleanup pass (docstrings, remove debug prints, `flake8 src/`)
- [ ] Final tidy GitHub push
