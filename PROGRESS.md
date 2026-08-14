# RAKSHAK — Remaining Work

Only unfinished items - completed work has been stripped out to keep this file focused and
short. See CLAUDE.md's "Current Status" for the coarse phase-level checklist.

All 8 phases are now complete (slide deck deliberately skipped per project owner's call).

---

## Deferred — revisit later if time allows

**U2R further improvement** (as of 2026-08-14): current state is a defensible, evidence-based
result (F1 0.35 → 0.66 via SHAP investigation + validated threshold tuning + Colab hyperparameter
tuning), not a bug with an obvious fix left on the table. Root cause is only ~1,587 real training
examples - not fixable without more real data. Ideas on record if revisited: Borderline-SMOTE
instead of plain SMOTE, feature selection targeted specifically at the U2R-vs-Normal boundary
(`PSH Flag Count` lead from the SHAP notebook), session/multi-flow correlation features (see the
live `nmap` scan test's finding in `reports/internship_report.md` Section 7).

## (Optional, not blocking) Data-quality note
Check whether any duplicate-feature rows in CICIDS2017 carry conflicting labels — flagged during
Phase 3 but never investigated. Worth a line in the report either way if picked up later.
