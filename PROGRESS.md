# RAKSHAK — Remaining Work

Detailed breakdown of what's left in each phase, against instructions.md's build guide.
See CLAUDE.md's "Current Status" for the coarse phase-level checklist — this is the
zoomed-in version. Update this as items get done; it's a working note, not a deliverable.

---

## Deferred — revisit later if time allows

**U2R further improvement** (as of 2026-08-14): deliberately parking this here, not abandoning
it. Current state is a defensible, evidence-based result (F1 0.35 → 0.65 via SHAP investigation
+ validated threshold tuning — see Phase 5), not a bug with an obvious fix left on the table.
Decided to prioritize Phase 6-8 (live capture, dashboard, report) instead, since those are
100% unstarted and load-bearing for the project being a finished, deliverable system, whereas
further U2R work has real diminishing returns (root cause is only ~1,587 real training examples
- not fixable without more real data). Nothing is locked in by deferring this: the classifier
sits behind a stable `predict()`/`predict_proba()` interface, so swapping in an improved version
later won't require touching `detector.py` or anything downstream. Ideas already on record if
revisited: Borderline-SMOTE instead of plain SMOTE, feature selection targeted specifically at
the U2R-vs-Normal boundary (see `PSH Flag Count` lead from the SHAP notebook).

Suggested order from here: finish Colab tuning + cross-dataset validation (Phase 4/5 loose
ends, already in flight) → Phase 6 → Phase 7 → Phase 8 → circle back to U2R only if time remains.

---

## Phase 1 — Environment Setup
- [ ] Set up Google Colab + Google Drive for the heavy hyperparameter-tuning runs (everything
      else in this phase is done and verified)

## Phase 2 — Data Pipeline
- [x] Write UNSW-NB15's own EDA notebook (`notebooks/04_unsw_eda.ipynb`) — mirrors `01_eda.ipynb`'s
      structure. Key finding: UNSW-NB15's class balance is very different from CICIDS2017's -
      Normal is only 55.7% here (vs. 80.3%), U2R is proportionally ~15x more common (1.05% vs
      0.07%), and R2L isn't a minority class at all (17.1% vs 0.57%). Also confirmed UNSW-NB15's
      top correlated features (`dttl`, `ct_*` rolling-connection-counts) are a completely
      different family than CICIDS2017's packet-size/timing stats - expected, given the two
      datasets use different capture tools (Argus vs CICFlowMeter). Both points matter directly
      for Phase 5's cross-dataset validation - a real evaluation, not an apples-to-apples one.

## Phase 3 — Feature Engineering
Nothing left — fully complete and verified (derived features, top-25 selection, SMOTE,
stratified split, the row-duplication leak fix).

## Phase 4 — Model Training
- [x] Train all three models locally as untuned baselines (RF, XGBoost, LightGBM) — all saved
      to `models/*.joblib`, all clear weighted F1 > 95% (0.9972 / 0.9964 / 0.9970). U2R is weak
      on all three (best: LightGBM 0.3067 precision); R2L is weakest on XGBoost (0.7960
      precision) and strongest on RF (0.9795) — different models have different weak points,
      which is a good sign for the ensemble.
- [ ] Move XGBoost + LightGBM training to Colab with `RandomizedSearchCV` hyperparameter
      tuning — current versions are still the untuned diagnostic runs, not the final models
- [ ] Decide whether Random Forest also gets a tuned pass, or stays as-is
- [x] Run SHAP analysis (`notebooks/02_shap_analysis.ipynb`) — compared real U2R attacks vs.
      Normal flows misclassified as U2R (targeted sample, not a random one — false positives are
      only ~0.17% of the test set). Finding: nearly the same features (packet-size stats,
      Destination Port) drive both groups, and both look like small/sparse low-volume flows —
      evidence the model is using genuine, if ambiguous, signal rather than reacting to noise.
      `PSH Flag Count` stood out as present in true positives but not false positives - a
      possible lead for a more discriminating feature. Ties back to Phase 3: SMOTE likely
      amplified this narrow "small flow" profile since the real 1,587 U2R examples were already
      clustered around it.
- [ ] Pull the tuned models back down from Colab into local `models/`
- [x] Found and fixed a real bug along the way: `train_model.py`'s custom classes
      (`LabelDecodingClassifier` etc.) broke when loaded from anywhere other than another direct
      `python src/train_model.py` run, because Python tags classes defined in a directly-executed
      script under the module `__main__`. Fixed by moving the pipeline into `main()` and adding
      `src/run_training.py` as the real entry point (imports `train_model` instead of executing
      it) - `python src/train_model.py` now refuses to run directly with an explanatory error.
      CLAUDE.md's Key Commands updated accordingly.

## Phase 5 — Ensemble & Cross-Dataset Validation
- [x] Build the soft-voting ensemble (XGBoost + LightGBM + Random Forest) — custom
      `SoftVotingEnsemble` class (equal-weight `predict_proba()` averaging), not sklearn's
      `VotingClassifier` (would have re-fit all three models from scratch)
- [x] Evaluate the ensemble on the CICIDS2017 test set — best of all 4 (weighted F1 0.9977,
      macro F1 0.8835). R2L recovered to near-RF-best (0.9619 precision) as hoped. U2R improved
      over RF/XGBoost but didn't quite reach LightGBM alone (0.2941 vs 0.3067 precision) -
      equal-weight averaging dilutes LightGBM's edge on that one class. Possible future
      refinement: weighted voting instead of equal weights - not pursued yet.
- [x] Reduced U2R's false-positive rate via a validation-tuned decision threshold, not
      retraining. Built `split_train_val_test()` (train/val/test three-way split - X_train
      identical to before), `SoftVotingEnsemble` weights support, and `ThresholdedEnsemble`
      (requires extra confidence before predicting U2R, falls back to next-most-likely class
      otherwise - doesn't touch how any other class is decided). Weighted voting was tried first
      and rejected: best case only lifted U2R F1 by ~3% relative while measurably hurting R2L
      (0.9817 → 0.9706 as weight shifted toward LightGBM) - not a good trade. Threshold tuning
      won clearly instead: swept 0.2-0.99 on validation only (never test, to avoid a quieter
      repeat of the duplicate-row leak fixed earlier); U2R F1 and macro F1 both peak at
      **threshold = 0.80** (`U2R_DECISION_THRESHOLD` in train_model.py) and decline past it.
      Confirmed once on the untouched test set: U2R F1 0.4457 → **0.6526**, false U2R alarms
      458 → **122** (-73%), weighted F1 0.9977 → **0.9985**, macro F1 0.8795 → **0.9245** - the
      overall system improved, not just U2R, confirming this isn't a zero-sum trade. Test-set
      numbers closely matched what validation predicted (0.5596/0.7828 vs 0.5692/0.7437
      precision/recall), confirming the threshold genuinely generalizes rather than overfitting
      to validation noise. Real cost, worth stating plainly: recall dropped from 93% to 78%, so
      some real U2R attacks now go undetected in exchange for far fewer false alarms - a
      genuine security trade-off, not a pure win, and worth presenting in the report as a
      tunable parameter with documented trade-offs rather than an unambiguous improvement.
      Merged into `main()` as an additional final evaluation step, kept alongside the plain
      ensemble rather than replacing it so the improvement stays visible.
- [ ] Train/evaluate a model on the common 6-feature schema for genuine cross-dataset
      validation — infrastructure already built and verified (`cicids_common.parquet` /
      `unsw_common.parquet`), just needs a model trained on it
- [ ] Document the generalization gap in a notebook (expected to be real and worth reporting,
      not a bug to hide)
- [x] Target metric (weighted F1 > 95%) — cleared by every individual model AND the ensemble
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
