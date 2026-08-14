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
- [x] Train/evaluate a model on the common 6-feature schema for genuine cross-dataset
      validation, and document the generalization gap (`notebooks/05_cross_dataset_validation.ipynb`).
      Single Random Forest (`class_weight="balanced"`, no SMOTE - deliberately simpler than the
      main pipeline), trained once on CICIDS2017's common-schema split, evaluated twice without
      retraining: in-distribution on CICIDS2017's own test split (weighted F1 0.9770, macro F1
      0.6922) vs. zero-shot on the entirety of UNSW-NB15 (weighted F1 **0.3988**, macro F1
      **0.1431** - the model predicts "Normal" for essentially every row). Traced the collapse to
      a real, measurable cause rather than leaving it unexplained: UNSW-NB15's flows are
      systematically larger across every shared feature (median duration ~7.6x longer, forward
      bytes ~15x more, throughput ~6.6x higher) - likely differing CICFlowMeter vs. Argus
      capture/aggregation conventions - so Random Forest's hard numeric thresholds, learned from
      CICIDS2017's value ranges, collapse almost all UNSW-NB15 rows into the same leaf. Legitimate,
      citable finding for the report: a shared feature name/definition doesn't guarantee a shared
      feature distribution - genuine transfer would need per-dataset normalization or a
      domain-adaptation approach, not just matching column names.
- [x] Target metric (weighted F1 > 95%) — cleared by every individual model AND the ensemble
- [ ] (Optional, flagged but not investigated) Check whether any duplicate-feature rows in
      CICIDS2017 carry conflicting labels — a data-quality note worth a line in the report
      either way

## Phase 6 — Live Detection Engine
- [x] Write `src/capture.py` — Scapy packet capture, flow grouping by a direction-independent
      5-tuple key, feature extraction matching `selected_features.json`'s 25 columns. Built and
      tested against real local traffic (DNS, HTTPS, QUIC-over-UDP). Found and fixed two real
      bugs via live testing: (1) packet length included the Ethernet frame header, inflating
      every length feature by 14 bytes versus CICFlowMeter's IP-packet-only convention - fixed
      to use the IP layer's length instead; (2) forward/backward direction was decided eagerly
      from whichever packet was captured first, which silently reversed `Destination Port` and
      every Fwd/Bwd feature for any TCP connection that was already open before `capture.py`
      started sniffing (confirmed live: a `20.207.73.82:443` flow got labeled backwards, with
      `Destination Port` coming out as the local machine's ephemeral port instead of 443) - fixed
      by deferring direction to flow-close time and using a bare TCP SYN (flag `S` without `A`)
      as authoritative proof of who initiated the connection, falling back to first-packet-seen
      only when no SYN was ever observed (UDP, or a connection that predates the capture start).
      Two documented approximations, not silently exact: `Subflow Fwd/Bwd Bytes` approximated as
      total fwd/bwd bytes (real CICFlowMeter sub-flow splitting not replicated); standard
      deviation uses population (`ddof=0`) not sample (`ddof=1`) formula, since CICFlowMeter's own
      convention isn't recoverable from the training CSVs and population std gives single-packet
      flows a well-defined 0.0 instead of NaN.
- [x] Write `src/detector.py` — loads the ensemble via `train_model.py`'s own saved artifacts
      (no duplicated logic), reuses `capture.py`'s `FlowManager`/`extract_features()`, logs every
      classification to SQLite (`detections.db` - timestamp, src/dst IP+port, protocol, label,
      confidence), prints `[ALERT]` only for non-Normal predictions. Confidence is looked up at
      the *actually predicted* label's index, not `max(predict_proba())` - those differ once
      `ThresholdedEnsemble` overrides the winning class, which a naive `.max()` would get wrong.
      Found and fixed a real, unrelated bug via this testing: `models/xgb_model.joblib` and
      `models/lgbm_model.joblib` were still the ones from the *first, broken* Colab reassembly
      (mismatched feature set, from before the sklearn-version feature-selection bug was fixed)
      - they'd silently drifted out of sync with `rf_model.joblib` and `selected_features.json`
      and nothing had exercised the full three-model ensemble together since, so it went
      unnoticed until `detector.py` did. Fixed by rerunning `python src/run_training.py` to
      regenerate a fully consistent (untuned) baseline set of all three models + scaler +
      feature list together - confirmed identical to the original baseline numbers (weighted F1
      0.9985, U2R F1 0.6526 with the threshold applied), so nothing was lost. The eventual
      Colab-tuned models still need to be swapped in properly once that run completes.
- [x] Test against real local network traffic - both `capture.py` (feature extraction) and
      `detector.py` (end-to-end capture -> classify -> log) verified against real DNS/HTTPS/QUIC
      traffic on the local network, logging correctly to `detections.db`.

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
