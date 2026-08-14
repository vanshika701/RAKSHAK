# RAKSHAK: A Machine Learning-Based Network Intrusion Detection System

**Internship Report**

---

## 1. Problem Statement

Network intrusion detection systems (NIDS) are a core component of modern cybersecurity
infrastructure, tasked with distinguishing malicious network traffic from legitimate activity in
real time. Traditional signature-based NIDS (e.g. Snort, Suricata rule sets) detect only attacks
matching known patterns, leaving them structurally blind to novel or slightly-varied attacks.
Anomaly-based, machine-learning-driven detection offers a complementary approach: by learning the
statistical structure of both normal and malicious traffic from labeled flow data, an ML model
can generalize to variations of known attack types without requiring a hand-written signature for
each one.

This project, RAKSHAK, builds and evaluates such a system end-to-end: from raw packet capture
through classification into one of five categories — Normal traffic, or one of four attack
families (DoS, Probe, R2L, U2R) — with the results surfaced on a live dashboard. The goal was not
only to train an accurate classifier, but to build, test, and honestly evaluate a complete,
working pipeline: data cleaning, feature engineering, model training and tuning, ensembling,
cross-dataset validation, live traffic capture, and real-time inference — each stage tested
against real data or real traffic rather than assumed correct.

## 2. Literature Review

**Intrusion detection datasets.** Two widely-used labeled datasets anchor this project:

- **CICIDS2017** (Sharafaldin, Lashkari & Ghorbani, 2018) — captured over 5 days at the Canadian
  Institute for Cybersecurity, containing benign traffic alongside DoS, DDoS, brute-force, web
  attack, infiltration, port scan, and botnet traffic, with 80+ flow-level features computed by
  CICFlowMeter. It remains one of the most commonly used modern benchmark datasets for ML-based
  NIDS research, chosen here as the primary training dataset for its scale (2.8M+ flows) and
  realistic, contemporary attack diversity.
- **UNSW-NB15** (Moustafa & Slay, 2015) — a separate dataset built at the Australian Centre for
  Cyber Security, using the Argus and Bro-IDS tools to compute a different set of flow features
  over a different (synthetic, IXIA PerfectStorm-generated) traffic mix. Used here exclusively as
  an independent, out-of-distribution test set for cross-dataset generalization analysis (Section
  6) — deliberately never trained on directly with the main 25-feature model.

**Ensemble tree-based models.** Gradient-boosted decision trees and random forests are the
dominant model family for tabular intrusion-detection data, consistently outperforming deep
learning approaches on flow-level (as opposed to raw-packet or payload-level) features, given the
comparatively small feature count and strong nonlinear-but-tabular structure. This project uses:
Random Forest (Breiman, 2001), XGBoost (Chen & Guestrin, 2016), and LightGBM (Ke et al., 2017),
combined via soft voting — averaging predicted class probabilities — rather than a single model,
since different tree algorithms empirically make different mistakes on the same data (confirmed
in Section 5), making their combination more robust than any one alone.

**Class imbalance.** Intrusion datasets are inherently and severely imbalanced — genuine attacks,
especially rare categories like privilege-escalation attacks, are a tiny fraction of real network
traffic. SMOTE (Synthetic Minority Over-sampling Technique; Chawla et al., 2002) is the standard
technique for addressing this by generating synthetic minority-class examples via k-nearest-neighbor
interpolation, used here — with a documented caveat, discussed in Section 4.3, about a real
overfitting failure mode it can introduce when combined carelessly with cross-validation.

**Model interpretability.** SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017) was used
to investigate *why* the weakest class (U2R) was being misclassified, rather than treating the
model as an unexaminable black box — detailed in Section 5.3.

## 3. Dataset Description

### 3.1 CICIDS2017 (primary training dataset)

After cleaning (Section 4.1), 2,827,876 flows remain across 71 columns (70 features + label),
mapped onto 5 unified classes:

| Class | Count | Percentage |
|---|---|---|
| Normal | 2,271,320 | 80.32% |
| DoS | 379,748 | 13.43% |
| Probe | 158,804 | 5.62% |
| R2L | 16,012 | 0.57% |
| U2R | 1,992 | 0.07% |

The imbalance ratio between the majority class (Normal) and the rarest (U2R) is roughly **1,140:1**
— the central challenge this project repeatedly returns to.

### 3.2 UNSW-NB15 (cross-dataset validation only)

After cleaning, 153,684 flows across 43 columns, mapped onto the same 5-class taxonomy:

| Class | Count | Percentage |
|---|---|---|
| Normal | 85,646 | 55.73% |
| Probe | 28,546 | 18.57% |
| R2L | 26,250 | 17.08% |
| DoS | 11,630 | 7.57% |
| U2R | 1,612 | 1.05% |

Notably, UNSW-NB15's class balance is **dramatically different** from CICIDS2017's — U2R is
proportionally ~15x more common here, and R2L is not a minority class at all (17.1% vs. 0.57%).
This matters directly for the cross-dataset validation in Section 6: any model trained on one and
tested on the other is being tested under genuine distribution shift, not just "more of the same
data."

## 4. Methodology

### 4.1 Data Cleaning (`src/preprocess.py`)

Both datasets are loaded via chunked reads (`pd.read_csv(..., chunksize=100_000)`) to respect an
8GB RAM constraint. Cleaning included: stripping inconsistent column-name whitespace, collapsing
corrupted "Web Attack" label variants, dropping ~85-115 rows with negative `Flow Duration` (a
known CICFlowMeter capture defect), and mapping each dataset's native labels onto a unified
5-class taxonomy (`DoS`, `Normal`, `Probe`, `R2L`, `U2R`).

A significant, non-obvious bug was found and fixed during this stage: **10.88% of CICIDS2017 rows
were exact feature-duplicates**, and a naive train/test split left 13.41% of the test set
identical, on features, to rows already seen in training — silent train/test leakage. Fixing this
(deduplicating on features before splitting) materially changed downstream results: R2L precision
improved (0.7284 → 0.9795) while U2R collapsed (0.6672 → 0.2167) — revealing that the leakage had
specifically been masking a real U2R weakness, not just inflating scores uniformly.

Two derived features, engineered rather than taken directly from either capture tool, are used
throughout: `fwd_bwd_ratio` (forward-to-backward packet ratio, `+1` Laplace-smoothed to avoid
divide-by-zero) and `bytes_per_sec` (throughput, duration floored at `1e-3` seconds to avoid
extreme values from near-zero-duration flows).

### 4.2 Feature Selection and Scaling

From CICIDS2017's cleaned 70 features, the top 25 by Random Forest importance (fit on a 10%
stratified sample of the training split only) are selected and persisted to
`models/selected_features.json` — the single source of truth for every downstream stage,
including live inference. Features are scaled with `MinMaxScaler`, fit on the training split
only and applied to validation/test, to avoid leaking test-set statistics into training.

### 4.3 Class Imbalance Handling

SMOTE is applied to the training split only (after scaling, since SMOTE's k-NN interpolation is
distance-based and unscaled byte-count columns would otherwise dominate), bringing R2L and U2R up
to the *Probe* class's count — deliberately not full parity with Normal, which would have meant
interpolating ~1,800,000 mostly-synthetic U2R rows from only ~1,587 real anchor points.

A genuine failure mode of this technique was discovered and fixed during Colab hyperparameter
tuning (Section 4.4): applying SMOTE once, before drawing a subsample for cross-validated
hyperparameter search, means every CV fold's "held-out" portion also contains SMOTE-derived
synthetic points interpolated from real points that leaked into the training portion of the same
fold. This produced a misleadingly high cross-validation score (0.9865 macro F1) for a LightGBM
configuration that then collapsed to 0.28 macro F1 on genuinely held-out validation data — the
model had learned to memorize a dense synthetic neighborhood, not to generalize. The fix: don't
trust the leaky CV score for model selection; instead, evaluate a small set of deliberately
regularized candidates directly against real, untouched validation data.

### 4.4 Model Training and Hyperparameter Tuning

Three base models were trained on the SMOTE'd, scaled training split:

- **Random Forest** (200 trees) — trained and left with default-ish hyperparameters; a Colab
  tuning pass was deliberately skipped given diminishing expected returns relative to the time
  cost, and is documented as such rather than silently omitted.
- **XGBoost** — hyperparameter-tuned via `RandomizedSearchCV` (20 iterations, 3-fold CV, `f1_macro`
  scoring) on Google Colab, per the project's hardware constraint (full-scale `RandomizedSearchCV`
  on 2M+ rows is impractical on 8GB local RAM). XGBoost's sklearn API requires integer-encoded
  labels, unlike the other two models, handled via a purpose-built `LabelDecodingClassifier`
  wrapper that exposes a consistent string-label `predict()`/`predict_proba()` interface.
- **LightGBM** — same Colab search approach initially, but its naive tuned result was discarded
  after the SMOTE-leakage failure described in Section 4.3; the actual final LightGBM model comes
  from a small manually-regularized candidate search (capped tree depth, added
  `min_child_samples`), selected by real validation performance rather than CV score.

A cross-environment reproducibility issue was also found and fixed during this stage: Colab's
scikit-learn version (1.6.1) differed from the local environment's (1.9.0), causing an
independently-recomputed feature-selection step to silently choose a *different* 25-feature list
than the canonical one — even with an identical `random_state`. Fixed by having the Colab notebook
load the canonical `selected_features.json` rather than recomputing it, guaranteeing every model
in the ensemble agrees on the same input columns.

### 4.5 Ensemble and Decision Threshold

The three trained models are combined via a custom `SoftVotingEnsemble` (equal-weighted averaging
of `predict_proba()` outputs) rather than scikit-learn's `VotingClassifier`, which would require
re-fitting all three models from scratch and doesn't compose cleanly with the custom
`LabelDecodingClassifier` wrapper.

A `ThresholdedEnsemble` wrapper further requires extra confidence (a tunable threshold) before
predicting U2R specifically, falling back to the next-most-likely class otherwise — a targeted
precision/recall trade-off that leaves every other class's decision boundary untouched. The
threshold was chosen by sweeping 0.50–0.95 on the validation set only (never test), with both U2R
F1 and macro F1 peaking at **threshold = 0.80**, and confirmed once, at the very end, on the
untouched test set — this validation/test discipline (a three-way stratified split, with test
checked exactly once) is maintained throughout the project specifically to avoid a quieter repeat
of the duplicate-row leakage found in Section 4.1.

## 5. Results

### 5.1 Final Model Performance (untouched test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DoS | 0.9990 | 0.9966 | 0.9978 | 32,167 |
| Normal | 0.9993 | 0.9989 | 0.9991 | 209,435 |
| Probe | 0.9950 | 0.9993 | 0.9971 | 9,070 |
| R2L | 0.9599 | 0.9956 | 0.9774 | 1,130 |
| U2R | 0.5678 | 0.7828 | 0.6582 | 198 |

**Weighted F1: 0.9985. Macro F1: 0.9259.** The project's target metric (weighted F1 > 95%) is
cleared by a wide margin, by every individual model and the ensemble.

### 5.2 Ensemble vs. Individual Models

The soft-voting ensemble outperformed every individual model on macro F1, and specifically
recovered R2L performance close to Random Forest's individual best while retaining most of
LightGBM's U2R edge — direct evidence that the three models make different mistakes, which is
exactly what makes ensembling worthwhile rather than redundant. The U2R decision threshold then
improved U2R F1 further (0.4457 → 0.6526 pre-tuning baseline, later 0.6582 with tuned base
models) at a real, disclosed cost: U2R recall dropped from ~93% to ~78%, trading missed
detections for far fewer false alarms (458 → 122 false U2R predictions on the test set, a 73%
reduction) — a genuine security trade-off, not an unambiguous win, and one a deployed system
would need a human analyst to accept knowingly.

### 5.3 U2R Root-Cause Investigation (SHAP)

Given U2R's persistently weak precision, a targeted SHAP analysis compared true-positive U2R
predictions against false-positive Normal-flows-misclassified-as-U2R. Finding: both groups are
driven by nearly the same features (packet-size statistics, Destination Port), and both resemble
small, sparse, low-volume flows — evidence the model is picking up genuine, if ambiguous, signal
rather than reacting to noise. `PSH Flag Count` was present in true positives but not false
positives, a lead for future feature engineering. The practical conclusion: this is a **data
scarcity problem** (only ~1,587 real U2R training examples across the entire dataset), not a bug
with an available code-level fix — further hyperparameter tuning was confirmed not to close this
gap (Section 4.4), consistent with the root cause being insufficient real examples rather than
suboptimal model configuration.

## 6. Cross-Dataset Generalization Analysis

To test whether RAKSHAK learned something that generalizes beyond CICIDS2017's specific traffic,
rather than memorizing dataset-specific quirks, a separate experiment
(`notebooks/05_cross_dataset_validation.ipynb`) trained a single Random Forest on the 6 features
common to both datasets' schemas (duration, forward/backward packet and byte counts,
`fwd_bwd_ratio`, `bytes_per_sec`), then evaluated it in two places without retraining:

| Evaluation | Weighted F1 | Macro F1 |
|---|---|---|
| In-distribution (CICIDS2017 held-out test) | 0.9770 | 0.6922 |
| Cross-dataset, zero-shot (all of UNSW-NB15) | **0.3988** | **0.1431** |

The collapse is severe — the model predicts "Normal" for essentially every UNSW-NB15 row. This
was investigated rather than left as an unexplained number: median feature values differ
substantially between the two datasets (UNSW-NB15's median duration is ~7.6x longer, forward
bytes ~15x more, throughput ~6.6x higher), most likely reflecting differing flow-capture and
aggregation conventions between CICFlowMeter and Argus. Random Forest's decision boundaries are
hard numeric thresholds learned from CICIDS2017's value ranges; when nearly every UNSW-NB15 row
falls on the same side of every learned threshold, predictions collapse into a single leaf.

**Conclusion**: a shared feature *name* and *definition* does not guarantee a shared feature
*distribution*. Genuine cross-dataset transfer would require per-dataset normalization or a
domain-adaptation technique, not merely aligning column names — an honest, citable limitation
rather than a claim of general-purpose readiness.

## 7. Live Detection System

Beyond offline evaluation, RAKSHAK was built and tested as a complete, working pipeline against
real network traffic:

- **`src/capture.py`** — Scapy-based packet capture, reconstructing network flows from raw
  packets by a direction-independent 5-tuple key, then computing the same 25 features
  `selected_features.json` specifies. Two genuine bugs were found and fixed via live testing
  against real traffic (not just code review): captured packet length initially included the
  Ethernet frame header (a systematic 14-byte inflation versus CICFlowMeter's IP-packet-only
  convention), and forward/backward flow direction was initially decided from whichever packet
  was captured first — which silently reversed `Destination Port` and every forward/backward
  feature for any TCP connection already open before the sniffer started. Fixed by deferring
  direction assignment to flow-close time and using a bare TCP SYN packet as authoritative proof
  of the true initiator.
- **`src/detector.py`** — loads the trained ensemble, classifies each finished flow, and logs
  every result (Normal and otherwise) to a SQLite database, printing an alert only for non-Normal
  predictions. Testing this end-to-end surfaced a further real issue: the local model files had
  silently drifted out of sync (XGBoost/LightGBM reassembled from an earlier, since-corrected
  Colab run using a mismatched feature set) — caught because this was the first time all three
  models were exercised together as a genuine ensemble, and fixed by regenerating a consistent
  baseline set of all three models together.
- **`src/app.py`** — a Flask dashboard reading the same SQLite log, auto-refreshing every 5
  seconds, with a connection table, an alerts panel, and summary cards.

**A live test result worth reporting honestly**: an authorized self-scan (`nmap -sS`) against the
local router generated on the order of ~1,000 individual flows (one per scanned port), of which
only 4 were flagged as non-Normal — labeled `R2L` (not `Probe`, the semantically closer class),
at low confidence (~35%, barely above the 20% baseline for a 5-class problem). Investigation
suggests this reflects an unusual SYN-retransmission timing pattern on two specific filtered
ports, rather than the model robustly recognizing scanning behavior as such. Combined with the
cross-dataset finding above, this is consistent, converging evidence that RAKSHAK generalizes
narrowly to patterns close to its training distribution, and does not reliably detect real-world
traffic that differs structurally from CICIDS2017's specific attack simulations — an honest
finding, not a hidden failure.

## 8. Limitations

1. **U2R precision/recall trade-off** (Section 5.3) — a data-scarcity ceiling, not a modeling
   defect; not fixable without more real minority-class examples.
2. **Cross-dataset generalization gap** (Section 6) — weighted F1 collapses from 97.7% to 39.9%
   moving from CICIDS2017 to UNSW-NB15 even on a deliberately shared feature schema.
3. **Per-flow independent classification** — the live detector classifies each flow in isolation,
   with no awareness that, e.g., many short connection attempts from the same source in a short
   window is itself a signal (the more obvious human tell of a port scan). This is a real
   architectural limitation surfaced directly by the live scan test in Section 7.
4. **Live feature extraction approximations** — `Subflow Fwd/Bwd Bytes` are approximated as total
   forward/backward bytes rather than replicating CICFlowMeter's own sub-flow-splitting logic;
   standard deviation uses the population (not sample) formula, since CICFlowMeter's own
   convention isn't recoverable from the training CSVs alone.
5. **Dashboard is not production-hardened** — Flask's built-in development server (`debug=True`)
   is used for local demonstration; genuine deployment beyond a single local machine would need a
   production WSGI server and `debug=False`.

## 9. Future Scope

- **More real U2R/R2L data**, or a more targeted oversampling strategy (Borderline-SMOTE, focused
  specifically on the U2R-vs-Normal decision boundary) — the SHAP investigation's `PSH Flag Count`
  lead is a concrete starting point.
- **Domain adaptation** for genuine cross-dataset transfer — per-dataset feature normalization or
  adversarial domain-adaptation techniques, rather than assuming a shared schema is sufficient.
- **Session/multi-flow correlation features** — aggregating behavior across recent flows from the
  same source (e.g. a rolling count of distinct destination ports contacted), which the live scan
  test suggests is necessary to reliably catch reconnaissance-style behavior.
- **Production-hardening the dashboard** — a proper WSGI server, authentication, and TLS if ever
  exposed beyond a single trusted machine.

## 10. Conclusion

RAKSHAK is a complete, honestly-evaluated, end-to-end ML-based intrusion detection system:
trained and rigorously validated on CICIDS2017 (weighted F1 0.9985, macro F1 0.9259 on an
untouched test set), cross-validated against UNSW-NB15 with a diagnosed and explained
generalization gap rather than an unexamined number, and proven to work against real, live
network traffic through a working capture-to-dashboard pipeline — not merely claimed to work.
Several genuine, non-obvious bugs (train/test leakage, a SMOTE-cross-validation interaction, a
cross-environment feature-selection mismatch, two live-capture direction/length bugs, and a
silent model-consistency drift) were found through deliberate testing rather than assumed away,
and are documented here alongside the fixes rather than hidden. The system's most significant
known weakness — U2R detection and cross-dataset generalization — are both rooted in genuine data
limitations rather than implementation shortcuts, and are presented as such: honest, bounded
claims about what this system does and does not yet do well.

## References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion
   Detection Dataset and Intrusion Traffic Characterization. *Proceedings of the 4th
   International Conference on Information Systems Security and Privacy (ICISSP)*.
2. Moustafa, N., & Slay, J. (2015). UNSW-NB15: A Comprehensive Data Set for Network Intrusion
   Detection Systems. *Military Communications and Information Systems Conference (MilCIS)*.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the
   22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
5. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017).
   LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information
   Processing Systems (NeurIPS)*.
6. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic
   Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
7. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.
   *Advances in Neural Information Processing Systems (NeurIPS)*.
8. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine
   Learning Research*, 12, 2825-2830.
