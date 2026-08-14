"""
rebuild_ensemble.py

Rebuilds the soft-voting ensemble from the current models/*.joblib files
(after reassemble_tuned_models.py has swapped in the Colab-tuned
XGBoost/LightGBM) and re-sweeps the U2R decision threshold on validation.

Deliberately validation-only, same discipline as the original threshold
tuning: the models changed, so U2R_DECISION_THRESHOLD=0.80 (chosen for the
old, untuned ensemble) is not assumed to still be optimal - it needs to be
re-derived, and only ever against validation, never test. Once a threshold
is chosen from this script's output, confirm it once on test separately.

Run via: python src/rebuild_ensemble.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_model import (  # noqa: E402
    build_soft_voting_ensemble,
    evaluate_model,
    load_trained_models,
    reconstruct_val_test_splits,
    sweep_u2r_threshold,
)


def main() -> None:
    """Load the current three models, build the ensemble, evaluate it on
    validation, and sweep U2R thresholds - all validation-only.
    """
    models = load_trained_models()
    X_val, y_val, X_test, y_test = reconstruct_val_test_splits()

    ensemble = build_soft_voting_ensemble([models["rf"], models["xgb"], models["lgbm"]])

    print("\n--- Plain ensemble, validation set ---")
    evaluate_model(ensemble, X_val, y_val, "Rebuilt Soft-Voting Ensemble (validation)")

    print("\n--- U2R threshold sweep, validation set ---")
    thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    sweep_u2r_threshold(ensemble, X_val, y_val, thresholds=thresholds)


if __name__ == "__main__":
    main()
