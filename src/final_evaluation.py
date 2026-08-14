"""
final_evaluation.py

One-time final check: evaluates the current models/*.joblib (tuned
XGBoost/LightGBM + Random Forest), combined into the thresholded ensemble,
on the untouched test set - not validation. This is the actual number
that goes in the report.

Run this exactly once per model configuration - re-running it after
tweaking anything based on what it prints would be the same test-set
leakage split_train_val_test() and the validation-only threshold sweep
were built to avoid in the first place.

Run via: python src/final_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_model import (  # noqa: E402
    U2R_DECISION_THRESHOLD,
    ThresholdedEnsemble,
    build_soft_voting_ensemble,
    evaluate_model,
    load_trained_models,
    reconstruct_val_test_splits,
)


def main() -> None:
    """Load the current three models, build the thresholded ensemble, and
    evaluate it on the test set - once.
    """
    models = load_trained_models()
    _, _, X_test, y_test = reconstruct_val_test_splits()

    ensemble = build_soft_voting_ensemble([models["rf"], models["xgb"], models["lgbm"]])
    final_model = ThresholdedEnsemble(
        ensemble, target_class="U2R", threshold=U2R_DECISION_THRESHOLD
    )

    evaluate_model(
        final_model,
        X_test,
        y_test,
        f"FINAL - Thresholded Ensemble (U2R >= {U2R_DECISION_THRESHOLD}) - TEST SET",
    )


if __name__ == "__main__":
    main()
