"""
reassemble_tuned_models.py

One-time script to bring the Colab-tuned XGBoost and LightGBM models into
models/, replacing the untuned local ones.

XGBoost was saved from Colab as two separate files (raw model + label
encoder) rather than wrapped in LabelDecodingClassifier, because defining
that wrapper inside a Colab notebook cell would tag the class under the
module "__main__" - the exact pickling bug already hit and fixed once for
this project (see run_training.py's docstring). Importing the real,
already-correctly-pathed LabelDecodingClassifier here instead avoids that
entirely.

Run via: python src/reassemble_tuned_models.py
"""

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_model import LabelDecodingClassifier, MODELS_DIR  # noqa: E402

STAGING_DIR = MODELS_DIR / "tuned_from_colab"


def main() -> None:
    """Wrap the raw tuned XGBoost model and copy the tuned LightGBM model
    into models/, overwriting the untuned versions.
    """
    xgb_raw = joblib.load(STAGING_DIR / "xgb_model_raw.joblib")
    xgb_label_encoder = joblib.load(STAGING_DIR / "xgb_label_encoder.joblib")
    xgb_wrapped = LabelDecodingClassifier(xgb_raw, xgb_label_encoder)
    joblib.dump(xgb_wrapped, MODELS_DIR / "xgb_model.joblib")
    print(f"Wrapped tuned XGBoost saved to {MODELS_DIR / 'xgb_model.joblib'}")

    lgbm_tuned = joblib.load(STAGING_DIR / "lgbm_model.joblib")
    joblib.dump(lgbm_tuned, MODELS_DIR / "lgbm_model.joblib")
    print(f"Tuned LightGBM saved to {MODELS_DIR / 'lgbm_model.joblib'}")

    print("\nDone. models/xgb_model.joblib and models/lgbm_model.joblib now hold the tuned versions.")
    print("models/rf_model.joblib is untouched (still the untuned baseline).")


if __name__ == "__main__":
    main()
