"""
train_model.py

Trains and evaluates the XGBoost / LightGBM / Random Forest ensemble on
CICIDS2017.

Pipeline (see instructions.md Steps 5-9 and CLAUDE.md Model Rules):
    1. Load the cleaned, unscaled parquet produced by preprocess.py
    2. Stratified train/test split
    3. Feature selection via quick Random Forest importances (fit on train only)
    4. MinMax scaling (fit on train only)
    5. SMOTE oversampling (train only, applied after scaling)
    6. Train XGBoost, LightGBM, Random Forest
    7. Soft-voting ensemble + evaluation, then cross-dataset validation on UNSW-NB15

Steps 3-7 are built incrementally; this file currently implements step 2.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CICIDS_PATH = DATA_PROCESSED_DIR / "cicids_clean.parquet"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Label"

# What fraction of X_train to use when fitting the quick Random Forest for
# feature importances - a speed optimization, not a modeling choice.
FEATURE_SAMPLE_FRAC = 0.10
N_SELECTED_FEATURES = 25


def load_features_and_target(parquet_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a cleaned parquet file and split it into features (X) and target (y)."""
    df = pd.read_parquet(parquet_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split.

    stratify=y keeps class proportions identical in both splits - critical
    here since U2R is only ~0.07% of the data; a plain random split could
    easily over- or under-represent it in the test set purely by chance.

    Everything downstream that fits to data (feature selection, the
    scaler, SMOTE) must only ever be fit on the training split returned
    here, never the test split - see the leakage discussion in
    preprocess.py's main() docstring.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def select_top_features(
    X_train: pd.DataFrame, y_train: pd.Series, n_features: int = N_SELECTED_FEATURES
) -> list[str]:
    """Rank features by importance from a quick Random Forest and return the top N.

    The forest is fit on a stratified FEATURE_SAMPLE_FRAC sample of
    X_train/y_train only - never the test split, and never the full
    training set, since ranking features doesn't need 2M+ rows to be
    reliable and this is meant to be fast, not the final tuned model.

    Random Forest importance = averaged, across every tree and every
    split, how much using that feature reduced class impurity. Features
    the forest leaned on heavily score high; features it barely touched
    score near zero.
    """
    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=FEATURE_SAMPLE_FRAC,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    quick_rf = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
    )
    quick_rf.fit(X_sample, y_sample)

    importances = pd.Series(quick_rf.feature_importances_, index=X_train.columns)
    return importances.sort_values(ascending=False).head(n_features).index.tolist()


def save_selected_features(features: list[str], save_path: Path) -> None:
    """Persist the selected feature list as JSON.

    Per CLAUDE.md, this file is the single source of truth for feature
    names everywhere downstream - training, scaling, and live inference in
    detector.py all load from here instead of hardcoding column names.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(features, f, indent=2)


if __name__ == "__main__":
    X, y = load_features_and_target(CICIDS_PATH)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")

    print("\nTrain class distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nSelecting top features...")
    top_features = select_top_features(X_train, y_train)
    save_selected_features(top_features, SELECTED_FEATURES_PATH)
    print(f"Top {N_SELECTED_FEATURES} features saved to {SELECTED_FEATURES_PATH}:")
    for feature in top_features:
        print(f"  {feature}")

    X_train = X_train[top_features]
    X_test = X_test[top_features]

    print("\nTest class distribution:")
    print(y_test.value_counts(normalize=True))
