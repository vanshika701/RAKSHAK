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

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CICIDS_PATH = DATA_PROCESSED_DIR / "cicids_clean.parquet"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
LGBM_MODEL_PATH = MODELS_DIR / "lgbm_model.joblib"

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# SMOTE brings classes smaller than this reference class up to its size,
# rather than all the way to the majority class - see apply_smote().
SMOTE_TARGET_CLASS = "Probe"
TARGET_COLUMN = "Label"

RF_N_ESTIMATORS = 200

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

LGBM_N_ESTIMATORS = 300
LGBM_MAX_DEPTH = 6
LGBM_LEARNING_RATE = 0.1

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


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, save_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit MinMaxScaler on X_train only, transform both splits.

    Same leakage principle as feature selection: the scaler's min/max must
    never be influenced by values the model won't see until evaluation.
    The fitted scaler is saved so detector.py can apply the identical
    transform to live traffic later.
    """
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)
    return X_train_scaled, X_test_scaled


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample severely underrepresented classes in the training set only.

    Runs after scaling, not before: SMOTE generates synthetic minority-class
    rows by interpolating between a real point and its nearest neighbors
    (Euclidean distance), so unscaled byte-count columns would otherwise
    dominate the distance calculation over ratio columns like fwd_bwd_ratio.

    Rather than the default 'auto' strategy (which would oversample every
    non-majority class up to match Normal - ~1.8M rows, ballooning U2R from
    1,595 real rows to 1.8M mostly-synthetic ones), this brings only R2L
    and U2R up to SMOTE_TARGET_CLASS's size. Full parity with the majority
    would mean interpolating repeatedly from too few real anchor points to
    produce genuine diversity, and would roughly quadruple the training set
    size for no real benefit.

    Applied only to X_train/y_train - never the test set, which must keep
    real-world class frequencies so evaluation reflects genuine performance.
    """
    class_counts = y_train.value_counts()
    target_count = class_counts[SMOTE_TARGET_CLASS]
    sampling_strategy = {
        label: target_count for label, count in class_counts.items() if count < target_count
    }

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train the Random Forest ensemble member on the scaled, SMOTE'd training set.

    No class_weight='balanced' here, deliberately: apply_smote() already
    rebalanced the classes. Adding loss-level class weighting on top of
    that would double-correct for imbalance and likely overcorrect against
    Normal/DoS precision.
    """
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    return rf


class LabelDecodingClassifier:
    """Wraps a classifier that requires integer-encoded labels so its
    predict()/predict_proba() interface matches every other model's -
    working with the original string class names (Normal, DoS, ...).

    XGBoost's sklearn wrapper requires y to already be integers 0..n-1 for
    fit(), unlike RandomForestClassifier or LightGBM, which accept the
    string labels directly. Defined as a real class (not a closure or a
    monkey-patched instance method) so it survives joblib.dump()/load()
    correctly - a lambda-based patch would not.
    """

    def __init__(self, model, label_encoder: LabelEncoder):
        self.model = model
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        # Columns already come out in label_encoder's class order, which
        # matches self.classes_ - needed later for the soft-voting ensemble.
        return self.model.predict_proba(X)


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> LabelDecodingClassifier:
    """Train the XGBoost ensemble member on the scaled, SMOTE'd training set.

    tree_method="hist" is the fast, histogram-based split-finding
    algorithm - the standard choice for tabular data at this scale.
    Trained on the exact same data as train_random_forest() so the two
    models' results are directly comparable, in particular on U2R, where
    the Random Forest baseline is currently weak (0.22 precision).
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    xgb = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train_encoded)
    return LabelDecodingClassifier(xgb, label_encoder)


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    """Train the LightGBM ensemble member on the scaled, SMOTE'd training set.

    Unlike XGBoost, LightGBM's sklearn wrapper accepts the original string
    labels directly for fit() - it handles the integer encoding internally,
    the same as RandomForestClassifier - so no LabelDecodingClassifier
    wrapper is needed here. verbose=-1 silences LightGBM's per-iteration
    training log, which is otherwise very noisy at this row count.
    """
    lgbm = LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        max_depth=LGBM_MAX_DEPTH,
        learning_rate=LGBM_LEARNING_RATE,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    return lgbm


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> float:
    """Print a classification report and confusion matrix, return weighted F1.

    Weighted F1 (not accuracy) is CLAUDE.md's target metric (>95%) -
    accuracy alone would be misleading given how rare U2R/R2L are; a model
    that never predicts them could still score high on accuracy.
    """
    y_pred = model.predict(X_test)

    print(f"\n=== {model_name} evaluation ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix (rows=actual, cols=predicted):")
    labels = sorted(y_test.unique())
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels))

    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\nWeighted F1: {weighted_f1:.4f}")
    return weighted_f1


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

    print("\nScaling features...")
    X_train, X_test = scale_features(X_train, X_test, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    print("\nApplying SMOTE to training set...")
    print("Before:", y_train.value_counts().to_dict())
    X_train, y_train = apply_smote(X_train, y_train)
    print("After: ", y_train.value_counts().to_dict())

    print(f"\nFinal X_train: {X_train.shape}")
    print(f"Final X_test:  {X_test.shape}")

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, RF_MODEL_PATH)
    print(f"Model saved to {RF_MODEL_PATH}")

    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    print(f"Model saved to {XGB_MODEL_PATH}")

    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    print("\nTraining LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train)
    joblib.dump(lgbm_model, LGBM_MODEL_PATH)
    print(f"Model saved to {LGBM_MODEL_PATH}")

    evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
