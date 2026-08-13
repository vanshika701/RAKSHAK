"""
preprocess.py

Cleans and prepares the CICIDS2017 and UNSW-NB15 datasets for model training.

Pipeline (see instructions.md Step 3 for the full spec):
    1. Chunked CSV loading (memory-safe for large files)
    2. Column name cleanup
    3. Label mapping to 5 unified classes: Normal, DoS, Probe, R2L, U2R
    4. Handling of infinite/NaN values
    5. Categorical encoding
    6. Dropping constant/near-zero-variance columns
    7. MinMax scaling
    8. Saving as .parquet
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT = the RAKSHAK/ folder, computed relative to this file's own
# location so the script works no matter what directory it's run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CICIDS_DIR = DATA_RAW_DIR / "cicids2017"
UNSW_DIR = DATA_RAW_DIR / "unsw_nb15"

CICIDS_OUTPUT_PATH = DATA_PROCESSED_DIR / "cicids_clean.parquet"
UNSW_OUTPUT_PATH = DATA_PROCESSED_DIR / "unsw_clean.parquet"

# Fitted LabelEncoders for UNSW-NB15's text columns, saved here so the exact
# same string -> int mapping can be reapplied to live traffic in detector.py.
UNSW_ENCODERS_PATH = MODELS_DIR / "unsw_label_encoders.joblib"

# --------------------------------------------------------------------------
# Pipeline constants
# --------------------------------------------------------------------------
# Number of rows read from each CSV at a time (see CLAUDE.md data rules).
CHUNK_SIZE = 100_000

# Used anywhere randomness is involved (train/test splits, SMOTE, etc.) so
# results are reproducible across runs.
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Label mappings: raw dataset labels -> 5 unified attack classes
# --------------------------------------------------------------------------
# CICIDS2017's three "Web Attack ..." labels are stored with a corrupted
# separator character due to an encoding issue in the source CSVs. All three
# are normalised to the single string "Web Attack" before this map is
# applied (see clean_cicids_labels(), written in the next step).
CICIDS_LABEL_MAP = {
    "BENIGN": "Normal",
    "DDoS": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS Hulk": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",
    "Heartbleed": "DoS",
    "PortScan": "Probe",
    "FTP-Patator": "R2L",
    "SSH-Patator": "R2L",
    "Web Attack": "R2L",
    "Bot": "U2R",
    "Infiltration": "U2R",
}

UNSW_LABEL_MAP = {
    "Normal": "Normal",
    "DoS": "DoS",
    "Generic": "DoS",
    "Reconnaissance": "Probe",
    "Analysis": "Probe",
    "Fuzzers": "Probe",
    "Backdoor": "R2L",
    "Exploits": "R2L",
    "Shellcode": "U2R",
    "Worms": "U2R",
}

# UNSW-NB15's text (non-numeric) feature columns. CICIDS2017 has no
# equivalent — all of its features are already numeric flow statistics.
UNSW_CATEGORICAL_COLUMNS = ["proto", "service", "state"]


# --------------------------------------------------------------------------
# Column cleaning
# --------------------------------------------------------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all column names.

    CICIDS2017's CSVs have inconsistent spacing in their headers
    (e.g. " Destination Port" vs "Total Length of Fwd Packets"), which would
    otherwise make column lookups like df["Label"] fail unpredictably
    depending on which file the chunk came from.
    """
    df.columns = df.columns.str.strip()
    return df


def normalize_web_attack_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the three "Web Attack ..." label variants into one string.

    CICIDS2017's CSVs store these labels with a corrupted separator
    character (an encoding artifact in the source files), producing three
    distinct-looking strings that should all map to the same class. Matching
    on the stable "Web Attack" prefix avoids depending on that corrupted
    character.
    """
    is_web_attack = df["Label"].str.startswith("Web Attack")
    df.loc[is_web_attack, "Label"] = "Web Attack"
    return df


# --------------------------------------------------------------------------
# Dataset loaders
# --------------------------------------------------------------------------
def load_cicids2017() -> pd.DataFrame:
    """Load and clean all CICIDS2017 daily CSVs into a single DataFrame.

    Each of the 8 daily CSVs is read in CHUNK_SIZE-row chunks (see
    CLAUDE.md data rules). For every chunk: column names are stripped, the
    "Web Attack" label variants are normalised, and Label is mapped to one
    of the 5 unified classes (Normal, DoS, Probe, R2L, U2R).
    """
    csv_files = sorted(CICIDS_DIR.glob("*.csv"))
    cleaned_chunks = []

    for csv_path in csv_files:
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
            chunk = clean_column_names(chunk)
            chunk = normalize_web_attack_labels(chunk)
            chunk["Label"] = chunk["Label"].map(CICIDS_LABEL_MAP)
            chunk = handle_inf_and_nan(chunk)
            cleaned_chunks.append(chunk)

    df = pd.concat(cleaned_chunks, ignore_index=True)
    df = drop_constant_columns(df)
    return df


# --------------------------------------------------------------------------
# Cleaning shared by both datasets
# --------------------------------------------------------------------------
def handle_inf_and_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN, then drop any row containing NaN.

    Rate-based features (e.g. CICIDS2017's "Flow Bytes/s", UNSW-NB15's
    "rate") are computed as some_value / duration, and become +/-inf when
    duration == 0. Converting these to NaN first means a single dropna()
    call handles both "infinite" and "naturally missing" values together.
    Affected rows are a tiny fraction of the dataset (see EDA notebook).
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns where every row holds the exact same value.

    A constant column gives a model nothing to split on, and would later
    break MinMaxScaler outright (max - min == 0 means a division by zero).
    Which columns qualify is computed dynamically per-dataset rather than
    hardcoded, since it depends on exactly which raw CSVs got loaded.
    Must run on the fully concatenated DataFrame, not per-chunk — a column
    could look constant within one chunk but vary across the full dataset.
    """
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    return df.drop(columns=constant_cols)


def encode_categorical_columns(
    df: pd.DataFrame, columns: list[str], save_path: Path
) -> pd.DataFrame:
    """Label-encode the given text columns in place and persist the encoders.

    XGBoost, LightGBM, and Random Forest all split on thresholds (e.g. "is
    this feature <= 1.5?") rather than on linear combinations, so the
    arbitrary numeric order LabelEncoder introduces (tcp=2, udp=1, ...)
    does not mislead them the way it would a linear model.

    The fitted encoders are saved to `save_path` so detector.py can apply
    the exact same string -> int mapping to live traffic later. This must
    run on the full column, not per-chunk — fitting a fresh encoder on
    each chunk would give the same string a different number depending on
    which chunk it appeared in.
    """
    encoders = {}
    for column in columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, save_path)
    return df


def load_unsw_nb15() -> pd.DataFrame:
    """Load and clean both UNSW-NB15 CSVs (training + testing) into one DataFrame.

    Both files are read in CHUNK_SIZE-row chunks using "utf-8-sig" encoding,
    which strips the BOM character at the start of the "id" column. For
    every chunk: column names are stripped, a unified "Label" column is
    built from "attack_cat" via UNSW_LABEL_MAP, the non-feature / leakage
    columns ("id", "label", "attack_cat") are dropped, and inf/NaN rows are
    removed. Once every chunk has been loaded and concatenated, the text
    columns in UNSW_CATEGORICAL_COLUMNS are label-encoded, and any constant
    columns are dropped.
    """
    csv_files = sorted(UNSW_DIR.glob("*.csv"))
    cleaned_chunks = []

    for csv_path in csv_files:
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, encoding="utf-8-sig"):
            chunk = clean_column_names(chunk)
            chunk["Label"] = chunk["attack_cat"].map(UNSW_LABEL_MAP)
            chunk = chunk.drop(columns=["id", "label", "attack_cat"])
            chunk = handle_inf_and_nan(chunk)
            cleaned_chunks.append(chunk)

    df = pd.concat(cleaned_chunks, ignore_index=True)
    df = encode_categorical_columns(df, UNSW_CATEGORICAL_COLUMNS, UNSW_ENCODERS_PATH)
    df = drop_constant_columns(df)
    return df


# --------------------------------------------------------------------------
# Pipeline entry point
# --------------------------------------------------------------------------
def main() -> None:
    """Run the full cleaning pipeline for both datasets and save as parquet.

    The saved files are cleaned but NOT scaled. MinMax scaling is
    deliberately deferred until after the train/test split (done later, in
    train_model.py) so the scaler is fit only on training data - fitting it
    here, on the full dataset, would leak test-set statistics into training
    and inflate evaluation results.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning CICIDS2017...")
    cicids_df = load_cicids2017()
    print(f"  {cicids_df.shape[0]:,} rows, {cicids_df.shape[1]} columns")
    cicids_df.to_parquet(CICIDS_OUTPUT_PATH, index=False)
    print(f"  saved to {CICIDS_OUTPUT_PATH}")

    print("Loading and cleaning UNSW-NB15...")
    unsw_df = load_unsw_nb15()
    print(f"  {unsw_df.shape[0]:,} rows, {unsw_df.shape[1]} columns")
    unsw_df.to_parquet(UNSW_OUTPUT_PATH, index=False)
    print(f"  saved to {UNSW_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
