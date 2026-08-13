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

# Common-schema versions of both datasets, used for genuine cross-dataset
# validation in Phase 5 (see build_*_common_features() below) - the main
# cleaned parquets above use each dataset's own native columns, which don't
# overlap enough for a model trained on one to run on the other at all.
CICIDS_COMMON_OUTPUT_PATH = DATA_PROCESSED_DIR / "cicids_common.parquet"
UNSW_COMMON_OUTPUT_PATH = DATA_PROCESSED_DIR / "unsw_common.parquet"

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

# Added to denominators in derived-feature ratios to avoid a divide-by-zero
# producing inf (e.g. a flow with 0 backward packets).
EPSILON = 1e-9

# Floor applied to flow duration (in seconds) before computing bytes_per_sec.
# UNSW-NB15 has ~1.4% of rows with duration == 0 exactly; dividing real byte
# counts by EPSILON there sends bytes_per_sec up to the trillions (a repeat
# of the same blowup fwd_bwd_ratio/flag_anomaly had). A near-zero duration
# is a resolution limit, not a real instant transfer, so flooring it at a
# plausible minimum (1ms) keeps the rate meaningful instead of astronomical.
MIN_DURATION_SEC = 1e-3

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
# Feature engineering (CICIDS2017 only - column names are CICFlowMeter-
# specific and don't map onto UNSW-NB15's schema)
# --------------------------------------------------------------------------
def drop_negative_duration_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a negative "Flow Duration".

    A small, known defect in the original CICIDS2017 capture tool corrupts
    the timestamp on a handful of flows, producing a negative duration.
    That makes any duration-derived feature (the native "Flow Bytes/s", or
    our own bytes_per_sec below) meaningless for those rows, so they're
    dropped rather than fed to a ratio.
    """
    return df[df["Flow Duration"] >= 0]


def engineer_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add three derived features to CICIDS2017 flow records.

    bytes_per_sec: overall byte rate for the flow. CICIDS2017 already ships
    a native "Flow Bytes/s" column, so this is a near-duplicate on purpose -
    Phase 3's feature-importance step will decide whether it earns its keep
    alongside the original.

    fwd_bwd_ratio: how one-directional a flow is. Flood-style DoS traffic
    tends to be almost entirely one-way (attacker -> victim), unlike normal
    two-way conversations. Uses "+1" (Laplace/additive smoothing) rather
    than "+EPSILON" in the denominator: ~16% of flows have 0 backward
    packets, which is real signal, not bad data - but dividing by a tiny
    epsilon sends the ratio into the hundreds of billions for those rows,
    which would make MinMaxScaler squash every other row's value down to
    ~0. Dividing by "count + 1" instead keeps the ratio bounded to
    something proportional to the actual packet counts involved.

    flag_anomaly: rate of "closing" TCP flags (FIN, RST) relative to
    "opening" flags (SYN). Unusual ratios can indicate scans or malformed
    handshakes rather than normal connection setup/teardown. Also uses "+1"
    smoothing for the same reason - 95% of flows have 0 SYN flags (many are
    UDP, which has no SYN flag at all), so an epsilon here blows up almost
    the entire column.
    """
    total_bytes = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
    flow_duration_sec = df["Flow Duration"] / 1_000_000  # CICFlowMeter reports microseconds
    df["bytes_per_sec"] = total_bytes / flow_duration_sec.clip(lower=MIN_DURATION_SEC)

    df["fwd_bwd_ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"] + 1)

    df["flag_anomaly"] = (df["FIN Flag Count"] + df["RST Flag Count"]) / (
        df["SYN Flag Count"] + 1
    )

    return df


def drop_duplicate_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are exact duplicates of another row's features.

    CICIDS2017 has a substantial number of these (~11% of rows) - common
    in automated attack traffic like DoS Hulk, which floods mechanically
    identical packets in a tight loop and produces many near-identical
    flow records. This must run before the train/test split: a random
    split has no way to know a "different" row is a byte-for-byte twin of
    one already seen in training, which silently inflates evaluation
    metrics without real generalization (confirmed empirically - 13.41% of
    a test split were exact duplicates of a training row before this fix).

    Deduplicating on features only, not features+Label together, is
    deliberate: it guarantees no feature pattern can appear on both sides
    of the split, even in the case where identical traffic somehow carries
    two different labels. keep="first" is an arbitrary tie-break for that
    conflicting-label case specifically.
    """
    feature_cols = [c for c in df.columns if c != "Label"]
    return df.drop_duplicates(subset=feature_cols, keep="first")


# --------------------------------------------------------------------------
# Dataset loaders
# --------------------------------------------------------------------------
def load_cicids2017() -> pd.DataFrame:
    """Load and clean all CICIDS2017 daily CSVs into a single DataFrame.

    Each of the 8 daily CSVs is read in CHUNK_SIZE-row chunks (see
    CLAUDE.md data rules). For every chunk: column names are stripped, the
    "Web Attack" label variants are normalised, and Label is mapped to one
    of the 5 unified classes (Normal, DoS, Probe, R2L, U2R). Once every
    chunk has been loaded, concatenated, and had its constant columns
    dropped, the three derived features are added.
    """
    csv_files = sorted(CICIDS_DIR.glob("*.csv"))
    cleaned_chunks = []

    for csv_path in csv_files:
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
            chunk = clean_column_names(chunk)
            chunk = normalize_web_attack_labels(chunk)
            chunk["Label"] = chunk["Label"].map(CICIDS_LABEL_MAP)
            chunk = handle_inf_and_nan(chunk)
            chunk = drop_negative_duration_rows(chunk)
            cleaned_chunks.append(chunk)

    df = pd.concat(cleaned_chunks, ignore_index=True)
    df = drop_duplicate_feature_rows(df)
    df = drop_constant_columns(df)
    # Derived features reference specific raw column names (e.g. "SYN Flag
    # Count") that must still exist under those names at this point - some
    # get removed as duplicates below (e.g. "SYN Flag Count" is an exact
    # duplicate of "Fwd PSH Flags" across the full dataset), so dedup must
    # run after, not before.
    df = engineer_derived_features(df)
    df = drop_duplicate_columns(df)
    df = drop_known_redundant_columns(df)
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


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are exact, bit-for-bit duplicates of an earlier column.

    CICIDS2017's raw CSVs genuinely repeat "Fwd Header Length" as a column
    header; pandas silently renames the second occurrence to
    "Fwd Header Length.1" instead of erroring, so an exact duplicate
    otherwise survives into the cleaned data. Columns are grouped by a hash
    of their values first, so only genuine hash collisions need a full
    equality check, rather than comparing every column against every other
    column (which would be O(n^2) over 2.8M rows).
    """
    seen_hashes: dict[int, str] = {}
    duplicate_cols = []
    for column in df.columns:
        column_hash = pd.util.hash_pandas_object(df[column], index=False).sum()
        if column_hash in seen_hashes and df[column].equals(df[seen_hashes[column_hash]]):
            duplicate_cols.append(column)
        else:
            seen_hashes[column_hash] = column
    return df.drop(columns=duplicate_cols)


def drop_known_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop CICIDS2017 columns confirmed to duplicate another column's
    calculation, up to floating-point noise.

    "Avg Bwd/Fwd Segment Size" and "Bwd/Fwd Packet Length Mean" are
    documented CICFlowMeter quirks - two different names for the same
    underlying computation (correlation 1.000000, max absolute difference
    ~1e-6 and ~1e-10 respectively - see notebooks/01_eda.ipynb Section 3).

    Unlike drop_duplicate_columns(), this pair can't be detected
    generically without risking false positives: some CICIDS2017 column
    pairs are highly correlated but genuinely different (e.g. "Subflow Fwd
    Bytes" vs "Total Length of Fwd Packets", correlation 0.999999 but a
    real difference of up to ~30,000 on some rows), so this list is
    hardcoded from a specific, verified finding rather than inferred.
    """
    columns_to_drop = ["Avg Bwd Segment Size", "Avg Fwd Segment Size"]
    return df.drop(columns=[c for c in columns_to_drop if c in df.columns])


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
    df = drop_duplicate_feature_rows(df)
    df = encode_categorical_columns(df, UNSW_CATEGORICAL_COLUMNS, UNSW_ENCODERS_PATH)
    df = drop_constant_columns(df)
    return df


# --------------------------------------------------------------------------
# Common feature set for cross-dataset validation (Phase 5)
# --------------------------------------------------------------------------
# CICIDS2017 (CICFlowMeter) and UNSW-NB15 (Argus) compute entirely
# different feature sets under different names, so a model trained on
# CICIDS2017's 25 selected features cannot run on UNSW-NB15 at all - most
# of those columns don't exist there. These five, however, measure the
# same underlying quantity in both datasets and can be mapped onto one
# shared schema, which is what makes a genuine (if smaller) cross-dataset
# comparison possible. fwd_bwd_ratio and bytes_per_sec are recomputed the
# same way for both, since they're derived from these five, not native to
# either capture tool.
def build_cicids_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map CICIDS2017 columns onto the shared cross-dataset feature schema.

    "Flow Duration" is converted from microseconds to seconds so it's on
    the same scale as UNSW-NB15's "dur" - confirmed to already be in
    seconds (its max value tops out just under 60).
    """
    common = pd.DataFrame(
        {
            "duration": df["Flow Duration"] / 1_000_000,
            "fwd_packets": df["Total Fwd Packets"],
            "bwd_packets": df["Total Backward Packets"],
            "fwd_bytes": df["Total Length of Fwd Packets"],
            "bwd_bytes": df["Total Length of Bwd Packets"],
        }
    )
    common["fwd_bwd_ratio"] = common["fwd_packets"] / (common["bwd_packets"] + 1)
    common["bytes_per_sec"] = (common["fwd_bytes"] + common["bwd_bytes"]) / common[
        "duration"
    ].clip(lower=MIN_DURATION_SEC)
    common["Label"] = df["Label"]
    return common


def build_unsw_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map UNSW-NB15 columns onto the shared cross-dataset feature schema.

    No unit conversion needed here - "dur" is already in seconds.
    """
    common = pd.DataFrame(
        {
            "duration": df["dur"],
            "fwd_packets": df["spkts"],
            "bwd_packets": df["dpkts"],
            "fwd_bytes": df["sbytes"],
            "bwd_bytes": df["dbytes"],
        }
    )
    common["fwd_bwd_ratio"] = common["fwd_packets"] / (common["bwd_packets"] + 1)
    common["bytes_per_sec"] = (common["fwd_bytes"] + common["bwd_bytes"]) / common[
        "duration"
    ].clip(lower=MIN_DURATION_SEC)
    common["Label"] = df["Label"]
    return common


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

    print("Building common cross-dataset feature set...")
    cicids_common_df = build_cicids_common_features(cicids_df)
    cicids_common_df.to_parquet(CICIDS_COMMON_OUTPUT_PATH, index=False)
    print(f"  CICIDS2017 common features saved to {CICIDS_COMMON_OUTPUT_PATH}")

    unsw_common_df = build_unsw_common_features(unsw_df)
    unsw_common_df.to_parquet(UNSW_COMMON_OUTPUT_PATH, index=False)
    print(f"  UNSW-NB15 common features saved to {UNSW_COMMON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
