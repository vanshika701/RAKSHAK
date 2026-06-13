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

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT = the RAKSHAK/ folder, computed relative to this file's own
# location so the script works no matter what directory it's run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CICIDS_DIR = DATA_RAW_DIR / "cicids2017"
UNSW_DIR = DATA_RAW_DIR / "unsw_nb15"

CICIDS_OUTPUT_PATH = DATA_PROCESSED_DIR / "cicids_clean.parquet"
UNSW_OUTPUT_PATH = DATA_PROCESSED_DIR / "unsw_clean.parquet"

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
            cleaned_chunks.append(chunk)

    return pd.concat(cleaned_chunks, ignore_index=True)


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


def load_unsw_nb15() -> pd.DataFrame:
    """Load and clean both UNSW-NB15 CSVs (training + testing) into one DataFrame.

    Both files are read in CHUNK_SIZE-row chunks using "utf-8-sig" encoding,
    which strips the BOM character at the start of the "id" column. For
    every chunk: column names are stripped, a unified "Label" column is
    built from "attack_cat" via UNSW_LABEL_MAP, and the non-feature /
    leakage columns ("id", "label", "attack_cat") are dropped.
    """
    csv_files = sorted(UNSW_DIR.glob("*.csv"))
    cleaned_chunks = []

    for csv_path in csv_files:
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, encoding="utf-8-sig"):
            chunk = clean_column_names(chunk)
            chunk["Label"] = chunk["attack_cat"].map(UNSW_LABEL_MAP)
            chunk = chunk.drop(columns=["id", "label", "attack_cat"])
            cleaned_chunks.append(chunk)

    return pd.concat(cleaned_chunks, ignore_index=True)
