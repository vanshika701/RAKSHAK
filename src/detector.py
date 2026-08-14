"""
detector.py

Live intrusion detection: sniffs traffic (reusing capture.py's FlowManager
and extract_features()), classifies each finished flow with the trained
ensemble, and logs every result to SQLite - printing prominently only for
non-Normal predictions, so the console isn't flooded by routine DNS/HTTPS
traffic while still keeping a complete record for later review.

Run via: sudo python src/detector.py
(sudo is required - same as capture.py, raw packet capture needs elevated
privileges.)
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from scapy.all import sniff

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture import Flow, FlowManager, extract_features, true_forward_endpoint  # noqa: E402
from src.train_model import (  # noqa: E402
    MODELS_DIR,
    SCALER_PATH,
    SELECTED_FEATURES_PATH,
    U2R_DECISION_THRESHOLD,
    ThresholdedEnsemble,
    build_soft_voting_ensemble,
    load_trained_models,
)

DB_PATH = MODELS_DIR.parent / "detections.db"


def load_detector() -> tuple[ThresholdedEnsemble, object, list[str]]:
    """Load the trained ensemble (with the U2R confidence threshold
    applied), scaler, and selected feature list - everything inference
    needs, built entirely from train_model.py's own saved artifacts so
    swapping in new tuned models later needs no changes here.
    """
    models = load_trained_models()
    ensemble = build_soft_voting_ensemble([models["rf"], models["xgb"], models["lgbm"]])
    thresholded = ThresholdedEnsemble(
        ensemble, target_class="U2R", threshold=U2R_DECISION_THRESHOLD
    )

    scaler = joblib.load(SCALER_PATH)
    with open(SELECTED_FEATURES_PATH) as f:
        selected_features = json.load(f)

    return thresholded, scaler, selected_features


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create (if needed) and connect to the SQLite detections log."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            timestamp TEXT NOT NULL,
            src_ip TEXT NOT NULL,
            src_port INTEGER NOT NULL,
            dst_ip TEXT NOT NULL,
            dst_port INTEGER NOT NULL,
            protocol TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def classify_flow(
    flow: Flow, model: ThresholdedEnsemble, scaler, selected_features: list[str]
) -> tuple[str, float]:
    """Turn a finished flow into a prediction: extract features, reorder
    to match training's exact column order, scale, then predict.

    Column order matters here specifically because MinMaxScaler and the
    trained models only remember feature *positions*, not names - feeding
    them a differently-ordered row would silently scale/predict against
    the wrong columns without ever raising an error.

    Returns (label, confidence) - confidence is the predicted class's own
    probability from predict_proba(), not just the highest probability
    among all classes (though for an unthresholded prediction those are
    the same thing; ThresholdedEnsemble can override which class "wins",
    so looking up the actual predicted label's probability is what keeps
    this correct in both cases).
    """
    features = extract_features(flow)
    row = pd.DataFrame([features])[selected_features]
    scaled = pd.DataFrame(scaler.transform(row), columns=selected_features)

    label = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    class_index = list(model.classes_).index(label)
    confidence = float(proba[class_index])

    return label, confidence


def log_detection(
    conn: sqlite3.Connection,
    fwd_ip: str,
    fwd_port: int,
    bwd_ip: str,
    bwd_port: int,
    protocol: str,
    label: str,
    confidence: float,
) -> None:
    """Insert one classification result into the SQLite log."""
    conn.execute(
        "INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            fwd_ip,
            fwd_port,
            bwd_ip,
            bwd_port,
            protocol,
            label,
            confidence,
        ),
    )
    conn.commit()


def main() -> None:
    """Sniff traffic indefinitely, classifying and logging every finished
    flow - printing prominently only when the prediction isn't Normal.
    """
    model, scaler, selected_features = load_detector()
    conn = init_db(DB_PATH)
    manager = FlowManager()

    def handle_packet(packet) -> None:
        manager.process_packet(packet)
        for flow in manager.pop_expired_flows():
            label, confidence = classify_flow(flow, model, scaler, selected_features)

            fwd_ip, fwd_port = true_forward_endpoint(flow)
            bwd_ip, bwd_port = (
                (flow.backward_ip, flow.backward_port)
                if (fwd_ip, fwd_port) == (flow.forward_ip, flow.forward_port)
                else (flow.forward_ip, flow.forward_port)
            )

            log_detection(conn, fwd_ip, fwd_port, bwd_ip, bwd_port, flow.protocol, label, confidence)

            if label != "Normal":
                print(
                    f"[ALERT] {label} ({confidence:.2%} confidence) - "
                    f"{fwd_ip}:{fwd_port} -> {bwd_ip}:{bwd_port} ({flow.protocol})"
                )

    print(f"Detector running - logging to {DB_PATH}. Press Ctrl+C to stop.")
    sniff(prn=handle_packet, store=False)


if __name__ == "__main__":
    main()
