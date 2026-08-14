"""
app.py

Flask dashboard for RAKSHAK: reads detector.py's SQLite log (detections.db)
and serves it as a live-refreshing web page - a connection table, an
alerts panel, and summary cards.

Deliberately simple, not over-engineered for this project's scale: each
route opens its own short-lived SQLite connection rather than using
Flask's `g`/teardown pattern (that pattern earns its complexity under
real concurrent request load; a single-user local dashboard polling every
5 seconds doesn't need it). The frontend uses plain JavaScript polling
(fetch() on a setInterval()) rather than WebSockets, for the same reason -
polling is simple, well-understood, and entirely sufficient here.

Run via: python src/app.py
(No sudo needed - unlike capture.py/detector.py, this only reads
detections.db, it never touches raw packets.)
"""

import sqlite3
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import DB_PATH, init_db  # noqa: E402

RECENT_LIMIT = 50
ALERTS_LIMIT = 50

app = Flask(__name__)

# Ensures the detections table exists even if the dashboard is started
# before detector.py has ever run - otherwise the very first /api/recent
# request would fail with "no such table".
init_db(DB_PATH).close()


def get_db_connection() -> sqlite3.Connection:
    """Open a fresh connection for one request, with row_factory set so
    each row can be converted straight into a JSON-serializable dict.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    """Serve the dashboard page itself - an empty skeleton that its own
    JavaScript fills in by calling the /api/* routes below.
    """
    return render_template("dashboard.html")


@app.route("/api/recent")
def api_recent():
    """The most recent flows, any label - powers the connection table."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?", (RECENT_LIMIT,)
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


@app.route("/api/alerts")
def api_alerts():
    """The most recent non-Normal flows only - powers the alerts panel."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM detections WHERE label != 'Normal' ORDER BY timestamp DESC LIMIT ?",
        (ALERTS_LIMIT,),
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


@app.route("/api/stats")
def api_stats():
    """Summary counts - powers the summary cards: total flows seen, a
    count per label, and how many of those were alerts (non-Normal).
    """
    conn = get_db_connection()
    total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    per_label_rows = conn.execute(
        "SELECT label, COUNT(*) AS count FROM detections GROUP BY label"
    ).fetchall()
    conn.close()

    per_label = {row["label"]: row["count"] for row in per_label_rows}
    alert_count = sum(count for label, count in per_label.items() if label != "Normal")

    return jsonify({"total": total, "per_label": per_label, "alert_count": alert_count})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
