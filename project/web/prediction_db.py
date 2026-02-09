"""
Simple SQL (SQLite) storage for prediction history.

Default DB path: project/data/predictions.sqlite3

Env vars:
  - SQLITE_DB_PATH / DB_PATH: override sqlite file path
  - DATABASE_URL: supports "sqlite:////abs/path" or "sqlite:///relative/path"
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _workspace_default_db_path() -> Path:
    # web/ -> project/ -> data/
    return (Path(__file__).resolve().parent.parent / "data" / "predictions.sqlite3").resolve()


def _resolve_db_path() -> Path:
    db_path = os.getenv("SQLITE_DB_PATH") or os.getenv("DB_PATH")
    if db_path:
        return Path(db_path).expanduser().resolve()

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if database_url.startswith("sqlite:///"):
            # sqlite:////abs/path OR sqlite:///relative/path
            raw = database_url[len("sqlite:///") :]
            if raw.startswith("/"):
                return Path("/" + raw.lstrip("/")).resolve()
            return Path(raw).expanduser().resolve()
        raise ValueError("Only sqlite DATABASE_URL is supported (sqlite:///...)")

    return _workspace_default_db_path()


def _connect() -> sqlite3.Connection:
    path = _resolve_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Better concurrency for a single-file DB
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT NOT NULL,
              symbol TEXT NOT NULL,
              current_price REAL,
              predicted_price REAL,
              prediction_pct REAL,
              direction TEXT,
              confidence REAL,
              signal_strength REAL,
              funding_rate REAL,
              vzo_15m REAL,
              slope_15m REAL,
              vzo_signal_15m TEXT,
              trade_action TEXT,
              raw_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol_created_at ON predictions(symbol, created_at);")

        # Migration: add new columns to existing tables (safe to run repeatedly)
        _migrate_add_vzo_columns(conn)


def _migrate_add_vzo_columns(conn: sqlite3.Connection) -> None:
    """Add VZO/slope columns if they don't exist yet (for existing DBs)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    new_cols = [
        ("vzo_15m", "REAL"),
        ("slope_15m", "REAL"),
        ("vzo_signal_15m", "TEXT"),
        ("trade_action", "TEXT"),
    ]
    for col_name, col_type in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"  Migration: added column '{col_name}' to predictions table")


@dataclass(frozen=True)
class StoredPrediction:
    id: int
    timestamp: str
    symbol: str
    current_price: float
    predicted_price: float
    prediction_pct: float
    direction: str
    confidence: float
    signal_strength: float
    funding_rate: float
    vzo_15m: Optional[float]
    slope_15m: Optional[float]
    vzo_signal_15m: Optional[str]
    trade_action: Optional[str]
    raw: Dict[str, Any]
    created_at: str


def save_prediction(
    *,
    result: Dict[str, Any],
    predicted_price: float,
    vzo_15m: Optional[float] = None,
    slope_15m: Optional[float] = None,
    vzo_signal_15m: Optional[str] = None,
    trade_action: Optional[str] = None,
) -> int:
    """
    Persist a PredictionServer.predict() result.

    Returns inserted row id.
    """
    now = datetime.now(timezone.utc).isoformat()
    created_at = now
    timestamp = str(result.get("timestamp") or created_at)
    symbol = str(result.get("symbol") or "UNKNOWN")

    current_price = float(result.get("current_price") or 0.0)
    prediction_pct = float(result.get("prediction_pct") or 0.0)
    direction = str(result.get("direction") or "")
    confidence = float(result.get("confidence") or 0.0)
    funding_rate = float(result.get("funding_rate") or 0.0)
    signal_strength = float((result.get("features_summary") or {}).get("signal_strength") or 0.0)

    raw_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
              timestamp, symbol, current_price, predicted_price, prediction_pct,
              direction, confidence, signal_strength, funding_rate,
              vzo_15m, slope_15m, vzo_signal_15m, trade_action,
              raw_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                symbol,
                current_price,
                float(predicted_price),
                prediction_pct,
                direction,
                confidence,
                signal_strength,
                funding_rate,
                vzo_15m,
                slope_15m,
                vzo_signal_15m,
                trade_action,
                raw_json,
                created_at,
            ),
        )
        return int(cur.lastrowid)


def fetch_recent_predictions(limit: int = 50, symbol: Optional[str] = None) -> List[StoredPrediction]:
    limit = max(1, min(int(limit), 500))
    with _connect() as conn:
        if symbol:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (symbol.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    items: List[StoredPrediction] = []
    for r in rows:
        raw = json.loads(r["raw_json"]) if r["raw_json"] else {}
        items.append(
            StoredPrediction(
                id=int(r["id"]),
                timestamp=str(r["timestamp"]),
                symbol=str(r["symbol"]),
                current_price=float(r["current_price"] or 0.0),
                predicted_price=float(r["predicted_price"] or 0.0),
                prediction_pct=float(r["prediction_pct"] or 0.0),
                direction=str(r["direction"] or ""),
                confidence=float(r["confidence"] or 0.0),
                signal_strength=float(r["signal_strength"] or 0.0),
                funding_rate=float(r["funding_rate"] or 0.0),
                vzo_15m=float(r["vzo_15m"]) if r["vzo_15m"] is not None else None,
                slope_15m=float(r["slope_15m"]) if r["slope_15m"] is not None else None,
                vzo_signal_15m=str(r["vzo_signal_15m"]) if r["vzo_signal_15m"] else None,
                trade_action=str(r["trade_action"]) if r["trade_action"] else None,
                raw=raw,
                created_at=str(r["created_at"]),
            )
        )
    return items


@dataclass(frozen=True)
class VzoSlope:
    """Lightweight record for VZO/slope history used by TradeSignalManager."""
    timestamp: str
    vzo_15m: float
    slope_15m: float
    vzo_signal_15m: Optional[str]
    trade_action: Optional[str]


def fetch_recent_vzo_slopes(limit: int = 20, symbol: Optional[str] = None) -> List[VzoSlope]:
    """
    Fetch recent VZO/slope values from stored predictions (oldest-first).

    Returns list of VzoSlope ordered by id ASC (oldest first) so that
    the last element is the most recent.
    """
    limit = max(1, min(int(limit), 200))
    with _connect() as conn:
        if symbol:
            rows = conn.execute(
                """
                SELECT timestamp, vzo_15m, slope_15m, vzo_signal_15m, trade_action
                FROM predictions
                WHERE symbol = ? AND vzo_15m IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (symbol.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT timestamp, vzo_15m, slope_15m, vzo_signal_15m, trade_action
                FROM predictions
                WHERE vzo_15m IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    items: List[VzoSlope] = []
    for r in rows:
        items.append(
            VzoSlope(
                timestamp=str(r["timestamp"]),
                vzo_15m=float(r["vzo_15m"]),
                slope_15m=float(r["slope_15m"]) if r["slope_15m"] is not None else 0.0,
                vzo_signal_15m=str(r["vzo_signal_15m"]) if r["vzo_signal_15m"] else None,
                trade_action=str(r["trade_action"]) if r["trade_action"] else None,
            )
        )
    # Reverse so oldest is first, newest is last
    items.reverse()
    return items

