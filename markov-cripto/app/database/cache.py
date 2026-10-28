"""Caché local SQLite de series diarias (el esquema vive en schema.sql)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).parent
DB_PATH = _DIR / "cache.sqlite3"


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript((_DIR / "schema.sql").read_text(encoding="utf-8"))
    return con


def covers(key: str, start: str, end: str, path: Path = DB_PATH) -> bool:
    with connect(path) as con:
        r = con.execute("SELECT start_date, end_date FROM meta WHERE key=?", (key,)).fetchone()
    return bool(r and r[0] <= start and r[1] >= end)


def save(key: str, df: pd.DataFrame, start: str, end: str, path: Path = DB_PATH) -> None:
    rows = [(key, str(d), None if pd.isna(v) else float(v)) for d, v in zip(df["date"], df["value"])]
    with connect(path) as con:
        con.executemany("INSERT OR REPLACE INTO series VALUES (?,?,?)", rows)
        old = con.execute("SELECT start_date, end_date FROM meta WHERE key=?", (key,)).fetchone()
        s = min(start, old[0]) if old else start
        e = max(end, old[1]) if old else end
        con.execute("INSERT OR REPLACE INTO meta VALUES (?,?,?,?)",
                    (key, s, e, datetime.now(timezone.utc).isoformat()))


def load(key: str, start: str, end: str, path: Path = DB_PATH) -> pd.DataFrame:
    with connect(path) as con:
        df = pd.read_sql_query(
            "SELECT date, value FROM series WHERE key=? AND date BETWEEN ? AND ? ORDER BY date",
            con, params=(key, start, end))
    df["date"] = pd.to_datetime(df["date"])
    return df
