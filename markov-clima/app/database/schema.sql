CREATE TABLE IF NOT EXISTS series (
    key   TEXT NOT NULL,
    date  TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (key, date)
);

CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    start_date TEXT NOT NULL,
    end_date   TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
