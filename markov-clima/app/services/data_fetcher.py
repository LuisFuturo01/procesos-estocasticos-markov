"""Precipitación diaria histórica desde Open-Meteo (sin API key) con caché SQLite."""
from __future__ import annotations

import pandas as pd
import requests

from app.database import cache

URL = "https://archive-api.open-meteo.com/v1/archive"

CITIES = {
    "La Paz, Bolivia": (-16.50, -68.15),
    "Santa Cruz, Bolivia": (-17.78, -63.18),
    "Cochabamba, Bolivia": (-17.39, -66.16),
    "Lima, Perú": (-12.05, -77.04),
    "Buenos Aires, Argentina": (-34.61, -58.38),
}


def get_precipitation(lat: float, lon: float, start: str, end: str, force: bool = False) -> pd.DataFrame:
    """DataFrame [date, value] con precipitación diaria (mm) entre start y end (YYYY-MM-DD)."""
    key = f"clima:{lat:.2f}:{lon:.2f}"
    if force or not cache.covers(key, start, end):
        r = requests.get(URL, timeout=30, params={
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "daily": "precipitation_sum", "timezone": "auto"})
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({"date": d["time"], "value": d["precipitation_sum"]})
        cache.save(key, df, start, end)
    return cache.load(key, start, end).dropna()
