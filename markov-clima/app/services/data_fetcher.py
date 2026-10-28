"""Precipitación diaria histórica desde data.json (estático) u Open-Meteo con caché SQLite."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

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


def get_static_file_path() -> Path | None:
    """Busca el archivo data.json en las rutas relativas usuales."""
    candidates = [
        Path(__file__).resolve().parents[2] / "data.json",
        Path(__file__).resolve().parents[1] / "data.json",
        Path.cwd() / "markov-clima" / "data.json",
        Path.cwd() / "data.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def has_static_data() -> bool:
    """Verifica si existe el archivo data.json estático."""
    return get_static_file_path() is not None


def get_static_date_range() -> tuple[date, date]:
    """Retorna la fecha mínima y máxima disponible en data.json."""
    p = get_static_file_path()
    if not p:
        hoy = date.today()
        return hoy.replace(year=hoy.year - 3), hoy

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    times = data.get("daily", {}).get("time", [])
    if not times:
        hoy = date.today()
        return hoy.replace(year=hoy.year - 3), hoy

    return date.fromisoformat(times[0]), date.fromisoformat(times[-1])


def load_static_precipitation(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Carga los datos estáticos desde data.json y filtra opcionalmente por rango de fechas."""
    p = get_static_file_path()
    if not p:
        raise FileNotFoundError("No se encontró el archivo data.json con datos estáticos.")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    daily = data.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "value": daily.get("precipitation_sum", [])
    })

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    return df.dropna().reset_index(drop=True)


def get_precipitation(lat: float, lon: float, start: str, end: str, force: bool = False, use_static: bool = False) -> pd.DataFrame:
    """DataFrame [date, value] con precipitación diaria (mm) entre start y end (YYYY-MM-DD)."""
    if use_static and has_static_data():
        return load_static_precipitation(start, end)

    key = f"clima:{lat:.2f}:{lon:.2f}"
    if force or not cache.covers(key, start, end):
        try:
            r = requests.get(URL, timeout=30, params={
                "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                "daily": "precipitation_sum", "timezone": "auto"})
            r.raise_for_status()
            d = r.json()["daily"]
            df = pd.DataFrame({"date": d["time"], "value": d["precipitation_sum"]})
            cache.save(key, df, start, end)
        except Exception:
            # Fallback a datos estáticos si la API falla y el archivo existe
            if has_static_data():
                return load_static_precipitation(start, end)
            raise
    return cache.load(key, start, end).dropna()
