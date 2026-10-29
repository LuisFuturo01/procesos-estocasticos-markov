"""Precios diarios: CoinGecko (BTC/ETH) o Frankfurter (tipo de cambio).

Orden de prioridad: caché SQLite -> API -> data.json local (respaldo sin internet).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from app.database import cache

ASSETS = {
    "Bitcoin (BTC/USD)": ("cg", "bitcoin"),
    "Ethereum (ETH/USD)": ("cg", "ethereum"),
    "USD/EUR": ("fx", "EUR"),
    "USD/GBP": ("fx", "GBP"),
    "USD/JPY": ("fx", "JPY"),
}
# El plan público de CoinGecko solo entrega ~365 días hacia atrás.
CG_MAX_DAYS = 364
STATIC_PATH = Path(__file__).resolve().parents[2] / "data.json"


def _ts(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())


def _fetch(kind: str, ident: str, start: str, end: str) -> pd.DataFrame:
    if kind == "cg":
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{ident}/market_chart/range",
                         params={"vs_currency": "usd", "from": _ts(start), "to": _ts(end) + 86400},
                         timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["prices"], columns=["ms", "value"])
        df["date"] = pd.to_datetime(df["ms"], unit="ms").dt.strftime("%Y-%m-%d")
        # Un precio por día: el de las 00:00 UTC (el último punto del día actual es intradía).
        return df.drop_duplicates("date", keep="first")[["date", "value"]]
    r = requests.get(f"https://api.frankfurter.dev/v1/{start}..{end}",
                     params={"base": "USD", "symbols": ident}, timeout=30)
    r.raise_for_status()
    rates = r.json()["rates"]
    return pd.DataFrame({"date": list(rates), "value": [v[ident] for v in rates.values()]})


def min_start(asset: str, today: date | None = None) -> date | None:
    """Fecha más antigua que la API gratuita permite consultar (None = sin límite)."""
    today = today or date.today()
    return today - timedelta(days=CG_MAX_DAYS) if ASSETS[asset][0] == "cg" else None


# ---------------------------------------------------------------------------
# Datos locales (data.json)
# ---------------------------------------------------------------------------

def has_static_data(asset: str | None = None) -> bool:
    if not STATIC_PATH.is_file():
        return False
    return asset is None or ASSETS[asset][1] in json.loads(STATIC_PATH.read_text(encoding="utf-8"))


def load_static_prices(asset: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    data = json.loads(STATIC_PATH.read_text(encoding="utf-8"))[ASSETS[asset][1]]
    df = pd.DataFrame({"date": pd.to_datetime(data["date"]), "value": data["value"]})
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df.dropna().reset_index(drop=True)


def static_date_range(asset: str) -> tuple[date, date]:
    df = load_static_prices(asset)
    return df["date"].min().date(), df["date"].max().date()


def build_static_snapshot(fx_years: int = 3) -> Path:
    """Descarga una vez todas las series y las guarda en data.json."""
    end = date.today() - timedelta(days=1)
    out = {"_generated": datetime.now(timezone.utc).isoformat()}
    for asset, (kind, ident) in ASSETS.items():
        start = end - timedelta(days=CG_MAX_DAYS - 1 if kind == "cg" else 365 * fx_years)
        df = _fetch(kind, ident, str(start), str(end))
        out[ident] = {"asset": asset, "date": df["date"].astype(str).tolist(),
                      "value": [round(float(v), 6) for v in df["value"]]}
    STATIC_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return STATIC_PATH


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_prices(asset: str, start: str, end: str, force: bool = False,
               use_static: bool = False) -> pd.DataFrame:
    """DataFrame [date, value] de precios diarios; df.attrs["source"] indica el origen."""
    if use_static:
        df = load_static_prices(asset, start, end)
        df.attrs["source"] = "static"
        return df
    kind, ident = ASSETS[asset]
    key = f"cripto:{kind}:{ident}"
    source = "cache"
    if force or not cache.covers(key, start, end):
        try:
            cache.save(key, _fetch(kind, ident, start, end), start, end)
            source = "api"
        except Exception:
            if has_static_data(asset):
                df = load_static_prices(asset, start, end)
                df.attrs["source"] = "static-fallback"
                return df
            raise
    df = cache.load(key, start, end).dropna().reset_index(drop=True)
    df.attrs["source"] = source
    return df


def daily_returns(prices: pd.Series) -> pd.Series:
    """R_t = (Price_t - Price_{t-1}) / Price_{t-1}."""
    return prices.pct_change().dropna()
