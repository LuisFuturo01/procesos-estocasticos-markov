"""Precios diarios: CoinGecko (BTC/ETH) o Frankfurter (tipo de cambio), con caché SQLite."""
from __future__ import annotations

from datetime import datetime, timezone

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


def _ts(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())


def _fetch(kind: str, ident: str, start: str, end: str) -> pd.DataFrame:
    if kind == "cg":
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{ident}/market_chart/range",
                         params={"vs_currency": "usd", "from": _ts(start), "to": _ts(end) + 86400},
                         timeout=30)
        r.raise_for_status()
        p = r.json()["prices"]
        df = pd.DataFrame(p, columns=["ms", "value"])
        df["date"] = pd.to_datetime(df["ms"], unit="ms").dt.strftime("%Y-%m-%d")
        return df.drop_duplicates("date", keep="last")[["date", "value"]]
    r = requests.get(f"https://api.frankfurter.dev/v1/{start}..{end}",
                     params={"base": "USD", "symbols": ident}, timeout=30)
    r.raise_for_status()
    rates = r.json()["rates"]
    return pd.DataFrame({"date": list(rates), "value": [v[ident] for v in rates.values()]})


def get_prices(asset: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
    """DataFrame [date, value] de precios diarios."""
    kind, ident = ASSETS[asset]
    key = f"cripto:{kind}:{ident}"
    if force or not cache.covers(key, start, end):
        cache.save(key, _fetch(kind, ident, start, end), start, end)
    return cache.load(key, start, end).dropna()
