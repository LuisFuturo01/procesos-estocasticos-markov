import numpy as np
import pandas as pd

from app.core import markov_engine as me
from app.core.discretize import (
    returns_to_states, tercile_thresholds, volatility_thresholds,
)
from app.services.data_fetcher import ASSETS, daily_returns, has_static_data, load_static_prices


def test_discretizacion_por_umbrales():
    r = [-0.02, -0.015, 0.0, 0.015, 0.02]
    assert returns_to_states(r).tolist() == [0, 1, 1, 1, 2]


def test_retorno_simple():
    r = daily_returns(pd.Series([100.0, 110.0, 99.0]))
    assert np.allclose(r.values, [0.10, -0.10])


def test_umbrales_adaptativos():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.004, 3000)  # volatilidad típica de una divisa
    assert np.bincount(returns_to_states(r), minlength=3)[[0, 2]].sum() < 0.01 * len(r)
    lo, hi = tercile_thresholds(r)
    assert np.allclose(np.bincount(returns_to_states(r, lo, hi)) / len(r), 1 / 3, atol=0.01)
    lo, hi = volatility_thresholds(r, 0.5)
    assert lo < 0 < hi and np.isclose(hi - lo, r.std(ddof=1), rtol=1e-9)


def test_datos_locales_y_pipeline():
    assert has_static_data()
    for asset in ASSETS:
        df = load_static_prices(asset)
        assert len(df) > 200 and df["date"].is_monotonic_increasing
        s = returns_to_states(daily_returns(df["value"]), *tercile_thresholds(daily_returns(df["value"])))
        res = me.analyze(s)
        assert np.allclose(res["P"].sum(axis=1), 1) and not res["unobserved"]
