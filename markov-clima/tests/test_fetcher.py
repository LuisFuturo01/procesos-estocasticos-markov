from datetime import date
import pandas as pd
from app.services.data_fetcher import (
    has_static_data,
    get_static_date_range,
    load_static_precipitation,
)
from app.core.discretize import precip_to_states
from app.core import markov_engine as me


def test_static_data_exists_and_loads():
    assert has_static_data(), "data.json debe existir y ser detectado"
    min_d, max_d = get_static_date_range()
    assert min_d == date(2023, 10, 23)
    assert max_d == date(2026, 10, 22)

    df = load_static_precipitation()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1096
    assert set(df.columns) == {"date", "value"}


def test_static_data_date_filter():
    df_filtered = load_static_precipitation(start="2024-01-01", end="2024-01-31")
    assert len(df_filtered) == 31


def test_static_data_markov_pipeline():
    df = load_static_precipitation()
    states = precip_to_states(df["value"], leve=0.0, intensa=5.0)
    assert len(states) == len(df)

    res = me.analyze(states, 3)
    assert res["P"].shape == (3, 3)
    assert (res["P"] >= 0).all()
    assert (res["pi"] >= 0).all()
    assert abs(res["pi"].sum() - 1.0) < 1e-6
