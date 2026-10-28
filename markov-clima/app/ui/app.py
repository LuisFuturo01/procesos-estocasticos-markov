import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.core.discretize import LABELS, precip_to_states  # noqa: E402
from app.services.data_fetcher import (  # noqa: E402
    CITIES,
    get_precipitation,
    get_static_date_range,
    has_static_data,
    load_static_precipitation,
)
from app.ui.components import render_analysis  # noqa: E402

st.set_page_config(page_title="Markov · Clima", page_icon="🌦️", layout="wide")
st.title("🌦️ Analizador estocástico del clima")

static_available = has_static_data()

with st.sidebar:
    st.header("Parámetros")

    if static_available:
        fuente = st.radio(
            "Origen de datos",
            ["Estática (data.json - La Paz)", "En vivo (Open-Meteo API)"],
            index=0,
            help="data.json contiene 1,096 registros diarios (2023 - 2026) de La Paz sin requerir conexión a internet.",
        )
        usar_estatica = fuente.startswith("Estática")
    else:
        usar_estatica = False

    if usar_estatica:
        st.info("📁 **data.json local** (La Paz)", icon="ℹ️")
        min_d, max_d = get_static_date_range()
        rango = st.date_input("Rango de fechas", (min_d, max_d), min_value=min_d, max_value=max_d)
    else:
        city = st.selectbox("Ciudad", list(CITIES))
        max_d = date.today() - timedelta(days=6)
        rango = st.date_input("Rango de fechas", (max_d - timedelta(days=3 * 365), max_d), max_value=max_d)
        refrescar = st.checkbox("Ignorar caché")

    leve = st.number_input("Umbral seco (mm)", 0.0, 10.0, 0.0, 0.1)
    intensa = st.number_input("Umbral lluvia intensa (mm)", 0.5, 100.0, 5.0, 0.5)
    n_steps = st.slider("Pasos de predicción n", 1, 30, 7)
    inicio = st.selectbox("Estado inicial π₀", range(3), format_func=lambda i: LABELS[i])

if usar_estatica:
    st.caption("Cadena de Markov en tiempo discreto · DAT262 · Datos estáticos: `data.json` (La Paz, 1,096 días)")
else:
    st.caption(f"Cadena de Markov en tiempo discreto · DAT262 · Datos en vivo: Open-Meteo ({city})")

if len(rango) != 2:
    st.info("Selecciona fecha inicial y final.")
    st.stop()

try:
    if usar_estatica:
        df = load_static_precipitation(str(rango[0]), str(rango[1]))
    else:
        with st.spinner("Consultando datos..."):
            df = get_precipitation(*CITIES[city], str(rango[0]), str(rango[1]), force=refrescar, use_static=False)
except Exception as exc:
    st.error(f"No se pudo obtener datos: {exc}")
    st.stop()

if len(df) < 30:
    st.warning("Muy pocos datos para estimar P (se requieren al menos 30 observaciones).")
    st.stop()

states = precip_to_states(df["value"], leve, intensa)
render_analysis(states, LABELS, n_steps, inicio)
