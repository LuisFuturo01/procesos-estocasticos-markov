import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.core.discretize import LABELS, precip_to_states  # noqa: E402
from app.services.data_fetcher import CITIES, get_precipitation  # noqa: E402
from app.ui.components import render_analysis  # noqa: E402

st.set_page_config(page_title="Markov · Clima", page_icon="🌦️", layout="wide")
st.title("🌦️ Analizador estocástico del clima")
st.caption("Cadena de Markov en tiempo discreto · DAT262 · Datos: Open-Meteo")

with st.sidebar:
    st.header("Parámetros")
    city = st.selectbox("Ciudad", list(CITIES))
    max_d = date.today() - timedelta(days=6)
    rango = st.date_input("Rango de fechas", (max_d - timedelta(days=3 * 365), max_d), max_value=max_d)
    leve = st.number_input("Umbral seco (mm)", 0.0, 10.0, 0.0, 0.1)
    intensa = st.number_input("Umbral lluvia intensa (mm)", 0.5, 100.0, 5.0, 0.5)
    n_steps = st.slider("Pasos de predicción n", 1, 30, 7)
    inicio = st.selectbox("Estado inicial π₀", range(3), format_func=lambda i: LABELS[i])
    refrescar = st.checkbox("Ignorar caché")

if len(rango) != 2:
    st.info("Selecciona fecha inicial y final.")
    st.stop()

try:
    with st.spinner("Consultando datos..."):
        df = get_precipitation(*CITIES[city], str(rango[0]), str(rango[1]), force=refrescar)
except Exception as exc:
    st.error(f"No se pudo obtener datos: {exc}")
    st.stop()

if len(df) < 30:
    st.warning("Muy pocos datos para estimar P.")
    st.stop()

states = precip_to_states(df["value"], leve, intensa)
render_analysis(states, LABELS, n_steps, inicio)
