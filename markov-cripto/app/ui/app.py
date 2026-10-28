import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.core.discretize import LABELS, returns_to_states  # noqa: E402
from app.services.data_fetcher import ASSETS, get_prices  # noqa: E402
from app.ui.components import render_analysis  # noqa: E402

st.set_page_config(page_title="Markov · Cripto", page_icon="🪙", layout="wide")
st.title("🪙 Analizador estocástico de mercado")
st.caption("Cadena de Markov en tiempo discreto · DAT262 · Datos: CoinGecko / Frankfurter")

with st.sidebar:
    st.header("Parámetros")
    asset = st.selectbox("Activo", list(ASSETS))
    hoy = date.today()
    rango = st.date_input("Rango de fechas", (hoy - timedelta(days=360), hoy), max_value=hoy)
    bajista = st.number_input("Umbral bajista (%)", -10.0, 0.0, -1.5, 0.1) / 100
    alcista = st.number_input("Umbral alcista (%)", 0.0, 10.0, 1.5, 0.1) / 100
    n_steps = st.slider("Pasos de predicción n", 1, 30, 7)
    inicio = st.selectbox("Estado inicial π₀", range(3), format_func=lambda i: LABELS[i], index=1)
    refrescar = st.checkbox("Ignorar caché")
    st.caption("CoinGecko gratuito limita el histórico a ~365 días.")

if len(rango) != 2:
    st.info("Selecciona fecha inicial y final.")
    st.stop()

try:
    with st.spinner("Consultando datos..."):
        df = get_prices(asset, str(rango[0]), str(rango[1]), force=refrescar)
except Exception as exc:
    st.error(f"No se pudo obtener datos: {exc}")
    st.stop()

ret = df["value"].pct_change().dropna()
if len(ret) < 30:
    st.warning("Muy pocos datos para estimar P.")
    st.stop()

render_analysis(returns_to_states(ret, bajista, alcista), LABELS, n_steps, inicio)
