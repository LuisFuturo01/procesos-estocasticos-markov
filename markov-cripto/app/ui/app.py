import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.core.discretize import (  # noqa: E402
    COLORS, LABELS, SHORT, THRESHOLD_MODES, returns_to_states, rules_latex,
    tercile_thresholds, volatility_thresholds,
)
from app.services.data_fetcher import (  # noqa: E402
    ASSETS, daily_returns, get_prices, has_static_data, min_start, static_date_range,
)
from app.ui.components import Domain, chart, render_analysis  # noqa: E402

st.set_page_config(page_title="Markov · Cripto", page_icon="🪙", layout="wide")
st.title("🪙 Analizador estocástico de mercado")
st.markdown("¿Después de un día de caída, qué tan probable es otro día de caída? Clasificamos cada "
            "retorno diario como **bajista, lateral o alcista** y lo modelamos como una **cadena de "
            "Markov en tiempo discreto**, siguiendo el **Capítulo 3 de DAT262**.")

with st.sidebar:
    st.header("⚙️ Parámetros")
    st.subheader("1 · Datos")
    asset = st.selectbox("Activo", list(ASSETS))
    kind = ASSETS[asset][0]
    local_ok = has_static_data(asset)
    fuente = st.radio("Origen de datos", ["En vivo (API)", "Local (data.json)"] if local_ok else ["En vivo (API)"],
                      help="data.json es una copia descargada para trabajar sin internet o si la API "
                           "limita las consultas.")
    usar_estatica = fuente.startswith("Local")
    if usar_estatica:
        min_d, max_d = static_date_range(asset)
        rango = st.date_input("Rango de fechas", (min_d, max_d), min_value=min_d, max_value=max_d)
        refrescar = False
    else:
        hoy = date.today()
        lo = min_start(asset, hoy)
        default_start = lo if lo else hoy - timedelta(days=3 * 365)
        rango = st.date_input("Rango de fechas", (default_start, hoy - timedelta(days=1)),
                              min_value=lo, max_value=hoy)
        refrescar = st.checkbox("Ignorar caché local")
        if lo:
            st.caption("CoinGecko gratuito permite solo los últimos 365 días.")

    st.subheader("2 · Estados")
    modo = st.radio("Umbrales", THRESHOLD_MODES, index=0 if kind == "cg" else 1,
                    help="Fijo: el criterio clásico ±1.5 %. Volatilidad: se adapta al activo "
                         "(útil para divisas, que casi nunca se mueven ±1.5 %). Terciles: cada estado "
                         "ocupa 1/3 de los días.")
    if modo == THRESHOLD_MODES[0]:
        bajista = st.number_input("Umbral bajista (%)", -10.0, 0.0, -1.5, 0.1) / 100
        alcista = st.number_input("Umbral alcista (%)", 0.0, 10.0, 1.5, 0.1) / 100
    elif modo == THRESHOLD_MODES[1]:
        z = st.slider("z (múltiplo de la desviación típica σ)", 0.1, 2.0, 0.5, 0.1)

    st.subheader("3 · Predicción")
    n_steps = st.slider("Horizonte n (días)", 1, 30, 7)
    inicio = st.selectbox("Estado de hoy (π₀)", range(3), format_func=lambda i: LABELS[i], index=1)

if len(rango) != 2:
    st.info("Selecciona fecha inicial y final.")
    st.stop()

try:
    with st.spinner("Consultando datos…"):
        df = get_prices(asset, str(rango[0]), str(rango[1]), force=refrescar, use_static=usar_estatica)
except Exception as exc:
    st.error(f"No se pudo obtener datos: {exc}")
    st.stop()

src = df.attrs.get("source", "")
if src == "static-fallback":
    st.warning("La API no respondió (posible límite de consultas): se usan los datos locales de data.json.")
st.caption(f"📈 {asset} · {rango[0]} → {rango[1]} · fuente: "
           + {"static": "data.json local", "api": "API (descargado ahora)", "cache": "API (caché SQLite)",
              "static-fallback": "data.json local"}.get(src, src)
           + " · Herramienta educativa: no es una recomendación de inversión.")

df["date"] = pd.to_datetime(df["date"])
ret = daily_returns(df.set_index("date")["value"])
if len(ret) < 30:
    st.warning("Muy pocos datos para estimar P (se requieren al menos 30 retornos).")
    st.stop()

if modo == THRESHOLD_MODES[1]:
    bajista, alcista = volatility_thresholds(ret, z)
elif modo == THRESHOLD_MODES[2]:
    bajista, alcista = tercile_thresholds(ret)
if modo != THRESHOLD_MODES[0]:
    st.sidebar.caption(f"Umbrales calculados: {bajista * 100:.2f} % y {alcista * 100:+.2f} %")

states = returns_to_states(ret, bajista, alcista)


def data_view():
    fig = go.Figure(go.Scatter(x=df["date"], y=df["value"], mode="lines", name="Precio",
                               line=dict(color="#4F8BF9", width=2),
                               hovertemplate="%{x|%d %b %Y}: %{y:,.4f}<extra></extra>"))
    fig.update_yaxes(title="Precio")
    chart(fig, title=f"Precio diario · {asset}", height=300, showlegend=False)

    fig = go.Figure()
    for i in range(3):
        mask = states == i
        fig.add_bar(x=ret.index[mask], y=ret.values[mask], name=LABELS[i], marker_color=COLORS[i],
                    hovertemplate="%{x|%d %b %Y}: %{y:.2%}<extra>" + SHORT[i] + "</extra>")
    for u, c in ((bajista, COLORS[0]), (alcista, COLORS[2])):
        fig.add_hline(y=u, line=dict(color=c, dash="dash", width=1),
                      annotation_text=f"{u * 100:+.2f} %", annotation_position="top left")
    fig.update_yaxes(title="Retorno diario Rₜ", tickformat=".1%")
    chart(fig, title="Retornos Rₜ coloreados por estado Xₜ (líneas: umbrales)", height=340, bargap=0)

    fig = go.Figure(go.Histogram(x=ret.values, nbinsx=60, marker_color="#8B95A5", name="Rₜ",
                                 hovertemplate="%{x:.2%}: %{y} días<extra></extra>"))
    for u, c in ((bajista, COLORS[0]), (alcista, COLORS[2])):
        fig.add_vline(x=u, line=dict(color=c, dash="dash", width=2))
    fig.update_xaxes(tickformat=".1%", title="Retorno diario")
    fig.update_yaxes(title="Días")
    chart(fig, title="Distribución de retornos y cortes de discretización", height=300, showlegend=False)
    cnt = np.bincount(states, minlength=3)
    if cnt.min() < 10:
        st.warning(f"El estado {LABELS[int(cnt.argmin())]} tiene solo {cnt.min()} día{"s" if cnt.min() != 1 else ""}: su fila de P "
                   "será poco confiable. Prueba umbrales por **Volatilidad** o **Terciles**.")


dom = Domain(
    labels=LABELS, short=SHORT, colors=COLORS,
    # Las divisas no cotizan el fin de semana: un paso es un día hábil.
    step="día" if kind == "cg" else "día hábil", steps="días" if kind == "cg" else "días hábiles",
    rules_latex=rules_latex(bajista, alcista),
    omega=f"todas las trayectorias posibles del precio de {asset}.",
    variable="Rₜ = retorno diario simple del precio de cierre.",
    reading=["el precio cayó más que el umbral bajista.",
             "variación pequeña (mercado sin dirección clara).",
             "el precio subió más que el umbral alcista."],
    markov_hint="En mercados financieros es habitual por la **agrupación de volatilidad** "
                "(días agitados tienden a seguir a días agitados).",
)
render_analysis(states, dom, n_steps, inicio, data_view=data_view)
