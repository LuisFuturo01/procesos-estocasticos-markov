import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.core.discretize import (  # noqa: E402
    COLORS, LABELS, MONTHS, SEASONS, SHORT, precip_to_states, rules_latex,
)
from app.services.data_fetcher import (  # noqa: E402
    CITIES, consecutive_mask, get_precipitation, get_static_date_range, has_static_data,
)
from app.ui.components import Domain, chart, render_analysis  # noqa: E402

st.set_page_config(page_title="Markov · Clima", page_icon="🌦️", layout="wide")
st.title("🌦️ Analizador estocástico del clima")
st.markdown("¿Si hoy llovió, qué tan probable es que mañana también llueva? Modelamos la "
            "precipitación diaria como una **cadena de Markov en tiempo discreto** de 3 estados "
            "y aplicamos paso a paso la teoría del **Capítulo 3 de DAT262**.")

static_available = has_static_data()

with st.sidebar:
    st.header("⚙️ Parámetros")
    st.subheader("1 · Datos")
    if static_available:
        fuente = st.radio("Origen de datos", ["Local (data.json · La Paz)", "En vivo (Open-Meteo)"],
                          help="data.json trae 1 096 días de La Paz y funciona sin internet.")
        usar_estatica = fuente.startswith("Local")
    else:
        usar_estatica = False

    if usar_estatica:
        city = "La Paz, Bolivia"
        min_d, max_d = get_static_date_range()
        rango = st.date_input("Rango de fechas", (min_d, max_d), min_value=min_d, max_value=max_d)
        refrescar = False
    else:
        city = st.selectbox("Ciudad", list(CITIES))
        max_d = date.today() - timedelta(days=6)
        rango = st.date_input("Rango de fechas", (max_d - timedelta(days=3 * 365), max_d), max_value=max_d,
                              help="El archivo histórico de Open-Meteo tiene ~5 días de retraso.")
        refrescar = st.checkbox("Ignorar caché local")
    season = st.selectbox("Temporada", list(SEASONS),
                          help="P se supone constante en el tiempo (homogénea). En La Paz la lluvia es "
                               "muy estacional: estimar P solo con una temporada hace el modelo más fiel.")

    st.subheader("2 · Estados")
    leve = st.number_input("Umbral seco (mm)", 0.0, 10.0, 0.0, 0.1,
                           help="Días con precipitación ≤ este valor se consideran secos (S1).")
    intensa = st.number_input("Umbral lluvia intensa (mm)", 0.5, 100.0, 5.0, 0.5,
                              help="Días con precipitación ≥ este valor son lluvia intensa (S3).")
    if intensa <= leve:
        st.error("El umbral de lluvia intensa debe ser mayor que el umbral seco.")
        st.stop()

    st.subheader("3 · Predicción")
    n_steps = st.slider("Horizonte n (días)", 1, 30, 7)
    inicio = st.selectbox("Estado de hoy (π₀)", range(3), format_func=lambda i: LABELS[i])

if len(rango) != 2:
    st.info("Selecciona fecha inicial y final.")
    st.stop()

try:
    with st.spinner("Consultando datos…"):
        df = get_precipitation(*CITIES[city], str(rango[0]), str(rango[1]), force=refrescar,
                               use_static=usar_estatica, static_fallback=city.startswith("La Paz"))
except Exception as exc:
    st.error(f"No se pudo obtener datos de Open-Meteo para {city}: {exc}")
    st.stop()

src = df.attrs.get("source", "")
if src == "static-fallback":
    st.warning("Open-Meteo no respondió: se usan los datos locales de data.json (La Paz).")
st.caption(f"📍 {city} · {rango[0]} → {rango[1]} · fuente: "
           + {"static": "data.json local", "api": "Open-Meteo (descargado ahora)",
              "cache": "Open-Meteo (caché SQLite)", "static-fallback": "data.json local"}.get(src, src))

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.month.isin(SEASONS[season])].reset_index(drop=True)
if len(df) < 30:
    st.warning("Muy pocos datos para estimar P (se requieren al menos 30 días).")
    st.stop()

states = precip_to_states(df["value"], leve, intensa)
valid = consecutive_mask(df["date"])
df["estado"] = states


def data_view():
    fig = go.Figure()
    for i in range(3):
        sub = df[df["estado"] == i]
        fig.add_bar(x=sub["date"], y=np.maximum(sub["value"], 0.15), name=LABELS[i], marker_color=COLORS[i],
                    customdata=sub["value"], hovertemplate="%{x|%d %b %Y}: %{customdata:.1f} mm<extra>"
                    + SHORT[i] + "</extra>")
    fig.add_hline(y=intensa, line=dict(color=COLORS[2], dash="dash", width=1),
                  annotation_text=f"umbral intensa = {intensa} mm", annotation_position="top left")
    fig.update_yaxes(title="Precipitación (mm/día)")
    chart(fig, title="Serie observada Rₜ coloreada por estado Xₜ", height=360, bargap=0)

    m = df.assign(mes=df["date"].dt.month).groupby("mes")["estado"].value_counts(normalize=True)
    m = m.unstack(fill_value=0).reindex(columns=range(3), fill_value=0)
    fig = go.Figure()
    for i in range(3):
        fig.add_bar(x=[MONTHS[x - 1] for x in m.index], y=m[i], name=SHORT[i], marker_color=COLORS[i],
                    hovertemplate="%{x}: %{y:.0%}<extra>" + SHORT[i] + "</extra>")
    fig.update_yaxes(tickformat=".0%", title="% de días")
    chart(fig, barmode="stack", bargap=0.15, height=320,
          title="Estacionalidad: proporción de cada estado por mes")
    if season == "Todo el año":
        st.caption("Si las barras cambian mucho de un mes a otro, P no es la misma todo el año. "
                   "Prueba el filtro **Temporada** del panel lateral y compara las matrices.")
    else:
        st.caption("Con filtro de temporada, las transiciones entre días no consecutivos "
                   "(p. ej. 31-mar → 1-dic) se excluyen del conteo.")


dom = Domain(
    labels=LABELS, short=SHORT, colors=COLORS, step="día", steps="días",
    rules_latex=rules_latex(leve, intensa),
    omega="todas las secuencias posibles de lluvia diaria en " + city.split(",")[0] + ".",
    variable="Rₜ = precipitación acumulada del día t (mm), fuente Open-Meteo.",
    reading=["día sin lluvia (o por debajo del umbral seco).",
             "llovizna o lluvia moderada.",
             "evento de lluvia fuerte."],
    markov_hint="En La Paz suele deberse a la estacionalidad: con **Temporada = Época de lluvias** "
                "la prueba ya no rechaza H₀, porque dentro de una temporada P sí es aproximadamente constante.",
)
render_analysis(states, dom, n_steps, inicio, valid=valid, data_view=data_view)
