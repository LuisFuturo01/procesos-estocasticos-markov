"""Componentes de visualización compartidos (Plotly + Streamlit).

La interfaz sigue el orden del Capítulo 3 de DAT262: datos -> estados -> P ->
propiedad de Markov -> P^n -> pi -> convergencia/ergodicidad. En cada paso se
muestra la fórmula, la fórmula evaluada con los datos y una interpretación.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.core import markov_engine as me


@dataclass
class Domain:
    """Lo que cambia entre proyectos: nombres, colores y textos del fenómeno."""
    labels: list[str]          # "S1 Seco", ...
    short: list[str]           # "Seco", ...
    colors: list[str]          # un color fijo por estado
    step: str                  # "día"
    steps: str                 # "días"
    rules_latex: str           # regla de discretización X_t = f(dato)
    omega: str                 # descripción del espacio muestral
    variable: str              # qué se observa cada paso
    reading: list[str] = field(default_factory=list)  # lectura de cada estado
    markov_hint: str = ""      # sugerencia si se rechaza la propiedad de Markov


LAYOUT = dict(margin=dict(l=10, r=10, t=50, b=10), hovermode="closest",
              legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))


# ---------------------------------------------------------------------------
# Helpers de formato
# ---------------------------------------------------------------------------

def mat_tex(M, fmt: str = "{:.3f}") -> str:
    rows = [" & ".join(fmt.format(x) for x in r) for r in np.atleast_2d(M)]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def vec_tex(v, fmt: str = "{:.4f}") -> str:
    return r"\left(" + r",\ ".join(fmt.format(x) for x in v) + r"\right)"


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f} %"


def fmt_time(x: float) -> str:
    return "∞" if not np.isfinite(x) else f"{x:.2f}"


def chart(fig: go.Figure, **kw):
    fig.update_layout(**{**LAYOUT, **kw})
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def note(text: str):
    """Recuadro de interpretación (como las 'Notas' del texto DAT262)."""
    st.info(text, icon="💡")


# ---------------------------------------------------------------------------
# Vista principal
# ---------------------------------------------------------------------------

def render_analysis(states, dom: Domain, n_steps: int, pi0_state: int, *,
                    valid=None, data_view: Callable[[], None] | None = None):
    k = len(dom.labels)
    res = me.analyze(states, k, valid)
    P, pi, mu = res["P"], res["pi"], res["mu"]
    erg, mk = res["ergodicity"], res["markov"]

    _header(res, dom)

    tabs = st.tabs(["① Datos y estados", "② Matriz P", "③ Propiedad de Markov",
                    "④ Predicción Pⁿ", "⑤ Distribución estacionaria π",
                    "⑥ Convergencia y ergodicidad", "📘 Teoría DAT262"])
    with tabs[0]:
        _tab_data(states, res, dom, data_view)
    with tabs[1]:
        _tab_matrix(res, dom)
    with tabs[2]:
        _tab_markov(res, dom)
    with tabs[3]:
        _tab_nstep(P, pi, dom, n_steps, pi0_state)
    with tabs[4]:
        _tab_stationary(res, dom)
    with tabs[5]:
        _tab_convergence(res, dom, n_steps, pi0_state)
    with tabs[6]:
        _tab_theory(dom)
    return res


def _header(res, dom: Domain):
    pi, mu, k = res["pi"], res["mu"], len(dom.labels)
    cols = st.columns(k + 1)
    cols[0].metric("Transiciones observadas", f"{res['n_trans']:,}",
                   f"{res['n_obs']:,} {dom.steps} de datos", delta_color="off", border=True,
                   help="Cada par de días consecutivos (X_t, X_{t+1}) es una transición.")
    for i in range(k):
        cols[i + 1].metric(
            f"π{i + 1} · {dom.short[i]}", pct(pi[i]),
            f"μ{i + 1} = {fmt_time(mu[i])} {dom.steps}", delta_color="off", border=True,
            help=f"Fracción del tiempo a largo plazo en {dom.labels[i]} (π{i + 1}) y "
                 f"tiempo medio de retorno μ{i + 1} = 1/π{i + 1}.")

    erg, mk = res["ergodicity"], res["markov"]
    a, b, c = st.columns(3)
    (a.success if erg["ergodic"] else a.warning)(
        "Cadena **ergódica**: π existe, es única y no depende del inicio." if erg["ergodic"]
        else "Cadena **no ergódica**: revisar clasificación de estados (pestaña ⑥).")
    (b.success if mk["markov_ok"] else b.warning)(
        f"Propiedad de Markov: p-valor = {mk['p_value']:.3f} "
        + ("(no se rechaza)" if mk["markov_ok"] else "(se rechaza al 5 %)"))
    mix = res["mixing"]
    c.info(f"La cadena «olvida» su estado inicial en ≈ **{mix} {dom.steps}**"
           if mix is not None else "La cadena no converge (periódica o reducible).")
    if res["unobserved"]:
        st.warning("Sin datos para " + ", ".join(dom.labels[i] for i in res["unobserved"])
                   + ": su fila de P se rellenó con 1/3 y no es interpretable. "
                   "Ajusta los umbrales o amplía el rango de fechas.")


# ---------------------------------------------------------------------------
# ① Datos y estados
# ---------------------------------------------------------------------------

def _tab_data(states, res, dom: Domain, data_view):
    st.markdown("#### Del fenómeno continuo al proceso estocástico de estados discretos")
    if data_view:
        data_view()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Elementos del proceso** (Def. 1.1 y 3.1)")
        st.latex(r"\{X_t : t \in T\},\quad T=\{1,2,\dots,%d\}\ (\text{%s})" % (len(states), dom.steps))
        st.latex(r"S=\{S_1,S_2,S_3\}=\{\text{%s}\}" % r"},\ \text{".join(dom.short))
        st.markdown(f"- **Espacio muestral Ω:** {dom.omega}\n"
                    f"- **Variable observada:** {dom.variable}\n"
                    "- **Trayectoria:** la secuencia observada x₁, x₂, …, xₙ es *una* realización "
                    "X(·, ω₀) del proceso.")
        st.markdown("**Regla de discretización**")
        st.latex(dom.rules_latex)
    with right:
        cnt = np.bincount(states, minlength=len(dom.labels))
        fig = go.Figure(go.Bar(x=dom.short, y=cnt, marker_color=dom.colors,
                               text=[f"{c} ({c / cnt.sum():.0%})" for c in cnt], textposition="outside",
                               hovertemplate="%{x}: %{y} " + dom.steps + "<extra></extra>"))
        fig.update_yaxes(title=dom.steps.capitalize(), range=[0, cnt.max() * 1.2])
        chart(fig, title="Frecuencia de cada estado", showlegend=False, height=330)

    first = " → ".join(f"S{s + 1}" for s in states[:20])
    st.markdown(f"**Primeros 20 valores de la trayectoria discretizada:** `{first} …`")
    if dom.reading:
        st.markdown("\n".join(f"- **{lab}:** {r}" for lab, r in zip(dom.labels, dom.reading)))


# ---------------------------------------------------------------------------
# ② Matriz de transición
# ---------------------------------------------------------------------------

def _tab_matrix(res, dom: Domain):
    N, P, lo, hi = res["N"], res["P"], res["P_lo"], res["P_hi"]
    k = len(dom.labels)
    st.markdown("#### Estimación de la matriz de transición empírica")
    st.markdown("Se cuenta cuántas veces el estado **i** (hoy) fue seguido del estado **j** "
                "(mañana) y se divide entre el total de salidas de **i** (estimador de máxima "
                "verosimilitud):")
    st.latex(r"\hat p_{ij}=\frac{N_{ij}}{N_{i\cdot}}=\frac{N_{ij}}{\sum_{m}N_{im}},"
             r"\qquad p_{ij}\ge 0,\qquad \sum_{j}p_{ij}=1")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Heatmap(
            z=N, x=dom.short, y=dom.short, colorscale="Blues", showscale=False,
            text=N, texttemplate="%{text}",
            hovertemplate="De %{y} a %{x}: N = %{z}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title="Hoy (i)")
        fig.update_xaxes(title="Mañana (j)", side="top")
        chart(fig, title="Conteos Nᵢⱼ", height=360)
    with c2:
        txt = [[f"{P[i, j]:.1%}" for j in range(k)] for i in range(k)]
        fig = go.Figure(go.Heatmap(
            z=P, x=dom.short, y=dom.short, colorscale="Blues", zmin=0, zmax=1,
            text=txt, texttemplate="%{text}", customdata=np.dstack([lo, hi]),
            colorbar=dict(title="p", tickformat=".0%"),
            hovertemplate="P(%{x} | %{y}) = %{z:.3f}<br>IC 95 %: [%{customdata[0]:.3f}, "
                          "%{customdata[1]:.3f}]<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title="Hoy (i)")
        fig.update_xaxes(title="Mañana (j)", side="top")
        chart(fig, title="Matriz P = (pᵢⱼ)", height=360)

    st.markdown("**Cálculo paso a paso de una fila**")
    i = st.radio("Estado de partida i", range(k), format_func=lambda x: dom.labels[x],
                 horizontal=True, key="row_i")
    ni = int(N[i].sum())
    if ni == 0:
        st.warning(f"No hay ninguna transición que salga de {dom.labels[i]}.")
    else:
        parts = [r"\hat p_{%d%d}=\frac{%d}{%d}=%.4f" % (i + 1, j + 1, N[i, j], ni, P[i, j]) for j in range(k)]
        st.latex(r",\quad ".join(parts))
        st.latex(r"\sum_j \hat p_{%d j} = " % (i + 1) + " + ".join(f"{P[i, j]:.4f}" for j in range(k))
                 + f" = {P[i].sum():.4f}")
        best = int(np.argmax(P[i]))
        note(f"De los {ni} {dom.steps} en {dom.labels[i]}, el {pct(P[i, best])} fue seguido por "
             f"{dom.labels[best]}. Un {pct(P[i, i])} de las veces el estado se repitió "
             f"(persistencia p{i + 1}{i + 1}).")

    st.markdown("**Matriz estimada con intervalos de confianza al 95 %**")
    st.latex(r"\hat P=" + mat_tex(P, "{:.4f}") +
             r"\qquad \hat p_{ij}\pm 1.96\sqrt{\frac{\hat p_{ij}(1-\hat p_{ij})}{N_{i\cdot}}}")
    df = pd.DataFrame({f"→ {dom.short[j]}": [f"{P[i, j]:.3f}  [{lo[i, j]:.3f}, {hi[i, j]:.3f}]"
                                              for i in range(k)] for j in range(k)},
                      index=[f"{dom.labels[i]} (N={N[i].sum()})" for i in range(k)])
    df["Σ fila"] = [f"{s:.6f}" for s in P.sum(axis=1)]
    st.dataframe(df, use_container_width=True)
    note("Cuanto menos datos tiene una fila (Nᵢ· pequeño), más ancho es su intervalo: "
         "esas probabilidades son menos confiables.")


# ---------------------------------------------------------------------------
# ③ Propiedad de Markov
# ---------------------------------------------------------------------------

def _tab_markov(res, dom: Domain):
    mk, P = res["markov"], res["P"]
    k = len(dom.labels)
    st.markdown("#### ¿El futuro depende solo del presente?")
    st.latex(r"P(X_{n+1}=j\mid X_n=i,\ X_{n-1}=h,\dots,X_0)=P(X_{n+1}=j\mid X_n=i)=p_{ij}")
    st.markdown("Contraste de **Anderson–Goodman**: para cada estado presente *i* se compara "
                "la distribución del día siguiente según cuál fue el día **anterior** *h*. "
                "Si la cadena no tiene memoria, conocer *h* no cambia nada.")
    st.latex(r"H_0:\ P(j\mid i,h)=p_{ij}\ \ \forall h \qquad\text{vs}\qquad H_1:\ \text{depende de }h")
    st.latex(r"\chi^2=\sum_{i}\sum_{h,j}\frac{(N_{hij}-E_{hij})^2}{E_{hij}},\qquad "
             r"E_{hij}=\frac{N_{hi\cdot}\,N_{\cdot ij}}{N_{\cdot i\cdot}},\qquad "
             r"gl=\sum_i (r_i-1)(c_i-1)")

    a, b, c, d = st.columns(4)
    a.metric("χ²", f"{mk['chi2']:.2f}", border=True)
    b.metric("Grados de libertad", mk["df"], border=True)
    c.metric("p-valor", f"{mk['p_value']:.4f}", border=True)
    d.metric("Máx |P(j|i,h) − pᵢⱼ|", f"{mk['max_abs_diff']:.3f}", border=True)
    if mk["markov_ok"]:
        st.success(f"p-valor = {mk['p_value']:.3f} > α = 0.05 → **no se rechaza H₀**: los datos son "
                   "compatibles con una cadena de Markov de orden 1 (ausencia de memoria).")
    else:
        st.warning(f"p-valor = {mk['p_value']:.3f} ≤ α = 0.05 → **se rechaza H₀**: el día anterior "
                   "aporta información extra. El modelo de orden 1 es una *aproximación* útil, "
                   "pero hay memoria más larga (rachas, estacionalidad). " + dom.markov_hint)

    N3, P2 = mk["N3"], mk["P2"]
    rows = []
    for i in range(k):
        for h in range(k):
            n = int(N3[h, i].sum())
            if n == 0:
                continue
            row = {"Presente i": dom.short[i], "Anterior h": dom.short[h], "n": n}
            for j in range(k):
                row[f"P({dom.short[j]}|i,h)"] = round(float(P2[h, i, j]), 3)
                row[f"p_i→{dom.short[j]}"] = round(float(P[i, j]), 3)
            rows.append(row)
    st.markdown("**Evidencia empírica:** probabilidad condicionada a dos días vs a uno")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    note("Si las columnas P(j|i,h) son parecidas para distintos h y cercanas a pᵢⱼ, "
         "el pasado lejano no aporta información: eso es la propiedad de Markov.")


# ---------------------------------------------------------------------------
# ④ Predicción en n pasos
# ---------------------------------------------------------------------------

def _tab_nstep(P, pi, dom: Domain, n: int, s0: int):
    k = len(dom.labels)
    pi0 = np.eye(k)[s0]
    st.markdown(f"#### ¿Qué pasará dentro de n = {n} {dom.steps}?")
    st.markdown("Ecuaciones de **Chapman–Kolmogorov**: la probabilidad de ir de *i* a *j* en "
                "n pasos es la entrada (i, j) de la potencia n-ésima de P.")
    st.latex(r"p^{(n)}_{ij}=[P^n]_{ij},\qquad p^{(m+n)}_{ij}=\sum_{r}p^{(m)}_{ir}\,p^{(n)}_{rj},"
             r"\qquad \pi_n=\pi_0\,P^n=\pi_{n-1}P")

    Pn = me.n_step_matrix(P, n)
    pin = pi0 @ Pn
    st.latex(r"\pi_{%d}=\underbrace{%s}_{\pi_0\ (%s)}\cdot %s^{%d}=%s"
             % (n, vec_tex(pi0, "{:.0f}"), dom.short[s0], mat_tex(P, "{:.3f}"), n, vec_tex(pin)))

    cols = st.columns(k)
    for j in range(k):
        cols[j].metric(f"P(X_{n} = {dom.short[j]} | X₀ = {dom.short[s0]})", pct(pin[j], 2),
                       f"{(pin[j] - pi[j]) * 100:+.2f} pp vs π{j + 1}", delta_color="off", border=True)
    note(f"Si hoy el estado es {dom.labels[s0]}, dentro de {n} {dom.steps} lo más probable es "
         f"{dom.labels[int(np.argmax(pin))]} ({pct(pin.max())}). La columna «vs π» muestra cuánto "
         "influye todavía el estado de hoy: cerca de 0 significa que ya se olvidó.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Recursión πₙ = πₙ₋₁ · P** (Ejemplo 3.1 b)")
        traj = me.convergence(P, pi0, n)
        df = pd.DataFrame(traj, columns=[f"P(X_n={s})" for s in dom.short])
        df.index.name = "n"
        st.dataframe(df.style.format("{:.4f}"), use_container_width=True, height=min(38 * (n + 2), 420))
    with c2:
        st.markdown(f"**Matriz Pⁿ con n = {n}**")
        txt = [[f"{Pn[i, j]:.1%}" for j in range(k)] for i in range(k)]
        fig = go.Figure(go.Heatmap(z=Pn, x=dom.short, y=dom.short, colorscale="Blues", zmin=0, zmax=1,
                                   text=txt, texttemplate="%{text}", showscale=False,
                                   hovertemplate="p⁽ⁿ⁾(%{y}→%{x}) = %{z:.4f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title="Hoy")
        fig.update_xaxes(title=f"Dentro de {n} {dom.steps}", side="top")
        chart(fig, height=330)

    st.markdown("**Cálculo manual de un elemento de P² (Ejemplo 3.1 a)**")
    a, b = st.columns(2)
    i = a.selectbox("Desde i", range(k), format_func=lambda x: dom.labels[x], key="ck_i")
    j = b.selectbox("Hasta j", range(k), index=k - 1, format_func=lambda x: dom.labels[x], key="ck_j")
    terms = me.chapman_kolmogorov_terms(P, i, j)
    sym = " + ".join(r"p_{%d%d}p_{%d%d}" % (i + 1, m + 1, m + 1, j + 1) for m, *_ in terms)
    num = " + ".join(r"(%.3f)(%.3f)" % (a_, b_) for _, a_, b_, _ in terms)
    tot = " + ".join(f"{t:.4f}" for *_, t in terms)
    st.latex(r"p^{(2)}_{%d%d}=%s" % (i + 1, j + 1, sym))
    st.latex(r"=%s=%s=\mathbf{%.4f}" % (num, tot, sum(t for *_, t in terms)))
    note(f"Para pasar de {dom.short[i]} a {dom.short[j]} en 2 {dom.steps} hay que sumar todos los "
         "caminos intermedios posibles: eso es multiplicar matrices.")


# ---------------------------------------------------------------------------
# ⑤ Distribución estacionaria
# ---------------------------------------------------------------------------

def _tab_stationary(res, dom: Domain):
    P, pi, mu = res["P"], res["pi"], res["mu"]
    k = len(dom.labels)
    st.markdown("#### Equilibrio a largo plazo: π P = π")
    st.markdown("Se busca un vector de probabilidades que **no cambie** al aplicar un paso de la "
                "cadena. Escrito ecuación por ecuación (Ejemplo 3.2):")
    eqs = []
    for j in range(k):
        rhs = " + ".join(r"%.4f\,\pi_%d" % (P[i, j], i + 1) for i in range(k))
        eqs.append(r"\pi_%d &= %s" % (j + 1, rhs))
    eqs.append(r"1 &= " + " + ".join(r"\pi_%d" % (i + 1) for i in range(k)))
    st.latex(r"\begin{aligned}" + r" \\ ".join(eqs) + r"\end{aligned}")

    st.markdown("Una de las k ecuaciones de balance es redundante; se reemplaza por la "
                "normalización y se resuelve el sistema lineal A πᵀ = b (Aplicación 7.2):")
    A = P.T - np.eye(k)
    A[-1] = 1.0
    st.latex(r"\underbrace{%s}_{A=(P^T-I)\ \text{con última fila}=1}"
             r"\begin{bmatrix}\pi_1\\ \pi_2\\ \pi_3\end{bmatrix}=\begin{bmatrix}0\\0\\1\end{bmatrix}"
             r"\ \Longrightarrow\ \pi=%s" % (mat_tex(A, "{:.4f}"), vec_tex(pi)))
    check = pi @ P
    st.latex(r"\text{Verificación: }\ \pi P=%s\approx\pi\quad\checkmark" % vec_tex(check))

    st.markdown("**Tres métodos, mismo resultado**")
    meth = pd.DataFrame(
        [res["pi"], res["pi_eig"], res["pi_power"]], columns=[f"π{i + 1} {dom.short[i]}" for i in range(k)],
        index=["Sistema lineal (solve)", "Autovector izquierdo, λ = 1",
               f"Método de potencia ({res['power_iters']} iteraciones)"])
    st.dataframe(meth.style.format("{:.6f}"), use_container_width=True)

    st.markdown("**Tiempos característicos: teoría vs lo observado en los datos**")
    st.latex(r"\mu_i=\frac{1}{\pi_i}\ \ (\text{tiempo medio de retorno}),\qquad "
             r"\bar d_i=\frac{1}{1-p_{ii}}\ \ (\text{duración media de una racha, MTTF})")
    tbl = pd.DataFrame({
        "πᵢ (teórico)": [pct(x, 2) for x in pi],
        "Frecuencia observada": [pct(x, 2) for x in res["emp_freq"]],
        f"μᵢ = 1/πᵢ ({dom.steps})": [fmt_time(x) for x in mu],
        "Retorno observado": [fmt_time(x) if np.isfinite(x) else "—" for x in res["emp_return"]],
        "Racha 1/(1−pᵢᵢ)": [fmt_time(x) for x in res["sojourn"]],
        "Racha observada": [fmt_time(x) if np.isfinite(x) else "—" for x in res["emp_sojourn"]],
    }, index=dom.labels)
    st.dataframe(tbl, use_container_width=True)

    fig = go.Figure()
    fig.add_bar(x=dom.short, y=pi, name="π teórica (modelo)", marker_color=dom.colors,
                hovertemplate="π = %{y:.2%}<extra></extra>")
    fig.add_bar(x=dom.short, y=res["emp_freq"], name="Frecuencia observada",
                marker=dict(color=dom.colors, pattern_shape="/", opacity=0.55),
                hovertemplate="observada = %{y:.2%}<extra></extra>")
    fig.update_yaxes(tickformat=".0%", title="Proporción del tiempo")
    chart(fig, barmode="group", bargap=0.25, bargroupgap=0.08, height=340,
          title="Teorema ergódico: fracción de tiempo observada ≈ πᵢ")

    top = int(np.argmax(pi))
    rare = int(np.argmin(pi))
    note(f"A largo plazo el {pct(pi[top])} de los {dom.steps} será {dom.labels[top]}. "
         f"{dom.labels[rare]} es el estado menos frecuente ({pct(pi[rare])}): una vez que ocurre, "
         f"pasan en promedio μ{rare + 1} = {fmt_time(mu[rare])} {dom.steps} hasta que vuelve a ocurrir. "
         f"Cuando empieza una racha de {dom.labels[top]}, dura en promedio "
         f"{fmt_time(res['sojourn'][top])} {dom.steps}.")


# ---------------------------------------------------------------------------
# ⑥ Convergencia y ergodicidad
# ---------------------------------------------------------------------------

def _tab_convergence(res, dom: Domain, n_steps: int, s0: int):
    P, pi = res["P"], res["pi"]
    erg = res["ergodicity"]
    k = len(dom.labels)
    n_max = max(30, n_steps)

    st.markdown("#### Convergencia al equilibrio (Propiedad del Cap. 3, Figura 3.1)")
    st.latex(r"\text{Finita, irreducible y aperiódica}\ \Longrightarrow\ "
             r"\lim_{n\to\infty}p^{(n)}_{ij}=\pi_j\quad\forall i,\qquad "
             r"\lim_{n\to\infty}P^n=\mathbf 1\,\pi=" + mat_tex(np.tile(pi, (k, 1)), "{:.3f}"))

    conv = me.convergence(P, np.eye(k)[s0], n_max)
    fig = go.Figure()
    for j in range(k):
        fig.add_scatter(x=list(range(n_max + 1)), y=conv[:, j], name=dom.short[j], mode="lines+markers",
                        line=dict(color=dom.colors[j], width=2), marker=dict(size=5),
                        hovertemplate="n=%{x}: %{y:.4f}<extra>" + dom.short[j] + "</extra>")
        fig.add_hline(y=pi[j], line=dict(color=dom.colors[j], dash="dot", width=1),
                      annotation_text=f"π{j + 1} = {pi[j]:.3f}", annotation_position="right")
    fig.add_vline(x=n_steps, line=dict(color="gray", dash="dash"),
                  annotation_text=f"n = {n_steps}", annotation_position="top")
    fig.update_xaxes(title=f"n ({dom.steps})")
    fig.update_yaxes(title="P(Xₙ = j)", range=[0, 1])
    chart(fig, title=f"πₙ partiendo de {dom.labels[s0]} → π (punteado)", height=420,
          hovermode="x unified")

    st.markdown("**Pérdida de memoria: las filas de Pⁿ se vuelven idénticas**")
    sub = make_subplots(rows=1, cols=k, shared_yaxes=False,
                        subplot_titles=[f"p⁽ⁿ⁾(i → {s})" for s in dom.short])
    Q, hist = np.eye(k), []
    for _ in range(n_max + 1):
        hist.append(Q)
        Q = Q @ P
    hist = np.array(hist)
    for j in range(k):
        for i in range(k):
            sub.add_scatter(x=list(range(n_max + 1)), y=hist[:, i, j], row=1, col=j + 1,
                            name=f"desde {dom.short[i]}", legendgroup=str(i), showlegend=j == 0,
                            line=dict(color=dom.colors[i], width=2),
                            hovertemplate="n=%{x}: %{y:.4f}<extra>desde " + dom.short[i] + "</extra>")
    sub.update_xaxes(title_text="n")
    chart(sub, height=360, legend=dict(orientation="h", yanchor="top", y=-0.28, x=0))
    note("Cada panel muestra la probabilidad de estar en un estado según desde dónde se partió. "
         "Las tres curvas se juntan en πⱼ: después de suficientes pasos, el estado inicial ya no importa.")

    c1, c2 = st.columns([3, 2])
    with c1:
        d = me.power_distance(P, pi, n_max)
        fig = go.Figure(go.Scatter(x=list(range(n_max + 1)), y=np.maximum(d, 1e-16), mode="lines+markers",
                                   line=dict(color=dom.colors[0], width=2), name="d(n)",
                                   hovertemplate="n=%{x}: %{y:.2e}<extra></extra>"))
        lam2 = res["lambda2"]
        if 0 < lam2 < 1:
            fig.add_scatter(x=list(range(n_max + 1)), y=np.maximum(lam2 ** np.arange(n_max + 1), 1e-16),
                            mode="lines", line=dict(color="gray", dash="dot"), name="|λ₂|ⁿ")
        fig.add_hline(y=0.01, line=dict(color="gray", dash="dash", width=1),
                      annotation_text="1 %", annotation_position="bottom right")
        fig.update_yaxes(type="log", title="máx distancia TV")
        fig.update_xaxes(title="n")
        chart(fig, title="d(n) = maxᵢ ‖Pⁿ(i,·) − π‖_TV", height=340)
    with c2:
        lam = res["eigenvalues"]
        st.markdown("**Velocidad de convergencia**")
        st.latex(r"\lambda=" + vec_tex(np.round(lam.real, 4), "{:.4f}"))
        st.latex(r"d(n)\approx C\,|\lambda_2|^n,\qquad |\lambda_2|=%.4f" % res["lambda2"])
        mix = res["mixing"]
        st.markdown(f"Distancia menor a 1 % a partir de **n = {mix}** {dom.steps}." if mix is not None
                    else "La distancia no baja de 1 %: la cadena no converge.")

    st.markdown("**Clasificación de estados** (Def. Cap. 3)")
    a, b, c, d_ = st.columns(4)
    yes = lambda v: "✅ Sí" if v else "❌ No"  # noqa: E731
    a.metric("Irreducible", yes(erg["irreducible"]), border=True,
             help="Todos los estados se comunican entre sí.")
    b.metric("Aperiódica", yes(erg["aperiodic"]), border=True,
             help="d(i) = mcd{n : pᵢᵢ⁽ⁿ⁾ > 0} = 1 para todo i.")
    c.metric("Regular (Pᵐ > 0)", yes(erg["regular"]), border=True,
             help="Existe m tal que todas las entradas de Pᵐ son positivas.")
    d_.metric("Ergódica", yes(erg["ergodic"]), border=True,
              help="Irreducible + aperiódica (+ finita): π única y límite de Pⁿ.")
    cls = pd.DataFrame([{"Estado": dom.labels[r["estado"]], "Clase comunicante": r["clase"],
                         "Tipo": "Recurrente" if r["recurrente"] else "Transitorio",
                         "Período d(i)": r["periodo"],
                         "Aperiódico": "Sí" if r["aperiodico"] else "No"} for r in erg["states"]])
    st.dataframe(cls, use_container_width=True, hide_index=True)

    st.markdown("**Simulación Monte Carlo** (Ejercicio 3.2): se generan trayectorias artificiales "
                "con la matriz P estimada y se comprueba el teorema ergódico.")
    n_sim = st.select_slider("Pasos simulados", [1_000, 5_000, 10_000, 50_000, 100_000], value=10_000)
    x = me.simulate(P, s0, n_sim, seed=42)
    t = np.arange(1, n_sim + 1)
    fig = go.Figure()
    for j in range(k):
        run = np.cumsum(x == j) / t
        idx = np.unique(np.geomspace(1, n_sim, 400).astype(int)) - 1
        fig.add_scatter(x=t[idx], y=run[idx], name=dom.short[j], line=dict(color=dom.colors[j], width=2),
                        hovertemplate="t=%{x}: %{y:.4f}<extra>" + dom.short[j] + "</extra>")
        fig.add_hline(y=pi[j], line=dict(color=dom.colors[j], dash="dot", width=1))
    fig.update_xaxes(type="log", title="t (pasos simulados, escala log)")
    fig.update_yaxes(title="Fracción acumulada en el estado", range=[0, 1])
    chart(fig, title="(1/t)·Σ 1{Xₛ = j} → πⱼ", height=380)
    freq = me.empirical_distribution(x, k)
    st.dataframe(pd.DataFrame({"π teórica": pi, "Frecuencia simulada": freq,
                               "Error absoluto": np.abs(freq - pi)}, index=dom.labels)
                 .style.format("{:.4f}"), use_container_width=True)


# ---------------------------------------------------------------------------
# 📘 Teoría
# ---------------------------------------------------------------------------

def _tab_theory(dom: Domain):
    st.markdown("#### Resumen teórico — DAT262, Capítulo 3: Procesos Markovianos")
    sections = [
        ("1. Proceso estocástico (Def. 1.1)",
         r"X: T\times\Omega\to S,\qquad \{X_t : t\in T\}",
         f"Familia de variables aleatorias indexadas por el tiempo. Aquí T son {dom.steps} "
         "consecutivos y S tiene 3 estados: es una cadena de tiempo y estado discretos."),
        ("2. Propiedad de Markov",
         r"P(X_{n+1}=j\mid X_n=i,X_{n-1},\dots,X_0)=P(X_{n+1}=j\mid X_n=i)=p_{ij}",
         "«El futuro depende del pasado solo a través del presente». Se contrasta en la pestaña ③."),
        ("3. Matriz de transición",
         r"P=(p_{ij}),\quad p_{ij}\ge 0,\quad \sum_j p_{ij}=1,\quad \hat p_{ij}=\frac{N_{ij}}{N_{i\cdot}}",
         "Cada fila es una distribución de probabilidad: desde i, a dónde se va."),
        ("4. Probabilidades en n pasos (Chapman–Kolmogorov)",
         r"p^{(n)}_{ij}=[P^n]_{ij},\qquad \pi_n=\pi_0P^n",
         "Predicción del estado dentro de n pasos dada la distribución actual."),
        ("5. Clasificación de estados",
         r"i\leftrightarrow j \iff p^{(n)}_{ij}>0,\ p^{(m)}_{ji}>0;\qquad d(i)=\mathrm{mcd}\{n: p^{(n)}_{ii}>0\}",
         "Irreducible: todos se comunican. Recurrente: el regreso es seguro. Aperiódico: d(i) = 1."),
        ("6. Distribución estacionaria",
         r"\pi P=\pi,\qquad \sum_i\pi_i=1",
         "Si la cadena es finita, irreducible y aperiódica, π existe, es única y es el límite de πₙ "
         "para cualquier inicio."),
        ("7. Teorema ergódico y tiempos medios",
         r"\frac1T\sum_{t=1}^T\mathbf 1\{X_t=j\}\to\pi_j,\qquad \mu_j=\frac{1}{\pi_j},\qquad "
         r"\bar d_j=\frac{1}{1-p_{jj}}",
         "π es también la fracción del tiempo en cada estado; μⱼ es el tiempo medio de retorno y "
         "1/(1−pⱼⱼ) la duración media de una racha (MTTF en la Aplicación 7.2)."),
    ]
    for title, formula, text in sections:
        with st.expander(title, expanded=True):
            st.latex(formula)
            st.markdown(text)
    st.caption("Supuesto clave del modelo: **homogeneidad temporal** (P no cambia con t). "
               "Si el fenómeno tiene estacionalidad, conviene estimar P por temporadas.")
