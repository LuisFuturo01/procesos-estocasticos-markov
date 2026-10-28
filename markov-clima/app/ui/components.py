"""Componentes de visualización compartidos (Plotly + Streamlit)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.core import markov_engine as me

COLORS = ["#4F8BF9", "#F9A84F", "#F95F6B"]
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=40, b=10))


def render_analysis(states, labels, n_steps: int, pi0_state: int):
    k = len(labels)
    res = me.analyze(states, k)
    P, pi, mu, N = res["P"], res["pi"], res["mu"], res["N"]
    pi0 = np.eye(k)[pi0_state]

    st.subheader("Indicadores a largo plazo")
    cols = st.columns(k)
    for i, c in enumerate(cols):
        c.metric(f"π{i+1} · {labels[i]}", f"{pi[i]*100:.1f}%", f"μ{i+1} = {mu[i]:.2f} días", delta_color="off")

    t1, t2, t3, t4 = st.tabs(["Matriz P", "Convergencia", "Verificación", "Teoría"])

    with t1:
        txt = [[f"{P[i, j]*100:.1f}%<br>(N={N[i, j]})" for j in range(k)] for i in range(k)]
        fig = go.Figure(go.Heatmap(z=P, x=labels, y=labels, text=txt, texttemplate="%{text}",
                                   colorscale="Blues", zmin=0, zmax=1))
        fig.update_layout(title="Matriz de transición empírica P (fila = estado actual)",
                          yaxis=dict(autorange="reversed"), **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Suma de filas: {np.round(P.sum(axis=1), 6).tolist()} · observaciones: {len(states)}")

    with t2:
        conv = me.convergence(P, pi0, 30)
        fig = go.Figure()
        for i in range(k):
            fig.add_trace(go.Scatter(x=list(range(31)), y=conv[:, i], name=labels[i],
                                     line=dict(color=COLORS[i % 3], width=3)))
            fig.add_hline(y=pi[i], line=dict(color=COLORS[i % 3], dash="dot", width=1))
        fig.add_vline(x=n_steps, line=dict(color="white", dash="dash"))
        fig.update_layout(title=f"π_n → π (inicio en {labels[pi0_state]}; punteado = π)",
                          xaxis_title="n (pasos)", yaxis_title="Probabilidad", **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        pn = me.n_step(P, pi0, n_steps)
        st.latex(r"\pi_n=\pi_0 P^n \quad (n=%d)" % n_steps)
        st.write({labels[i]: f"{pn[i]*100:.2f}%" for i in range(k)})

        d = me.power_distance(P, pi, 30)
        fig2 = go.Figure(go.Scatter(x=list(range(31)), y=d, line=dict(color=COLORS[0], width=3)))
        fig2.update_layout(title="Pérdida de memoria: máx. distancia entre filas de Pⁿ y π",
                           xaxis_title="n", yaxis_type="log", **LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        e, m = res["ergodicity"], res["markov"]
        a, b, c = st.columns(3)
        a.metric("Irreducible", "Sí" if e["irreducible"] else "No")
        b.metric("Regular (Pᵐ > 0)", "Sí" if e["regular"] else "No")
        c.metric("Ergódica", "Sí" if e["ergodic"] else "No")
        st.markdown("**Propiedad de Markov** (orden 2 vs orden 1, χ² de Pearson)")
        st.write(f"χ² = {m['chi2']:.2f} · gl = {m['df']} · p-valor = {m['p_value']:.3f} · "
                 f"máx |P(·|i,i₋₁) − P(·|i)| = {m['max_abs_diff']:.3f}")
        (st.success if m["markov_ok"] else st.warning)(
            "No se rechaza la ausencia de memoria (p > 0.05)." if m["markov_ok"]
            else "Se rechaza la propiedad de Markov de orden 1 al 5%: el modelo es una aproximación.")
        st.write("π (sistema lineal) vs π (autovector):",
                 np.round(pi, 4).tolist(), np.round(me.stationary_eig(P), 4).tolist())

    with t4:
        st.latex(r"P(X_{n+1}=j\mid X_n=i,\dots,X_0)=P(X_{n+1}=j\mid X_n=i)=p_{ij}")
        st.latex(r"p_{ij}=\frac{N_{ij}}{\sum_j N_{ij}},\qquad \sum_j p_{ij}=1")
        st.latex(r"\pi_n=\pi_0P^n,\qquad \pi P=\pi,\ \ \sum_i\pi_i=1")
        st.latex(r"\mu_i=\frac{1}{\pi_i},\qquad \lim_{n\to\infty}P^n=\mathbf{1}\pi")
