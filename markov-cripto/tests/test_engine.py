"""Pruebas del motor CMTD contra los ejemplos resueltos del texto DAT262."""
import numpy as np

from app.core import markov_engine as me

# Ejemplo 3.1 / 3.2: servidor web (Operativo, Degradado, Caído)
P_SERVIDOR = np.array([[0.90, 0.07, 0.03],
                       [0.40, 0.50, 0.10],
                       [0.10, 0.30, 0.60]])
# Aplicación 7.2: enlace ISP (Normal, Congestionado, Caído)
P_ISP = np.array([[0.95, 0.04, 0.01],
                  [0.30, 0.60, 0.10],
                  [0.10, 0.40, 0.50]])


def test_ejemplo_3_1_dos_y_tres_pasos():
    P2 = me.n_step_matrix(P_SERVIDOR, 2)
    assert np.allclose(P2[0], [0.841, 0.107, 0.052], atol=5e-4)
    terms = me.chapman_kolmogorov_terms(P_SERVIDOR, 0, 2)
    assert np.isclose(sum(t[3] for t in terms), P2[0, 2])
    assert np.allclose(me.n_step(P_SERVIDOR, [1, 0, 0], 3), [0.805, 0.128, 0.067], atol=1e-3)


def test_ejemplo_3_2_estacionaria_tres_metodos():
    # El texto reporta (0.7427, 0.1616, 0.0961); el valor exacto es 170/229, 37/229, 22/229
    exacto = np.array([170, 37, 22]) / 229
    assert np.allclose(me.stationary(P_SERVIDOR), [0.7427, 0.1616, 0.0961], atol=1e-3)
    assert np.allclose(me.stationary(P_SERVIDOR), exacto, atol=1e-12)
    assert np.allclose(me.stationary_eig(P_SERVIDOR), exacto, atol=1e-10)
    pi, it = me.stationary_power(P_SERVIDOR)
    assert np.allclose(pi, exacto, atol=1e-10) and it < 10_000


def test_aplicacion_7_2_disponibilidad_y_mttf():
    # El texto reporta pi = (0.813, 0.136, 0.051), pero ese vector no cumple pi P = pi.
    # La solución exacta es (5/6, 1/8, 1/24): disponibilidad 83.3 %.
    pi = me.stationary(P_ISP)
    assert np.allclose(pi, [5 / 6, 1 / 8, 1 / 24], atol=1e-12)
    assert not np.allclose(np.array([0.813, 0.136, 0.051]) @ P_ISP, [0.813, 0.136, 0.051], atol=1e-3)
    assert np.isclose(me.mean_sojourn_times(P_ISP)[0], 20.0)
    assert np.allclose(me.mean_return_times(pi), 1 / pi)


def test_estimacion_desde_trayectoria():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 3, 2000)
    r = me.analyze(s)
    assert np.allclose(r["P"].sum(axis=1), 1)
    assert np.allclose(r["pi"] @ r["P"], r["pi"], atol=1e-10)
    assert np.allclose(r["pi"], r["pi_eig"], atol=1e-8)
    assert (r["P_lo"] <= r["P"]).all() and (r["P"] <= r["P_hi"]).all()
    assert r["ergodicity"]["ergodic"]
    assert me.transition_counts([0, 1, 1, 2, 0], 3).sum() == 4


def test_huecos_no_cuentan_transiciones():
    s = [0, 0, 2, 2]
    valid = [True, False, True]  # entre el 2º y 3er dato hay un hueco de fechas
    N = me.transition_counts(s, 3, valid)
    assert N[0, 0] == 1 and N[2, 2] == 1 and N[0, 2] == 0


def test_teorema_ergodico_por_simulacion():
    x = me.simulate(P_SERVIDOR, 2, 50_000, seed=7)
    pi = me.stationary(P_SERVIDOR)
    assert np.allclose(me.empirical_distribution(x), pi, atol=0.01)
    assert np.allclose(me.empirical_return_times(x), 1 / pi, rtol=0.05)
    assert np.allclose(me.empirical_sojourn_times(x), me.mean_sojourn_times(P_SERVIDOR), rtol=0.05)
    assert me.markov_property_test(x, P_SERVIDOR)["markov_ok"]


def test_convergencia_y_autovalores():
    pi = me.stationary(P_SERVIDOR)
    assert me.power_distance(P_SERVIDOR, pi, 60)[-1] < 1e-6
    assert np.isclose(me.eigenvalues(P_SERVIDOR)[0], 1)
    n = me.mixing_time(P_SERVIDOR, pi, 0.01)
    assert n is not None and 5 <= n <= 20  # "en unas 10 horas" (Figura 3.1)


def test_clasificacion_reducible_y_periodica():
    red = np.array([[1, 0, 0], [.5, .5, 0], [0, 0, 1.]])
    assert not me.is_irreducible(red)
    cls = {r["estado"]: r for r in me.classify_states(red)}
    assert cls[0]["recurrente"] and not cls[1]["recurrente"] and cls[2]["recurrente"]

    ciclo = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.]])
    assert me.is_irreducible(ciclo) and me.period(ciclo, 0) == 3
    assert not me.ergodicity_report(ciclo)["ergodic"]


def test_dependencia_de_orden_2_se_detecta():
    # X_{t+1} depende de X_{t-1}: patrón 0,0,1,1,0,0,1,1,... con ruido
    rng = np.random.default_rng(3)
    s = np.tile([0, 0, 1, 1], 500)
    s[rng.random(len(s)) < 0.05] = 2
    P = me.transition_matrix(me.transition_counts(s))
    assert not me.markov_property_test(s, P)["markov_ok"]
