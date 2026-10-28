import numpy as np

from app.core import markov_engine as me


def test_rows_sum_one_and_stationary():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 3, 2000)
    r = me.analyze(s)
    assert np.allclose(r["P"].sum(axis=1), 1)
    assert np.allclose(r["pi"] @ r["P"], r["pi"], atol=1e-8)
    assert np.allclose(r["pi"], me.stationary_eig(r["P"]), atol=1e-6)
    assert np.allclose(r["mu"], 1 / r["pi"])
    assert r["ergodicity"]["ergodic"]


def test_convergence_and_counts():
    P = np.array([[.5, .3, .2], [.2, .6, .2], [.3, .3, .4]])
    pi = me.stationary(P)
    assert np.allclose(me.n_step(P, [1, 0, 0], 200), pi, atol=1e-9)
    assert me.power_distance(P, pi, 30)[-1] < 1e-6
    assert me.transition_counts([0, 1, 1, 2, 0], 3).sum() == 4


def test_reducible():
    assert not me.is_irreducible(np.array([[1, 0, 0], [.5, .5, 0], [0, 0, 1.]]))
