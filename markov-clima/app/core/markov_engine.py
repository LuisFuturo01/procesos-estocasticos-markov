"""Motor estocástico CMTD (DAT262, Cap. 3). Estados indexados 0..k-1 (S1..Sk)."""
from __future__ import annotations

import numpy as np
from scipy import stats

K = 3


def transition_counts(states, k: int = K) -> np.ndarray:
    """N_ij: frecuencias absolutas de transición entre estados consecutivos."""
    s = np.asarray(states, dtype=int)
    N = np.zeros((k, k), dtype=int)
    np.add.at(N, (s[:-1], s[1:]), 1)
    return N


def transition_matrix(N: np.ndarray) -> np.ndarray:
    """P estocástica: p_ij = N_ij / sum_j N_ij (fila sin datos -> uniforme)."""
    N = np.asarray(N, dtype=float)
    k = N.shape[0]
    rows = N.sum(axis=1, keepdims=True)
    P = np.where(rows > 0, N / np.where(rows == 0, 1, rows), 1.0 / k)
    return P


def n_step(P: np.ndarray, pi0, n: int) -> np.ndarray:
    """pi_n = pi_0 · P^n."""
    return np.asarray(pi0, dtype=float) @ np.linalg.matrix_power(P, n)


def stationary(P: np.ndarray) -> np.ndarray:
    """Resuelve pi P = pi con sum(pi)=1 (mínimos cuadrados del sistema lineal)."""
    k = P.shape[0]
    A = np.vstack([P.T - np.eye(k), np.ones(k)])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


def stationary_eig(P: np.ndarray) -> np.ndarray:
    """Alternativa por autovector izquierdo asociado a lambda = 1."""
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = np.abs(pi)
    return pi / pi.sum()


def mean_return_times(pi: np.ndarray) -> np.ndarray:
    """mu_i = 1 / pi_i (inf si pi_i = 0)."""
    with np.errstate(divide="ignore"):
        return np.where(pi > 0, 1.0 / np.where(pi == 0, 1, pi), np.inf)


def is_irreducible(P: np.ndarray) -> bool:
    k = P.shape[0]
    reach = np.linalg.matrix_power(np.eye(k) + (P > 0), k) > 0
    return bool(reach.all())


def is_regular(P: np.ndarray) -> bool:
    """Existe m con P^m > 0 (cota de Wielandt: m <= k^2 - 2k + 2)."""
    k = P.shape[0]
    Q = np.eye(k)
    for _ in range(k * k - 2 * k + 2):
        Q = Q @ P
        if (Q > 0).all():
            return True
    return False


def ergodicity_report(P: np.ndarray) -> dict:
    irr, reg = is_irreducible(P), is_regular(P)
    return {"irreducible": irr, "regular": reg, "ergodic": irr and reg}


def convergence(P: np.ndarray, pi0, n_max: int = 30) -> np.ndarray:
    """Matriz (n_max+1, k): pi_n para n = 0..n_max."""
    out, cur = [], np.asarray(pi0, dtype=float)
    for _ in range(n_max + 1):
        out.append(cur)
        cur = cur @ P
    return np.array(out)


def power_distance(P: np.ndarray, pi: np.ndarray, n_max: int = 30) -> np.ndarray:
    """Distancia TV máx. entre cada fila de P^n y pi (pérdida de memoria inicial)."""
    d, Q = [], np.eye(P.shape[0])
    for _ in range(n_max + 1):
        d.append(0.5 * np.abs(Q - pi).sum(axis=1).max())
        Q = Q @ P
    return np.array(d)


def markov_property_test(states, P: np.ndarray, k: int = K) -> dict:
    """Contrasta P(X_{n+1}|X_n,X_{n-1}) vs P(X_{n+1}|X_n) con chi-cuadrado de Pearson."""
    s = np.asarray(states, dtype=int)
    N3 = np.zeros((k, k, k))
    np.add.at(N3, (s[:-2], s[1:-1], s[2:]), 1)
    N2 = N3.sum(axis=2, keepdims=True)
    expected = N2 * P[None, :, :]
    mask = expected > 0
    chi2 = float((((N3 - expected) ** 2)[mask] / expected[mask]).sum())
    df = k * (k - 1) ** 2
    P2 = np.where(N2 > 0, N3 / np.where(N2 == 0, 1, N2), np.nan)
    maxdiff = float(np.nanmax(np.abs(P2 - P[None, :, :])))
    pval = float(stats.chi2.sf(chi2, df))
    return {"chi2": chi2, "df": df, "p_value": pval, "max_abs_diff": maxdiff,
            "markov_ok": pval > 0.05}


def analyze(states, k: int = K) -> dict:
    N = transition_counts(states, k)
    P = transition_matrix(N)
    pi = stationary(P)
    return {"N": N, "P": P, "pi": pi, "mu": mean_return_times(pi),
            "ergodicity": ergodicity_report(P),
            "markov": markov_property_test(states, P, k)}
