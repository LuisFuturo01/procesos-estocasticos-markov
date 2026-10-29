"""Motor estocástico CMTD (DAT262, Cap. 3 y Aplicación 7.2).

Convenciones:
- Estados indexados 0..k-1 (S1..Sk).
- Las distribuciones son vectores FILA: pi_n = pi_0 · P^n.
- P es estocástica por filas: p_ij >= 0 y sum_j p_ij = 1.
- `valid` (longitud n-1) marca qué pares (X_t, X_{t+1}) son días realmente
  consecutivos; sirve para excluir huecos de fechas o cortes de temporada.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy import stats

K = 3


# ---------------------------------------------------------------------------
# Utilidades de la trayectoria observada
# ---------------------------------------------------------------------------

def _segments(states, valid=None) -> list[np.ndarray]:
    """Parte la trayectoria en tramos contiguos (sin huecos)."""
    s = np.asarray(states, dtype=int)
    if valid is None or len(s) < 2:
        return [s]
    cuts = np.flatnonzero(~np.asarray(valid, dtype=bool)) + 1
    return [seg for seg in np.split(s, cuts) if len(seg)]


# ---------------------------------------------------------------------------
# 3.1  Estimación de la matriz de transición
# ---------------------------------------------------------------------------

def transition_counts(states, k: int = K, valid=None) -> np.ndarray:
    """N_ij: nº de veces que X_t = i va seguido de X_{t+1} = j."""
    N = np.zeros((k, k), dtype=int)
    for seg in _segments(states, valid):
        if len(seg) > 1:
            np.add.at(N, (seg[:-1], seg[1:]), 1)
    return N


def transition_matrix(N: np.ndarray) -> np.ndarray:
    """Estimador de máxima verosimilitud p_ij = N_ij / N_i· .

    Una fila sin observaciones (estado nunca visitado) no puede estimarse;
    se rellena con 1/k para que P siga siendo estocástica y se reporta
    con `unobserved_states`.
    """
    N = np.asarray(N, dtype=float)
    k = N.shape[0]
    rows = N.sum(axis=1, keepdims=True)
    return np.where(rows > 0, N / np.where(rows == 0, 1, rows), 1.0 / k)


def unobserved_states(N: np.ndarray) -> list[int]:
    """Estados cuya fila de P no tiene datos (N_i· = 0)."""
    return [int(i) for i in np.flatnonzero(np.asarray(N).sum(axis=1) == 0)]


def transition_ci(N: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo de confianza aproximado (Wald) para cada p_ij:
    p_ij ± z·sqrt(p_ij(1-p_ij)/N_i·)."""
    N = np.asarray(N, dtype=float)
    P = transition_matrix(N)
    n = N.sum(axis=1, keepdims=True)
    se = np.sqrt(P * (1 - P) / np.where(n == 0, 1, n))
    se = np.where(n > 0, se, np.nan)
    return np.clip(P - z * se, 0, 1), np.clip(P + z * se, 0, 1)


# ---------------------------------------------------------------------------
# Probabilidades en n pasos (Chapman-Kolmogorov)
# ---------------------------------------------------------------------------

def n_step_matrix(P: np.ndarray, n: int) -> np.ndarray:
    """P^(n): p_ij^(n) = [P^n]_ij."""
    return np.linalg.matrix_power(np.asarray(P, dtype=float), n)


def n_step(P: np.ndarray, pi0, n: int) -> np.ndarray:
    """pi_n = pi_0 · P^n."""
    return np.asarray(pi0, dtype=float) @ n_step_matrix(P, n)


def chapman_kolmogorov_terms(P: np.ndarray, i: int, j: int) -> list[tuple[int, float, float, float]]:
    """Términos de p_ij^(2) = sum_m p_im · p_mj  (Ejemplo 3.1, paso a paso).

    Devuelve [(m, p_im, p_mj, p_im·p_mj), ...].
    """
    return [(m, float(P[i, m]), float(P[m, j]), float(P[i, m] * P[m, j]))
            for m in range(P.shape[0])]


def convergence(P: np.ndarray, pi0, n_max: int = 30) -> np.ndarray:
    """Matriz (n_max+1, k) con pi_n para n = 0..n_max (recursión pi_{n+1} = pi_n P)."""
    out, cur = [], np.asarray(pi0, dtype=float)
    for _ in range(n_max + 1):
        out.append(cur)
        cur = cur @ P
    return np.array(out)


# ---------------------------------------------------------------------------
# Distribución estacionaria  pi P = pi,  sum pi_i = 1
# ---------------------------------------------------------------------------

def stationary(P: np.ndarray) -> np.ndarray:
    """Sistema lineal de balance (método de la Aplicación 7.2).

    (P^T - I) pi^T = 0 tiene una ecuación redundante; se reemplaza la última
    por la normalización pi_1 + ... + pi_k = 1 y se resuelve exactamente.
    Si el sistema es singular (cadena reducible, pi no única) se recurre a
    mínimos cuadrados.
    """
    k = P.shape[0]
    A = P.T - np.eye(k)
    A[-1] = 1.0
    b = np.zeros(k)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        A = np.vstack([P.T - np.eye(k), np.ones(k)])
        pi = np.linalg.lstsq(A, np.r_[np.zeros(k), 1.0], rcond=None)[0]
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


def stationary_eig(P: np.ndarray) -> np.ndarray:
    """Autovector izquierdo de P asociado al autovalor lambda = 1."""
    w, v = np.linalg.eig(P.T)
    pi = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
    return pi / pi.sum()


def stationary_power(P: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000) -> tuple[np.ndarray, int]:
    """Método de potencia (Ejemplo 3.3, PageRank): pi <- pi P desde la uniforme.

    Devuelve (pi, iteraciones usadas)."""
    k = P.shape[0]
    pi = np.full(k, 1.0 / k)
    for it in range(1, max_iter + 1):
        new = pi @ P
        if np.max(np.abs(new - pi)) < tol:
            return new, it
        pi = new
    return pi, max_iter


def mean_return_times(pi: np.ndarray) -> np.ndarray:
    """Tiempo medio de recurrencia mu_i = 1 / pi_i (inf si pi_i = 0)."""
    pi = np.asarray(pi, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(pi > 0, 1.0 / np.where(pi == 0, 1, pi), np.inf)


def mean_sojourn_times(P: np.ndarray) -> np.ndarray:
    """Duración media de una racha en i: 1 / (1 - p_ii)  (MTTF, Aplicación 7.2).

    El tiempo de permanencia en i es geométrico con probabilidad de salida 1 - p_ii.
    """
    d = np.diag(np.asarray(P, dtype=float))
    with np.errstate(divide="ignore"):
        return np.where(d < 1, 1.0 / np.where(d == 1, 0.5, 1 - d), np.inf)


# ---------------------------------------------------------------------------
# Clasificación de estados
# ---------------------------------------------------------------------------

def reachability(P: np.ndarray) -> np.ndarray:
    """R[i, j] = True si j es accesible desde i (p_ij^(n) > 0 para algún n >= 0)."""
    k = P.shape[0]
    R = np.eye(k, dtype=bool) | (np.asarray(P) > 0)
    for _ in range(k):
        R = R | ((R.astype(int) @ R.astype(int)) > 0)
    return R


def communicating_classes(P: np.ndarray) -> list[list[int]]:
    """Clases de equivalencia de la relación 'i y j se comunican'."""
    R = reachability(P)
    C = R & R.T
    classes, seen = [], set()
    for i in range(P.shape[0]):
        if i not in seen:
            cls = [int(j) for j in np.flatnonzero(C[i])]
            seen.update(cls)
            classes.append(cls)
    return classes


def period(P: np.ndarray, i: int) -> int:
    """d(i) = mcd{n >= 1 : p_ii^(n) > 0}  (0 si nunca se regresa a i)."""
    k = P.shape[0]
    A = (np.asarray(P) > 0).astype(int)
    Q = np.eye(k, dtype=int)
    d = 0
    for n in range(1, 2 * k * k + 1):
        Q = ((Q @ A) > 0).astype(int)
        if Q[i, i]:
            d = gcd(d, n)
            if d == 1:
                break
    return d


def classify_states(P: np.ndarray) -> list[dict]:
    """En una cadena finita una clase es recurrente si y solo si es cerrada."""
    k = P.shape[0]
    out = []
    for c, cls in enumerate(communicating_classes(P)):
        rest = [j for j in range(k) if j not in cls]
        closed = not rest or float(np.asarray(P)[np.ix_(cls, rest)].sum()) == 0.0
        for i in cls:
            d = period(P, i)
            out.append({"estado": i, "clase": c + 1, "recurrente": closed,
                        "periodo": d, "aperiodico": d == 1})
    return sorted(out, key=lambda r: r["estado"])


def is_irreducible(P: np.ndarray) -> bool:
    return bool(reachability(P).all())


def is_aperiodic(P: np.ndarray) -> bool:
    return all(period(P, i) == 1 for i in range(P.shape[0]))


def is_regular(P: np.ndarray) -> bool:
    """Existe m con P^m > 0 en todas sus entradas (cota de Wielandt: m <= k^2 - 2k + 2)."""
    k = P.shape[0]
    Q = np.eye(k)
    for _ in range(k * k - 2 * k + 2):
        Q = Q @ P
        if (Q > 0).all():
            return True
    return False


def ergodicity_report(P: np.ndarray) -> dict:
    """Teorema de convergencia (Cap. 3): finita + irreducible + aperiódica => ergódica."""
    irr, ape = is_irreducible(P), is_aperiodic(P)
    return {"irreducible": irr, "aperiodic": ape, "regular": is_regular(P),
            "ergodic": irr and ape, "classes": communicating_classes(P),
            "states": classify_states(P)}


# ---------------------------------------------------------------------------
# Velocidad de convergencia (pérdida de memoria)
# ---------------------------------------------------------------------------

def eigenvalues(P: np.ndarray) -> np.ndarray:
    """Autovalores de P ordenados por módulo decreciente (lambda_1 = 1)."""
    w = np.linalg.eigvals(P)
    return w[np.argsort(-np.abs(w))]


def power_distance(P: np.ndarray, pi: np.ndarray, n_max: int = 30) -> np.ndarray:
    """d(n) = max_i ||P^n(i,·) - pi||_TV : qué tanto recuerda la cadena su estado inicial."""
    d, Q = [], np.eye(P.shape[0])
    for _ in range(n_max + 1):
        d.append(0.5 * np.abs(Q - pi).sum(axis=1).max())
        Q = Q @ P
    return np.array(d)


def mixing_time(P: np.ndarray, pi: np.ndarray, eps: float = 0.01, n_max: int = 1000) -> int | None:
    """Menor n con d(n) <= eps (None si no ocurre antes de n_max)."""
    Q = np.eye(P.shape[0])
    for n in range(n_max + 1):
        if 0.5 * np.abs(Q - pi).sum(axis=1).max() <= eps:
            return n
        Q = Q @ P
    return None


# ---------------------------------------------------------------------------
# Teorema ergódico: lo observado vs lo teórico
# ---------------------------------------------------------------------------

def empirical_distribution(states, k: int = K) -> np.ndarray:
    """Fracción del tiempo que la trayectoria pasa en cada estado."""
    s = np.asarray(states, dtype=int)
    return np.bincount(s, minlength=k) / max(len(s), 1)


def empirical_return_times(states, k: int = K, valid=None) -> np.ndarray:
    """Promedio observado de pasos entre dos visitas consecutivas a cada estado."""
    gaps = [[] for _ in range(k)]
    for seg in _segments(states, valid):
        for i in range(k):
            pos = np.flatnonzero(seg == i)
            gaps[i].extend(np.diff(pos).tolist())
    return np.array([np.mean(g) if g else np.nan for g in gaps])


def empirical_sojourn_times(states, k: int = K, valid=None) -> np.ndarray:
    """Duración media observada de las rachas (pasos seguidos en el mismo estado)."""
    runs = [[] for _ in range(k)]
    for seg in _segments(states, valid):
        if not len(seg):
            continue
        change = np.flatnonzero(np.diff(seg) != 0) + 1
        for block in np.split(seg, change):
            runs[int(block[0])].append(len(block))
    return np.array([np.mean(r) if r else np.nan for r in runs])


def simulate(P: np.ndarray, x0: int, n: int, seed: int | None = 42) -> np.ndarray:
    """Simulación Monte Carlo de n pasos (Ejercicio 3.2 / Aplicación 7.2)."""
    rng = np.random.default_rng(seed)
    cum = np.cumsum(P, axis=1)
    u = rng.random(n)
    x = np.empty(n, dtype=int)
    x[0] = x0
    for t in range(1, n):
        x[t] = min(int(np.searchsorted(cum[x[t - 1]], u[t], side="right")), P.shape[0] - 1)
    return x


# ---------------------------------------------------------------------------
# Propiedad de Markov (ausencia de memoria)
# ---------------------------------------------------------------------------

def second_order_counts(states, k: int = K, valid=None) -> np.ndarray:
    """N_hij: nº de veces que (X_{t-1}, X_t, X_{t+1}) = (h, i, j)."""
    N3 = np.zeros((k, k, k), dtype=int)
    for seg in _segments(states, valid):
        if len(seg) > 2:
            np.add.at(N3, (seg[:-2], seg[1:-1], seg[2:]), 1)
    return N3


def markov_property_test(states, P: np.ndarray, k: int = K, valid=None, alpha: float = 0.05) -> dict:
    """Contraste de Anderson-Goodman: H0 orden 1 vs H1 orden 2.

    Para cada estado presente i se forma la tabla de contingencia
    (estado anterior h) x (estado siguiente j). Bajo H0,
    P(X_{n+1}=j | X_n=i, X_{n-1}=h) no depende de h, y
    chi^2 = sum (O - E)^2 / E  ~  chi^2 con sum_i (r_i - 1)(c_i - 1) g.l.
    """
    N3 = second_order_counts(states, k, valid)
    chi2, df = 0.0, 0
    for i in range(k):
        T = N3[:, i, :].astype(float)
        T = T[T.sum(axis=1) > 0][:, T.sum(axis=0) > 0]
        if T.shape[0] < 2 or T.shape[1] < 2:
            continue
        E = T.sum(axis=1, keepdims=True) * T.sum(axis=0, keepdims=True) / T.sum()
        chi2 += float(((T - E) ** 2 / E).sum())
        df += (T.shape[0] - 1) * (T.shape[1] - 1)
    N2 = N3.sum(axis=2, keepdims=True)
    P2 = np.where(N2 > 0, N3 / np.where(N2 == 0, 1, N2), np.nan)
    diff = np.abs(P2 - np.asarray(P)[None, :, :])
    maxdiff = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
    pval = float(stats.chi2.sf(chi2, df)) if df > 0 else 1.0
    return {"chi2": chi2, "df": df, "p_value": pval, "alpha": alpha,
            "max_abs_diff": maxdiff, "markov_ok": pval > alpha,
            "N3": N3, "P2": P2}


# ---------------------------------------------------------------------------
# Análisis completo
# ---------------------------------------------------------------------------

def analyze(states, k: int = K, valid=None) -> dict:
    states = np.asarray(states, dtype=int)
    N = transition_counts(states, k, valid)
    P = transition_matrix(N)
    pi = stationary(P)
    lo, hi = transition_ci(N)
    pi_pow, iters = stationary_power(P)
    lam = eigenvalues(P)
    return {
        "N": N, "P": P, "P_lo": lo, "P_hi": hi, "unobserved": unobserved_states(N),
        "pi": pi, "pi_eig": stationary_eig(P), "pi_power": pi_pow, "power_iters": iters,
        "mu": mean_return_times(pi), "sojourn": mean_sojourn_times(P),
        "emp_freq": empirical_distribution(states, k),
        "emp_return": empirical_return_times(states, k, valid),
        "emp_sojourn": empirical_sojourn_times(states, k, valid),
        "eigenvalues": lam, "lambda2": float(np.abs(lam[1])) if k > 1 else 0.0,
        "mixing": mixing_time(P, pi),
        "ergodicity": ergodicity_report(P),
        "markov": markov_property_test(states, P, k, valid),
        "n_obs": int(len(states)), "n_trans": int(N.sum()),
    }
