import numpy as np

LABELS = ["S1 Bajista", "S2 Lateral", "S3 Alcista"]
SHORT = ["Bajista", "Lateral", "Alcista"]
# Rojo / gris / verde: convención de mercado; el gris neutro separa los extremos.
COLORS = ["#E5484D", "#8B95A5", "#30A46C"]

THRESHOLD_MODES = ["Fijo (±c %)", "Volatilidad (±z·σ)", "Terciles (estados equiprobables)"]


def returns_to_states(r, bajista: float = -0.015, alcista: float = 0.015) -> np.ndarray:
    """0: R < bajista; 1: bajista <= R <= alcista; 2: R > alcista."""
    r = np.asarray(r, dtype=float)
    return np.where(r > alcista, 2, np.where(r < bajista, 0, 1))


def volatility_thresholds(r, z: float = 0.5) -> tuple[float, float]:
    """Umbrales ±z·σ alrededor de la media: se adaptan a activos poco volátiles (divisas)."""
    r = np.asarray(r, dtype=float)
    m, s = float(r.mean()), float(r.std(ddof=1))
    return m - z * s, m + z * s


def tercile_thresholds(r) -> tuple[float, float]:
    """Percentiles 33.3 y 66.7: cada estado recibe ~1/3 de los días."""
    lo, hi = np.quantile(np.asarray(r, dtype=float), [1 / 3, 2 / 3])
    return float(lo), float(hi)


def rules_latex(bajista: float, alcista: float) -> str:
    return (r"R_t=\frac{\text{Precio}_t-\text{Precio}_{t-1}}{\text{Precio}_{t-1}},\qquad "
            r"X_t=\begin{cases}S_1\ \text{(bajista)} & \text{si } R_t< %.2f\%%\\"
            r"S_2\ \text{(lateral)} & \text{si } %.2f\%%\le R_t\le %.2f\%%\\"
            r"S_3\ \text{(alcista)} & \text{si } R_t> %.2f\%%\end{cases}"
            % (bajista * 100, bajista * 100, alcista * 100, alcista * 100))
