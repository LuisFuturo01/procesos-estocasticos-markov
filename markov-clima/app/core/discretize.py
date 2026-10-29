import numpy as np

LABELS = ["S1 Seco", "S2 Lluvia leve", "S3 Lluvia intensa"]
SHORT = ["Seco", "Lluvia leve", "Lluvia intensa"]
# Ámbar / celeste / azul profundo: se distinguen también en escala de grises.
COLORS = ["#E3A21A", "#4FA3D9", "#1F4E9E"]

MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
SEASONS = {
    "Todo el año": list(range(1, 13)),
    "Época de lluvias (Dic–Mar)": [12, 1, 2, 3],
    "Época seca (May–Ago)": [5, 6, 7, 8],
    "Transición (Abr, Sep–Nov)": [4, 9, 10, 11],
}


def precip_to_states(mm, leve: float = 0.0, intensa: float = 5.0) -> np.ndarray:
    """0: mm <= leve (seco); 1: leve < mm < intensa; 2: mm >= intensa."""
    mm = np.asarray(mm, dtype=float)
    return np.where(mm >= intensa, 2, np.where(mm > leve, 1, 0))


def rules_latex(leve: float, intensa: float) -> str:
    return (r"X_t=\begin{cases}S_1\ \text{(seco)} & \text{si } R_t\le %.1f\ \text{mm}\\"
            r"S_2\ \text{(lluvia leve)} & \text{si } %.1f< R_t< %.1f\ \text{mm}\\"
            r"S_3\ \text{(lluvia intensa)} & \text{si } R_t\ge %.1f\ \text{mm}\end{cases}"
            % (leve, leve, intensa, intensa))
