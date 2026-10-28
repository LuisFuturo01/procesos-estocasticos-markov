import numpy as np

LABELS = ["S1 Soleado/Seco", "S2 Lluvia leve", "S3 Lluvia intensa"]


def precip_to_states(mm, leve: float = 0.0, intensa: float = 5.0) -> np.ndarray:
    """0: mm <= leve (0 = seco); 1: leve < mm < intensa; 2: mm >= intensa."""
    mm = np.asarray(mm, dtype=float)
    return np.where(mm >= intensa, 2, np.where(mm > leve, 1, 0))
