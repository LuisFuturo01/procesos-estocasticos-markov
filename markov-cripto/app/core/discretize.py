import numpy as np

LABELS = ["S1 Bajista", "S2 Lateral", "S3 Alcista"]


def returns_to_states(r, bajista: float = -0.015, alcista: float = 0.015) -> np.ndarray:
    """0: R < bajista; 1: bajista <= R <= alcista; 2: R > alcista."""
    r = np.asarray(r, dtype=float)
    return np.where(r > alcista, 2, np.where(r < bajista, 0, 1))
