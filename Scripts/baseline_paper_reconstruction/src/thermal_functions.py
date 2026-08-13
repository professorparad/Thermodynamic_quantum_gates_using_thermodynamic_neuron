import numpy as np


def fermi_occupation(beta, epsilon):
    """Excited-state occupation g(beta epsilon) for a qubit Gibbs state."""

    x = np.asarray(beta, dtype=float) * float(epsilon)
    clipped = np.clip(x, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(clipped))


def inverse_fermi_occupation(probability, epsilon):
    """Inverse of g(beta epsilon), returning beta."""

    p = np.asarray(probability, dtype=float)
    clipped = np.clip(p, 1.0e-15, 1.0 - 1.0e-15)
    return np.log(1.0 / clipped - 1.0) / float(epsilon)
