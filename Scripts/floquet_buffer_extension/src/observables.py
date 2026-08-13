import numpy as np


def density_matrix_to_array(rho):
    """Convert QuTiP object or array-like density matrix to ndarray."""

    return rho.full() if hasattr(rho, "full") else np.asarray(rho, dtype=complex)


def purity(rho):
    """Density-matrix purity."""

    matrix = density_matrix_to_array(rho)
    return float(np.real(np.trace(matrix @ matrix)))


def trace_distance(rho_a, rho_b):
    """Trace distance between two density matrices."""

    delta = density_matrix_to_array(rho_a) - density_matrix_to_array(rho_b)
    singular_values = np.linalg.svd(delta, compute_uv=False)
    return 0.5 * float(np.sum(singular_values))

