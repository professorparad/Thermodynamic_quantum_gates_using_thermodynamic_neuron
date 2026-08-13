import numpy as np


def trace_distance_qubit(rho_a, rho_b):
    """Trace distance between two 2x2 density matrices."""

    delta = np.asarray(rho_a, dtype=complex) - np.asarray(rho_b, dtype=complex)
    eigenvalues = np.linalg.eigvalsh(delta.conj().T @ delta)
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    return 0.5 * float(np.sum(singular_values))


def excited_population(rho):
    """Excited-state population for computational basis |0>, |1>."""

    rho = np.asarray(rho, dtype=complex)
    return float(np.real(rho[1, 1]))

