import numpy as np


def structured_bosonic_spectral_density(omega, alpha, ohmic_exponent, cutoff_frequency):
    """Ohmic-family spectral density with exponential cutoff."""

    omega = np.asarray(omega, dtype=float)
    positive = np.maximum(omega, 0.0)
    return (
        2.0
        * float(alpha)
        * positive ** float(ohmic_exponent)
        * np.exp(-positive / float(cutoff_frequency))
    )

