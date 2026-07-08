import numpy as np


def virtual_temperature_two_qubit(beta0, beta1, epsilon0, epsilon1):
    """Paper Eq. 11 virtual temperature for C0-C1."""

    epsilon_v = float(epsilon0) - float(epsilon1)
    return (float(epsilon0) / epsilon_v) * beta0 - (float(epsilon1) / epsilon_v) * beta1


def classify_machine_regime(beta_v, beta_z):
    """Classify collector behavior relative to the output bath temperature."""

    beta_v = np.asarray(beta_v, dtype=float)
    labels = np.full(beta_v.shape, "heat_pump", dtype=object)
    labels[beta_v > beta_z] = "refrigerator"
    labels[beta_v < 0.0] = "heat_engine"
    return labels


def generate_virtual_temperature_curve(beta1_values, params):
    """Generate Fig. 2-style virtual-temperature data."""

    beta1_values = np.asarray(beta1_values, dtype=float)
    beta_v = virtual_temperature_two_qubit(
        params.beta0,
        beta1_values,
        params.epsilon0,
        params.epsilon1,
    )
    labels = classify_machine_regime(beta_v, params.beta0)
    return [
        {
            "beta1": float(beta1),
            "beta_v": float(bv),
            "regime": str(regime),
        }
        for beta1, bv, regime in zip(beta1_values, beta_v, labels)
    ]

