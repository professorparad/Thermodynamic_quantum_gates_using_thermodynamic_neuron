from pathlib import Path

import numpy as np

from .thermal_functions import fermi_occupation, inverse_fermi_occupation


def collector_energies(epsilon1, epsilon_z):
    """Return epsilon0, epsilon1, epsilon_z with epsilon_z = epsilon0 - epsilon1."""

    epsilon1 = float(epsilon1)
    epsilon_z = float(epsilon_z)
    return epsilon1 + epsilon_z, epsilon1, epsilon_z


def virtual_temperature(beta0, beta1, epsilon1, epsilon_z):
    """Paper Eq. 16 for the three-qubit collector virtual temperature."""

    epsilon0, epsilon1, epsilon_z = collector_energies(epsilon1, epsilon_z)
    return (epsilon0 / epsilon_z) * beta0 - (epsilon1 / epsilon_z) * beta1


def collector_current(beta_z, beta_v, epsilon_z, mu):
    """Paper Eq. 17 reset-model collector current."""

    return float(mu) * float(epsilon_z) * (
        fermi_occupation(beta_z, epsilon_z)
        - fermi_occupation(beta_v, epsilon_z)
    )


def modulator_current(beta_z, beta_r, epsilon_z, mu_prime):
    """Paper Eq. 19 reset-model modulator current."""

    return float(mu_prime) * float(epsilon_z) * (
        fermi_occupation(beta_z, epsilon_z)
        - fermi_occupation(beta_r, epsilon_z)
    )


def bounded_not_response(beta_v, params):
    """Paper Eq. 22 / Appendix A response bounded by beta_hot and beta_cold."""

    gz_hot = fermi_occupation(params.beta_hot, params.epsilon_z)
    gz_cold = fermi_occupation(params.beta_cold, params.epsilon_z)
    gz_virtual = fermi_occupation(beta_v, params.epsilon_z)
    q_value = gz_hot * gz_virtual + gz_cold * (1.0 - gz_virtual)
    return inverse_fermi_occupation(q_value, params.epsilon_z)


def generate_not_gate_curves(beta1_values, epsilon1_list, params):
    """Generate transfer-curve rows for several epsilon1 steepness values."""

    beta1_values = np.asarray(beta1_values, dtype=float)
    rows = []
    for epsilon1 in epsilon1_list:
        beta_v = virtual_temperature(
            params.beta0,
            beta1_values,
            epsilon1,
            params.epsilon_z,
        )
        beta_out = bounded_not_response(beta_v, params)
        for beta1, bv, bz in zip(beta1_values, beta_v, beta_out):
            rows.append(
                {
                    "epsilon1": float(epsilon1),
                    "epsilon0": float(epsilon1 + params.epsilon_z),
                    "epsilon_z": float(params.epsilon_z),
                    "beta1": float(beta1),
                    "beta_v": float(bv),
                    "beta_z_infinity": float(bz),
                }
            )
    return rows


def save_curves_csv(curves, output_path):
    """Save generated transfer curves without requiring pandas."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "epsilon1",
        "epsilon0",
        "epsilon_z",
        "beta1",
        "beta_v",
        "beta_z_infinity",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in curves:
            handle.write(",".join(f"{row[key]:.12g}" for key in headers) + "\n")
    return output_path

