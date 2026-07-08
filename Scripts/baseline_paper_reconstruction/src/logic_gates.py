import numpy as np

from .thermal_functions import fermi_occupation, inverse_fermi_occupation


def bounded_neuron_response(beta_v, beta_hot, beta_cold, epsilon_z):
    """Paper Eq. B3 response for a thermodynamic neuron."""

    gz_hot = fermi_occupation(beta_hot, epsilon_z)
    gz_cold = fermi_occupation(beta_cold, epsilon_z)
    gz_virtual = fermi_occupation(beta_v, epsilon_z)
    q_value = gz_hot * gz_virtual + gz_cold * (1.0 - gz_virtual)
    return inverse_fermi_occupation(q_value, epsilon_z)


def nor_virtual_temperature(beta1, beta2, epsilon_z=0.1, alpha=8.0):
    """Paper Eq. 46 virtual temperature for the NOR neuron."""

    return alpha * (1.0 - 2.0 * np.asarray(beta1, dtype=float) - 2.0 * np.asarray(beta2, dtype=float))


def majority_virtual_temperature(beta1, beta2, beta3, epsilon_z=0.1, alpha=8.0):
    """Paper Eq. 48 virtual temperature for the 3-majority neuron."""

    return alpha * (
        4.0
        - 3.0 * np.asarray(beta1, dtype=float)
        - 3.0 * np.asarray(beta2, dtype=float)
        - 3.0 * np.asarray(beta3, dtype=float)
    )


def perceptron_virtual_temperature(beta_inputs, bias, weights, alpha=8.0):
    """Generic linearly separable thermodynamic-neuron score."""

    total = float(bias)
    for beta, weight in zip(beta_inputs, weights):
        total += float(weight) * np.asarray(beta, dtype=float)
    return alpha * total


def generate_nor_surface(beta1_grid, beta2_grid, params, alpha=8.0):
    """Generate Fig. 6-style NOR response rows over a two-input grid."""

    beta_v = nor_virtual_temperature(
        beta1_grid,
        beta2_grid,
        epsilon_z=params.epsilon_z,
        alpha=alpha,
    )
    beta_out = bounded_neuron_response(
        beta_v,
        params.beta_hot,
        params.beta_cold,
        params.epsilon_z,
    )
    rows = []
    for beta1, beta2, bv, bz in zip(
        beta1_grid.ravel(),
        beta2_grid.ravel(),
        beta_v.ravel(),
        beta_out.ravel(),
    ):
        rows.append(
            {
                "beta1": float(beta1),
                "beta2": float(beta2),
                "beta_v": float(bv),
                "beta_z_infinity": float(bz),
            }
        )
    return rows


def generate_majority_volume(beta1_grid, beta2_grid, beta3_grid, params, alpha=8.0):
    """Generate Fig. 7-style 3-majority response data."""

    beta_v = majority_virtual_temperature(
        beta1_grid,
        beta2_grid,
        beta3_grid,
        epsilon_z=params.epsilon_z,
        alpha=alpha,
    )
    beta_out = bounded_neuron_response(
        beta_v,
        params.beta_hot,
        params.beta_cold,
        params.epsilon_z,
    )
    rows = []
    for beta1, beta2, beta3, bv, bz in zip(
        beta1_grid.ravel(),
        beta2_grid.ravel(),
        beta3_grid.ravel(),
        beta_v.ravel(),
        beta_out.ravel(),
    ):
        rows.append(
            {
                "beta1": float(beta1),
                "beta2": float(beta2),
                "beta3": float(beta3),
                "beta_v": float(bv),
                "beta_z_infinity": float(bz),
            }
        )
    return rows


def generate_xor_surface(beta1_grid, beta2_grid, params, alpha=8.0):
    """Generate a Fig. 8-style XOR response via NAND/OR feeding AND."""

    beta1 = np.asarray(beta1_grid, dtype=float)
    beta2 = np.asarray(beta2_grid, dtype=float)
    network_alpha = 10.0 * alpha

    nand_v = perceptron_virtual_temperature((beta1, beta2), bias=3.0, weights=(-2.0, -2.0), alpha=network_alpha)
    or_v = perceptron_virtual_temperature((beta1, beta2), bias=-1.0, weights=(2.0, 2.0), alpha=network_alpha)

    nand_out = bounded_neuron_response(nand_v, params.beta_hot, params.beta_cold, params.epsilon_z)
    or_out = bounded_neuron_response(or_v, params.beta_hot, params.beta_cold, params.epsilon_z)

    and_v = perceptron_virtual_temperature((nand_out, or_out), bias=-3.0, weights=(2.0, 2.0), alpha=network_alpha)
    xor_out = bounded_neuron_response(and_v, params.beta_hot, params.beta_cold, params.epsilon_z)

    rows = []
    for beta_a, beta_b, nand_beta, or_beta, final_beta in zip(
        beta1.ravel(),
        beta2.ravel(),
        nand_out.ravel(),
        or_out.ravel(),
        xor_out.ravel(),
    ):
        rows.append(
            {
                "beta1": float(beta_a),
                "beta2": float(beta_b),
                "nand_beta_z": float(nand_beta),
                "or_beta_z": float(or_beta),
                "xor_beta_z": float(final_beta),
            }
        )
    return rows
