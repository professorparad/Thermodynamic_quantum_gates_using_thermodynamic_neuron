import math

import numpy as np

from .not_gate import bounded_not_response, collector_current, modulator_current, virtual_temperature
from .thermal_functions import fermi_occupation, inverse_fermi_occupation


def _normal_cdf(value, mean, sigma):
    return 0.5 * (1.0 + math.erf((value - mean) / (sigma * math.sqrt(2.0))))


def not_decoding_error(beta_out, desired_logic, beta_hot, beta_cold, noise_sigma):
    """Gaussian decoding error from paper Eqs. 24-27 for a NOT output."""

    threshold = 0.5 * (beta_hot + beta_cold)
    if desired_logic == 1:
        return _normal_cdf(threshold, beta_out, noise_sigma)
    return 1.0 - _normal_cdf(threshold, beta_out, noise_sigma)


def generate_not_tradeoff(epsilon1_list, params, noise_sigma=0.12):
    """Generate Fig. 3C error versus reset-model entropy production."""

    rows = []
    inputs = [
        (params.beta_hot, 1),
        (params.beta_cold, 0),
    ]
    for epsilon1 in epsilon1_list:
        errors = []
        beta_outputs = []
        entropy_values = []
        for beta1, desired_logic in inputs:
            beta_v = virtual_temperature(params.beta0, beta1, epsilon1, params.epsilon_z)
            beta_out = float(bounded_not_response(beta_v, params))
            beta_outputs.append(beta_out)
            entropy_values.append(integrated_reset_entropy(beta_v, params)["entropy"])
            errors.append(
                not_decoding_error(
                    beta_out,
                    desired_logic,
                    params.beta_hot,
                    params.beta_cold,
                    noise_sigma,
                )
            )
        avg_error = float(np.mean(errors))
        avg_entropy = float(np.mean(entropy_values))
        rows.append(
            {
                "epsilon1": float(epsilon1),
                "average_error": avg_error,
                "average_entropy_production": avg_entropy,
                "beta_out_hot_input": beta_outputs[0],
                "beta_out_cold_input": beta_outputs[1],
            }
        )
    return rows


def reset_entropy_rate(beta_z, beta_v, params):
    """Entropy-production rate for the reset-current NOT model.

    The collector and modulator are treated as effective thermal contacts that
    try to pull the finite output reservoir toward beta_v and beta_r. This is
    the entropy accounting appropriate to the reduced reset model.
    """

    beta_r, mu_prime = modulator_design_from_bounds(params)
    j_collector = collector_current(beta_z, beta_v, params.epsilon_z, params.mu)
    j_modulator = modulator_current(beta_z, beta_r, params.epsilon_z, mu_prime)
    sigma_collector = (beta_v - beta_z) * j_collector
    sigma_modulator = (beta_r - beta_z) * j_modulator
    return max(0.0, float(sigma_collector + sigma_modulator))


def beta_z_derivative(beta_z, beta_v, params):
    """Finite output-reservoir ODE from the reset model."""

    beta_r, mu_prime = modulator_design_from_bounds(params)
    j_collector = collector_current(beta_z, beta_v, params.epsilon_z, params.mu)
    j_modulator = modulator_current(beta_z, beta_r, params.epsilon_z, mu_prime)
    return -((beta_z**2) / params.heat_capacity) * (j_collector + j_modulator)


def modulator_design_from_bounds(params):
    """Return beta_r and mu_prime consistent with the bounded NOT response."""

    g_hot = fermi_occupation(params.beta_hot, params.epsilon_z)
    g_cold = fermi_occupation(params.beta_cold, params.epsilon_z)
    delta = float(g_hot - g_cold)
    if not 0.0 < delta < 1.0:
        raise ValueError("hot/cold bounds do not define a valid modulator design")
    mu_prime = params.mu * (1.0 - delta) / delta
    g_r = g_cold / (1.0 - delta)
    beta_r = float(inverse_fermi_occupation(g_r, params.epsilon_z))
    return beta_r, float(mu_prime)


def integrated_reset_entropy(beta_v, params, steps=4000):
    """Integrate entropy production along the finite-bath beta trajectory."""

    beta_start = float(params.beta_z_initial)
    beta_final = float(bounded_not_response(beta_v, params))
    if abs(beta_final - beta_start) < 1.0e-12:
        return {"beta_z_final": beta_final, "entropy": 0.0}

    beta_path = np.linspace(beta_start, beta_final, steps, endpoint=False)
    beta_path = np.append(beta_path, beta_final - np.sign(beta_final - beta_start) * 1.0e-10)
    integrand = []
    for beta_z in beta_path:
        rate = reset_entropy_rate(beta_z, beta_v, params)
        speed = abs(beta_z_derivative(beta_z, beta_v, params))
        if speed < 1.0e-30:
            integrand.append(0.0)
        else:
            integrand.append(rate / speed)
    entropy = abs(float(np.trapz(integrand, beta_path)))
    return {
        "beta_z_final": beta_final,
        "entropy": entropy,
    }
