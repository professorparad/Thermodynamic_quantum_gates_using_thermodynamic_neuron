import math

import numpy as np

from .not_gate import bounded_not_response, virtual_temperature


def _normal_cdf(value, mean, sigma):
    return 0.5 * (1.0 + math.erf((value - mean) / (sigma * math.sqrt(2.0))))


def not_decoding_error(beta_out, desired_logic, beta_hot, beta_cold, noise_sigma):
    """Gaussian decoding error from paper Eqs. 24-27 for a NOT output."""

    threshold = 0.5 * (beta_hot + beta_cold)
    if desired_logic == 1:
        return _normal_cdf(threshold, beta_out, noise_sigma)
    return 1.0 - _normal_cdf(threshold, beta_out, noise_sigma)


def generate_not_tradeoff(epsilon1_list, params, noise_sigma=0.12):
    """Generate a Fig. 3C-style error versus dissipation trade-off.

    The error follows the paper's Gaussian readout model. The dissipation is
    an analytic monotone proxy proportional to the collector energy scale and
    distance from the threshold; full entropy production requires a detailed
    heat-current model.
    """

    rows = []
    inputs = [
        (params.beta_hot, 1),
        (params.beta_cold, 0),
    ]
    for epsilon1 in epsilon1_list:
        errors = []
        beta_outputs = []
        for beta1, desired_logic in inputs:
            beta_v = virtual_temperature(params.beta0, beta1, epsilon1, params.epsilon_z)
            beta_out = float(bounded_not_response(beta_v, params))
            beta_outputs.append(beta_out)
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
        dissipation_proxy = float(epsilon1 * np.mean([abs(beta - params.beta0) for beta, _ in inputs]))
        rows.append(
            {
                "epsilon1": float(epsilon1),
                "average_error": avg_error,
                "dissipation_proxy": dissipation_proxy,
                "beta_out_hot_input": beta_outputs[0],
                "beta_out_cold_input": beta_outputs[1],
            }
        )
    return rows
