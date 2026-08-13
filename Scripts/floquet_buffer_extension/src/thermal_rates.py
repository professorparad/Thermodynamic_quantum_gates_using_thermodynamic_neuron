import math


def qubit_excited_population(beta, epsilon):
    """Thermal excited-state population for a qubit."""

    x = max(min(float(beta) * float(epsilon), 700.0), -700.0)
    return 1.0 / (1.0 + math.exp(x))


def thermal_jump_rates(beta, epsilon, gamma):
    """Return down/up rates satisfying qubit detailed balance."""

    p_excited = qubit_excited_population(beta, epsilon)
    down = float(gamma) * (1.0 - p_excited)
    up = float(gamma) * p_excited
    return down, up

