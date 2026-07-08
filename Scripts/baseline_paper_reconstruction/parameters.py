from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NotGateParameters:
    """Parameters for the paper's analytic NOT-gate reconstruction."""

    beta_hot: float = 1.0
    beta_cold: float = 2.0
    beta0: float = 1.5
    epsilon_z: float = 0.1
    mu: float = 1.0e-4
    mu_prime: float = 1.0e-4
    heat_capacity: float = 1.0
    beta_z_initial: float = 1.5


@dataclass(frozen=True)
class VirtualQubitParameters:
    """Parameters for the three-qubit machine regime plot."""

    beta0: float = 1.5
    epsilon0: float = 2.0
    epsilon1: float = 1.5


@dataclass(frozen=True)
class LogicRangeParameters:
    """Temperature range used to encode logical states."""

    beta_hot: float = 0.0
    beta_cold: float = 1.0
    epsilon_z: float = 0.1


def epsilon1_values():
    """Representative steepness values used to compare transfer curves."""

    return np.array([1.0, 2.0, 5.0, 10.0, 20.0], dtype=float)


def beta1_sweep(num_points=400):
    """Input inverse-temperature sweep for the NOT transfer curve."""

    return np.linspace(0.5, 2.5, num_points)


def fig2_beta1_sweep(num_points=500):
    """Sweep wide enough to show refrigerator, pump, and engine regimes."""

    return np.linspace(0.2, 4.0, num_points)


def nor_input_grid(num_points=160):
    """Two-input temperature grid for the NOR response surface."""

    values = np.linspace(0.0, 1.0, num_points)
    return np.meshgrid(values, values)


def majority_input_grid(num_points=36):
    """Three-input temperature grid for 3-majority slices."""

    values = np.linspace(0.0, 1.0, num_points)
    return np.meshgrid(values, values, values)
