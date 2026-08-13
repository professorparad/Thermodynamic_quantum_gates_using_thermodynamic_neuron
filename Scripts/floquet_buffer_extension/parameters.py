from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FloquetBufferParameters:
    """Parameters for the first Floquet-buffer bridge model."""

    epsilon_s: float = 1.0
    epsilon_f: float = 1.0
    coupling: float = 0.08
    drive_amplitude: float = 0.35
    drive_frequency: float = 1.0
    bath_beta: float = 1.0
    bath_gamma: float = 0.08
    direct_gamma: float = 0.08
    t_end: float = 25.0
    num_steps: int = 350


def time_grid(params):
    """Simulation time grid."""

    return np.linspace(0.0, params.t_end, params.num_steps + 1)

