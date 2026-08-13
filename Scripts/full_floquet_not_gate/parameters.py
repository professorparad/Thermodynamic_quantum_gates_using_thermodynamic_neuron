from dataclasses import dataclass


@dataclass(frozen=True)
class FullNotGateParameters:
    """Parameters for the three-qubit thermodynamic NOT prototype."""

    beta_hot: float = 1.0
    beta_cold: float = 2.0
    beta0: float = 1.5
    epsilon1: float = 20.0
    epsilon_z: float = 0.1
    epsilon_buffer: float = 0.1
    input_gamma: float = 0.20
    output_gamma: float = 0.12
    buffer_gamma: float = 0.12
    buffer_coupling: float = 0.02
    drive_amplitude: float = 0.15
    drive_frequency: float = 1.0
    t_end: float = 120.0
    num_steps: int = 800


def logical_inputs(params):
    """Logical-temperature encoding used for the NOT table."""

    return [
        {"input_bit": 0, "beta1": params.beta_hot, "expected_output_bit": 1},
        {"input_bit": 1, "beta1": params.beta_cold, "expected_output_bit": 0},
    ]
