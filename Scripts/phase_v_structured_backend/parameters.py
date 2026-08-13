from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseVParameters:
    """Parameters for OQuPy/TEMPO structured-bath truth-table runs."""

    beta_hot: float = 1.0
    beta_cold: float = 2.0
    beta0: float = 1.5
    epsilon1: float = 20.0
    epsilon_z: float = 0.1
    epsilon_buffer: float = 0.1
    buffer_coupling: float = 0.08
    drive_amplitude: float = 0.15
    drive_frequency: float = 1.0
    alpha: float = 0.20
    cutoff_frequency: float = 5.0
    ohmic_exponent: float = 1.0
    dt: float = 0.2
    t_end: float = 6.0
    memory_time: float = 1.0
    svd_tolerance: float = 1.0e-5


def quick_parameters():
    """Usable default PT-MPO smoke run."""

    return PhaseVParameters(
        alpha=0.03,
        buffer_coupling=0.02,
        t_end=4.0,
        dt=0.2,
        memory_time=1.0,
        svd_tolerance=1.0e-5,
    )


def research_parameters():
    """Heavier Phase V run for robustness tuning."""

    return PhaseVParameters(
        alpha=0.20,
        buffer_coupling=0.08,
        t_end=6.0,
        dt=0.2,
        memory_time=1.0,
        svd_tolerance=1.0e-5,
    )


def logical_inputs(params):
    return [
        {"input_bit": 0, "beta1": params.beta_hot, "expected_output_bit": 1},
        {"input_bit": 1, "beta1": params.beta_cold, "expected_output_bit": 0},
    ]
