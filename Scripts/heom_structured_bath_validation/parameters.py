from dataclasses import dataclass


@dataclass(frozen=True)
class HEOMValidationParameters:
    """Small but nontrivial HEOM benchmark parameters."""

    epsilon_s: float = 1.0
    epsilon_f: float = 1.0
    system_buffer_coupling: float = 0.08
    drive_amplitude: float = 0.35
    drive_frequency: float = 1.2
    drive_phase: float = 0.0

    bath_temperature: float = 0.75
    reorganization_energy: float = 0.045
    bath_cutoff: float = 1.8
    matsubara_terms: int = 2

    direct_depth: int = 4
    buffered_depth: int = 4
    convergence_depth: int = 5

    t_end: float = 8.0
    num_steps: int = 160
