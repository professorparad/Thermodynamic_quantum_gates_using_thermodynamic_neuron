from dataclasses import dataclass


@dataclass(frozen=True)
class SingleQubitBenchmarkParameters:
    """Parameters for the first non-Markovian benchmark."""

    epsilon: float = 1.0
    beta: float = 1.0
    alpha: float = 0.05
    ohmic_exponent: float = 1.0
    cutoff_frequency: float = 5.0
    dt: float = 0.2
    t_end: float = 0.8
    memory_time: float = 0.6
    svd_tolerance: float = 1e-5


def smoke_parameters():
    """Small parameters for a quick terminal check."""

    return SingleQubitBenchmarkParameters()


def research_parameters():
    """Larger parameters for a more serious local run."""

    return SingleQubitBenchmarkParameters(
        dt=0.1,
        t_end=3.0,
        memory_time=1.2,
        svd_tolerance=1e-6,
    )


def convergence_parameter_sets(level="quick"):
    """Phase 3 convergence grids.

    quick: fast development scan.
    strong: heavier scan for a more serious convergence decision.
    """

    quick = [
        SingleQubitBenchmarkParameters(dt=0.25, t_end=1.0, memory_time=0.5, svd_tolerance=1e-4),
        SingleQubitBenchmarkParameters(dt=0.2, t_end=1.0, memory_time=0.6, svd_tolerance=1e-5),
        SingleQubitBenchmarkParameters(dt=0.1, t_end=1.0, memory_time=0.8, svd_tolerance=1e-5),
    ]
    strong = quick + [
        SingleQubitBenchmarkParameters(dt=0.08, t_end=1.0, memory_time=1.0, svd_tolerance=5e-6),
        SingleQubitBenchmarkParameters(dt=0.05, t_end=1.0, memory_time=1.0, svd_tolerance=1e-6),
    ]
    if level == "quick":
        return quick
    if level == "strong":
        return strong
    raise ValueError(f"unknown convergence level: {level}")


def dt_convergence_parameter_sets():
    """Hold memory fixed and refine dt."""

    return [
        SingleQubitBenchmarkParameters(dt=0.2, t_end=1.0, memory_time=1.0, svd_tolerance=1e-6),
        SingleQubitBenchmarkParameters(dt=0.1, t_end=1.0, memory_time=1.0, svd_tolerance=1e-6),
        SingleQubitBenchmarkParameters(dt=0.05, t_end=1.0, memory_time=1.0, svd_tolerance=1e-6),
    ]


def memory_convergence_parameter_sets():
    """Hold dt fixed and increase memory time."""

    return [
        SingleQubitBenchmarkParameters(dt=0.1, t_end=1.0, memory_time=0.4, svd_tolerance=1e-6),
        SingleQubitBenchmarkParameters(dt=0.1, t_end=1.0, memory_time=0.6, svd_tolerance=1e-6),
        SingleQubitBenchmarkParameters(dt=0.1, t_end=1.0, memory_time=0.8, svd_tolerance=1e-6),
        SingleQubitBenchmarkParameters(dt=0.1, t_end=1.0, memory_time=1.0, svd_tolerance=1e-6),
    ]
