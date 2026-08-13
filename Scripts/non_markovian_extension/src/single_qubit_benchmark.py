from pathlib import Path
from time import perf_counter

import numpy as np


REGIMES = {
    "sub_ohmic": 0.5,
    "ohmic": 1.0,
    "super_ohmic": 3.0,
}


def run_oqupy_single_qubit_benchmark(params, regimes=None):
    """Run a single-qubit structured-bath benchmark with OQuPy."""

    import oqupy

    selected_regimes = regimes or REGIMES
    system_hamiltonian = 0.5 * params.epsilon * oqupy.operators.sigma("z")
    system = oqupy.System(system_hamiltonian)
    coupling_operator = 0.5 * oqupy.operators.sigma("x")
    initial_state = oqupy.operators.spin_dm("z+")
    tempo_parameters = oqupy.TempoParameters(
        dt=params.dt,
        epsrel=params.svd_tolerance,
        tcut=params.memory_time,
    )

    all_rows = []
    summary_rows = []
    for regime_name, zeta in selected_regimes.items():
        start = perf_counter()
        spectral_density = oqupy.PowerLawSD(
            alpha=params.alpha,
            cutoff=params.cutoff_frequency,
            temperature=1.0 / params.beta,
            zeta=zeta,
        )
        bath = oqupy.Bath(coupling_operator, spectral_density)
        dynamics = oqupy.tempo_compute(
            system=system,
            bath=bath,
            initial_state=initial_state,
            start_time=0.0,
            end_time=params.t_end,
            parameters=tempo_parameters,
            progress_type="silent",
        )
        elapsed_seconds = perf_counter() - start

        traces = []
        hermiticity_errors = []
        purities = []
        sigma_z_values = []

        for time, rho in zip(dynamics.times, dynamics.states):
            trace = np.trace(rho)
            hermiticity_error = np.linalg.norm(rho - rho.conj().T)
            purity = np.real(np.trace(rho @ rho))
            sigma_z = np.real(np.trace(rho @ oqupy.operators.sigma("z")))
            excited = 0.5 * (1.0 - sigma_z)
            traces.append(np.real(trace))
            hermiticity_errors.append(hermiticity_error)
            purities.append(purity)
            sigma_z_values.append(sigma_z)
            all_rows.append(
                {
                    "regime": regime_name,
                    "zeta": float(zeta),
                    "time": float(time),
                    "trace": float(np.real(trace)),
                    "hermiticity_error": float(hermiticity_error),
                    "purity": float(purity),
                    "sigma_z": float(sigma_z),
                    "excited_population": float(excited),
                }
            )

        summary_rows.append(
            {
                "regime": regime_name,
                "zeta": float(zeta),
                "max_trace_deviation": float(np.max(np.abs(np.array(traces) - 1.0))),
                "max_hermiticity_error": float(np.max(hermiticity_errors)),
                "final_purity": float(purities[-1]),
                "final_sigma_z": float(sigma_z_values[-1]),
                "elapsed_seconds": float(elapsed_seconds),
                "dt": float(params.dt),
                "memory_time": float(params.memory_time),
                "svd_tolerance": float(params.svd_tolerance),
            }
        )

    return all_rows, summary_rows


def save_benchmark_csv(rows, output_path):
    """Save benchmark rows to CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "regime",
        "zeta",
        "time",
        "trace",
        "hermiticity_error",
        "purity",
        "sigma_z",
        "excited_population",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in headers) + "\n")
    return output_path


def save_summary_csv(rows, output_path):
    """Save benchmark summary rows to CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "regime",
        "zeta",
        "max_trace_deviation",
        "max_hermiticity_error",
        "final_purity",
        "final_sigma_z",
        "elapsed_seconds",
        "dt",
        "memory_time",
        "svd_tolerance",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in headers) + "\n")
    return output_path
