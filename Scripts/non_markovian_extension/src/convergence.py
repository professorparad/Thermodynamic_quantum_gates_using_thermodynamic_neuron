from pathlib import Path

from .single_qubit_benchmark import run_oqupy_single_qubit_benchmark


def convergence_row(dt, memory_time, svd_tolerance, max_bond_dimension, observable_value):
    """Standard row format for future convergence scans."""

    return {
        "dt": float(dt),
        "memory_time": float(memory_time),
        "svd_tolerance": float(svd_tolerance),
        "max_bond_dimension": int(max_bond_dimension),
        "observable_value": float(observable_value),
    }


def run_convergence_scan(parameter_sets, pass_tolerance=1.0e-3):
    """Run a small convergence scan for the Ohmic bath."""

    rows = []
    reference = None
    previous = None
    for params in parameter_sets:
        _, summary = run_oqupy_single_qubit_benchmark(params, regimes={"ohmic": 1.0})
        final_sigma_z = summary[0]["final_sigma_z"]
        if reference is None:
            reference = final_sigma_z
        step_drift = 0.0 if previous is None else abs(final_sigma_z - previous)
        rows.append(
            {
                "dt": params.dt,
                "memory_time": params.memory_time,
                "svd_tolerance": params.svd_tolerance,
                "t_end": params.t_end,
                "final_sigma_z": final_sigma_z,
                "drift_from_first": abs(final_sigma_z - reference),
                "step_drift": step_drift,
                "pass_tolerance": pass_tolerance,
                "passes_step_tolerance": step_drift < pass_tolerance if previous is not None else False,
                "elapsed_seconds": summary[0]["elapsed_seconds"],
            }
        )
        previous = final_sigma_z
    return rows


def save_convergence_csv(rows, output_path):
    """Save convergence rows to CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dt",
        "memory_time",
        "svd_tolerance",
        "t_end",
        "final_sigma_z",
        "drift_from_first",
        "step_drift",
        "pass_tolerance",
        "passes_step_tolerance",
        "elapsed_seconds",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in headers) + "\n")
    return output_path


def convergence_verdict(rows):
    """Return a plain-language convergence verdict."""

    if len(rows) < 2:
        return "Not enough runs to judge convergence."
    last = rows[-1]
    previous = rows[-2]
    if len({row["t_end"] for row in rows}) != 1:
        return (
            "INVALID SCAN: t_end changes across rows, so final_sigma_z drift mixes "
            "time evolution with numerical convergence. Keep t_end fixed."
        )
    if last["passes_step_tolerance"]:
        return (
            "PASS: last two runs agree within "
            f"{last['pass_tolerance']:.1e} in final sigma_z "
            f"(last step drift={last['step_drift']:.3e})."
        )
    return (
        "NOT CONVERGED: last two runs differ by "
        f"{last['step_drift']:.3e}, above tolerance "
        f"{last['pass_tolerance']:.1e}. "
        "Increase memory_time, reduce dt, and tighten svd_tolerance."
    )


def save_convergence_report(rows, output_path):
    """Write a readable convergence report."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verdict = convergence_verdict(rows)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Phase 3 convergence report\n")
        handle.write("==========================\n\n")
        handle.write(verdict + "\n\n")
        handle.write("Rows:\n")
        for row in rows:
            handle.write(
                "- "
                f"dt={row['dt']}, memory={row['memory_time']}, "
                f"tol={row['svd_tolerance']}, t_end={row['t_end']}, "
                f"final_sigma_z={row['final_sigma_z']:.8f}, "
                f"step_drift={row['step_drift']:.3e}, "
                f"runtime={row['elapsed_seconds']:.2f}s\n"
            )
    return output_path
