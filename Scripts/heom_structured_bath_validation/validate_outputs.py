import csv
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"


def _read_csv(name):
    with (OUTPUT_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row, key):
    return float(row[key])


def _check_work_bounds(rows, net_key, positive_key, absolute_key):
    for row in rows:
        net = _number(row, net_key)
        positive = _number(row, positive_key)
        absolute = _number(row, absolute_key)
        assert absolute + 1e-10 >= abs(net), (net, absolute)
        assert absolute + 1e-10 >= positive, (positive, absolute)
        assert positive >= -1e-12


def main():
    checks = []

    phase = _read_csv("heom_floquet_phase_sweep.csv")
    assert len(phase) == 324
    positive_gain = sum(_number(row, "trace_distance_gain") > 0.0 for row in phase)
    robust = sum(
        _number(row, "buffered_trace_distance") >= 0.75
        and _number(row, "trace_distance_gain") > 0.0
        for row in phase
    )
    assert positive_gain == 284
    assert robust == 229
    _check_work_bounds(phase, "drive_work_net", "drive_work_positive", "drive_work_absolute")
    checks.append(f"phase sweep: 324 points, {positive_gain} positive, {robust} robust")

    channel = _read_csv("heom_channel_metrics.csv")
    assert len(channel) == 2
    direct = next(row for row in channel if row["architecture"] == "direct")
    buffered = next(row for row in channel if row["architecture"] == "floquet_buffer")
    for row in channel:
        assert _number(row, "trace_preservation_residual") < 1e-12
        assert _number(row, "choi_min_eigenvalue") >= -1e-10
        assert 0.0 <= _number(row, "average_fidelity_optimal_unitary_frame") <= 1.0 + 1e-10
    assert _number(buffered, "worst_axis_trace_distance") > _number(
        direct, "worst_axis_trace_distance"
    )
    assert _number(buffered, "average_fidelity_optimal_unitary_frame") > 0.99
    checks.append("channel: CPTP audit passed and calibrated buffered fidelity > 0.99")

    uncertainty = _read_csv("heom_uncertainty_ensemble.csv")
    assert len(uncertainty) == 48
    assert all(int(row["robust_pass"]) == 1 for row in uncertainty)
    _check_work_bounds(
        uncertainty, "drive_work_net", "drive_work_positive", "drive_work_absolute"
    )
    gains = sorted(_number(row, "trace_distance_gain") for row in uncertainty)
    assert gains[0] > 0.0
    checks.append(f"uncertainty: 48/48 robust, minimum gain {gains[0]:.6f}")

    convergence = _read_csv("heom_convergence_matrix.csv")
    assert len(convergence) == 18
    assert all(math.isfinite(_number(row, "runtime_seconds")) for row in convergence)
    checks.append("convergence: complete 2x3x3 depth/Matsubara matrix")

    energy = _read_csv("energy_comparison.csv")
    assert len(energy) == 2
    _check_work_bounds(energy, "drive_work_net", "drive_work_positive", "drive_work_absolute")
    checks.append("energy: net/supplied/absolute work inequalities passed")

    figures = [
        "heom_channel_tomography.png",
        "heom_information_backflow.png",
        "heom_stroboscopic_phase.png",
        "heom_uncertainty_robustness.png",
        "heom_convergence_matrix.png",
        "heom_energy_fidelity_frontier.png",
    ]
    for figure in figures:
        path = OUTPUT_DIR / figure
        assert path.exists() and path.stat().st_size > 10_000, figure
    checks.append(f"figures: {len(figures)} publication plots present and nonempty")

    report_path = OUTPUT_DIR / "heom_output_validation_report.txt"
    report_path.write_text(
        "HEOM generated-output acceptance checks\n"
        "=======================================\n\n"
        + "\n".join(f"PASS: {check}" for check in checks)
        + "\n",
        encoding="utf-8",
    )
    print(report_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
