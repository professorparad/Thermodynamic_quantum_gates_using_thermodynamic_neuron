import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
FULL_GATE_ROOT = PROJECT_ROOT / "Scripts" / "full_floquet_not_gate"
sys.path.append(str(FULL_GATE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

from parameters import FullNotGateParameters  # noqa: E402
from src.model import final_truth_rows, run_truth_table, save_csv  # noqa: E402
from visualization.plots import (  # noqa: E402
    plot_drive_slices,
    plot_phase_3d,
    plot_phase_classification,
    plot_phase_maps,
)


OUTPUT_DIR = ROOT / "outputs"

PHASE_ROWS = [
    "drive_amplitude",
    "buffer_coupling",
    "coupling_scale",
    "architecture",
    "truth_table_accuracy",
    "min_margin",
    "final_trace_distance",
    "integrated_drive_work_proxy",
    "phase_label",
]

COMPARISON_ROWS = [
    "drive_amplitude",
    "buffer_coupling",
    "coupling_scale",
    "direct_accuracy",
    "buffered_accuracy",
    "direct_min_margin",
    "buffered_min_margin",
    "margin_gain",
    "trace_distance_gain",
    "buffered_work_proxy",
    "three_phase_label",
]


def min_margin(final_rows, architecture, params):
    threshold = 0.5 * (params.beta_hot + params.beta_cold)
    selected = [row for row in final_rows if row["architecture"] == architecture]
    return min(abs(row["output_beta_effective"] - threshold) for row in selected)


def phase_label(accuracy, margin, robust_margin):
    if accuracy < 1.0:
        return "fail"
    if margin < robust_margin:
        return "fragile_pass"
    return "robust_pass"


def summary_by_architecture(summary_rows):
    return {row["architecture"]: row for row in summary_rows}


def run_point(base_params, drive_amplitude, buffer_coupling, coupling_scale, robust_margin):
    params = replace(
        base_params,
        drive_amplitude=drive_amplitude,
        buffer_coupling=buffer_coupling,
        input_gamma=base_params.input_gamma * coupling_scale,
        output_gamma=base_params.output_gamma * coupling_scale,
        buffer_gamma=base_params.buffer_gamma * coupling_scale,
    )
    rows, summary_rows = run_truth_table(params)
    final_rows = final_truth_rows(rows, params)
    summary = summary_by_architecture(summary_rows)

    direct = summary["direct_three_qubit"]
    buffered = summary["floquet_buffered_three_qubit"]
    direct_margin = min_margin(final_rows, "direct_three_qubit", params)
    buffered_margin = min_margin(final_rows, "floquet_buffered_three_qubit", params)

    phase_rows = [
        {
            "drive_amplitude": drive_amplitude,
            "buffer_coupling": buffer_coupling,
            "coupling_scale": coupling_scale,
            "architecture": "direct_three_qubit",
            "truth_table_accuracy": direct["truth_table_accuracy"],
            "min_margin": direct_margin,
            "final_trace_distance": direct["final_trace_distance"],
            "integrated_drive_work_proxy": direct["integrated_drive_work_proxy"],
            "phase_label": phase_label(direct["truth_table_accuracy"], direct_margin, robust_margin),
        },
        {
            "drive_amplitude": drive_amplitude,
            "buffer_coupling": buffer_coupling,
            "coupling_scale": coupling_scale,
            "architecture": "floquet_buffered_three_qubit",
            "truth_table_accuracy": buffered["truth_table_accuracy"],
            "min_margin": buffered_margin,
            "final_trace_distance": buffered["final_trace_distance"],
            "integrated_drive_work_proxy": buffered["integrated_drive_work_proxy"],
            "phase_label": phase_label(buffered["truth_table_accuracy"], buffered_margin, robust_margin),
        },
    ]
    if buffered["truth_table_accuracy"] < 1.0:
        three_phase = "fail"
    elif buffered_margin < robust_margin:
        three_phase = "fragile_pass"
    else:
        three_phase = "robust_pass"
    comparison = {
        "drive_amplitude": drive_amplitude,
        "buffer_coupling": buffer_coupling,
        "coupling_scale": coupling_scale,
        "direct_accuracy": direct["truth_table_accuracy"],
        "buffered_accuracy": buffered["truth_table_accuracy"],
        "direct_min_margin": direct_margin,
        "buffered_min_margin": buffered_margin,
        "margin_gain": buffered_margin - direct_margin,
        "trace_distance_gain": buffered["final_trace_distance"] - direct["final_trace_distance"],
        "buffered_work_proxy": buffered["integrated_drive_work_proxy"],
        "three_phase_label": three_phase,
    }
    return phase_rows, comparison


def run_phase_diagram():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_params = FullNotGateParameters(t_end=80.0, num_steps=360)
    robust_margin = 0.05
    drive_amplitudes = [0.0, 0.1, 0.2, 0.35]
    buffer_couplings = [0.0, 0.01, 0.02, 0.04, 0.08]
    coupling_scales = [0.5, 1.0, 1.5, 2.0]

    phase_rows = []
    comparison_rows = []
    total = len(drive_amplitudes) * len(buffer_couplings) * len(coupling_scales)
    done = 0
    for drive_amplitude in drive_amplitudes:
        for buffer_coupling in buffer_couplings:
            for coupling_scale in coupling_scales:
                done += 1
                print(
                    f"[{done}/{total}] A={drive_amplitude:.3f}, "
                    f"g_SF={buffer_coupling:.3f}, scale={coupling_scale:.2f}",
                    flush=True,
                )
                rows, comparison = run_point(
                    base_params,
                    drive_amplitude,
                    buffer_coupling,
                    coupling_scale,
                    robust_margin,
                )
                phase_rows.extend(rows)
                comparison_rows.append(comparison)

    phase_csv = save_csv(phase_rows, OUTPUT_DIR / "robustness_phase_diagram.csv", PHASE_ROWS)
    comparison_csv = save_csv(
        comparison_rows,
        OUTPUT_DIR / "robustness_phase_comparison.csv",
        COMPARISON_ROWS,
    )
    maps_path = plot_phase_maps(comparison_rows, OUTPUT_DIR / "robustness_phase_maps.png")
    class_path = plot_phase_classification(
        comparison_rows,
        OUTPUT_DIR / "robustness_three_phase_classification.png",
    )
    phase_3d_path = plot_phase_3d(comparison_rows, OUTPUT_DIR / "robustness_phase_3d.png")
    gif_path = plot_drive_slices(comparison_rows, OUTPUT_DIR / "robustness_drive_slices.gif")
    report_path = save_report(comparison_rows, robust_margin, OUTPUT_DIR / "robustness_phase_report.txt")

    print("Robustness phase-diagram sweep complete.")
    print(f"Saved architecture CSV: {phase_csv}")
    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved heatmaps: {maps_path}")
    print(f"Saved three-phase plot: {class_path}")
    print(f"Saved 3D phase plot: {phase_3d_path}")
    print(f"Saved GIF: {gif_path}")
    print(f"Saved report: {report_path}")


def save_report(rows, robust_margin, output_path):
    robust = [row for row in rows if row["three_phase_label"] == "robust_pass"]
    fragile = [row for row in rows if row["three_phase_label"] == "fragile_pass"]
    failed = [row for row in rows if row["three_phase_label"] == "fail"]
    best_margin = max(rows, key=lambda row: row["buffered_min_margin"])
    best_gain = max(rows, key=lambda row: row["margin_gain"])
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Floquet-buffer robustness phase-diagram report\n")
        handle.write("================================================\n\n")
        handle.write(
            "This is a GKSL strong-coupling surrogate phase map for the three-qubit "
            "thermodynamic NOT architecture. It complements, but does not replace, "
            "the Phase V PT-MPO/TEMPO structured-bath backend.\n\n"
        )
        handle.write(f"Robust margin threshold: {robust_margin:.6f}\n")
        handle.write(f"Total points: {len(rows)}\n")
        handle.write(f"Robust pass points: {len(robust)}\n")
        handle.write(f"Fragile pass points: {len(fragile)}\n")
        handle.write(f"Fail points: {len(failed)}\n\n")
        handle.write("Best buffered margin:\n")
        handle.write(
            f"- A={best_margin['drive_amplitude']}, g_SF={best_margin['buffer_coupling']}, "
            f"scale={best_margin['coupling_scale']}, margin={best_margin['buffered_min_margin']:.6f}, "
            f"accuracy={best_margin['buffered_accuracy']:.3f}\n\n"
        )
        handle.write("Best buffered-minus-direct margin gain:\n")
        handle.write(
            f"- A={best_gain['drive_amplitude']}, g_SF={best_gain['buffer_coupling']}, "
            f"scale={best_gain['coupling_scale']}, gain={best_gain['margin_gain']:.6f}, "
            f"work_proxy={best_gain['buffered_work_proxy']:.6f}\n\n"
        )
        handle.write("Interpretation:\n")
        handle.write(
            "The three phases are fail, fragile pass, and robust pass. A pass is "
            "fragile when the NOT truth table is correct but the minimum thermal "
            "decoder margin is below the chosen threshold. This is the same "
            "distinction used in CMOS noise-margin analysis: correctness at a point "
            "is weaker than a separated logic level.\n"
        )
    return output_path


if __name__ == "__main__":
    run_phase_diagram()
