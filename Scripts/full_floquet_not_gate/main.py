import os
from pathlib import Path

from parameters import FullNotGateParameters
from src.model import final_truth_rows, run_truth_table, save_csv
from visualization.plots import plot_truth_dynamics

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))


def save_report(summary_rows, final_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Full-ladder Floquet-buffer thermodynamic NOT prototype\n")
        handle.write("======================================================\n\n")
        handle.write("Status:\n")
        handle.write(
            "This is a runnable three-qubit GKSL surrogate with three bath channels "
            "and an optional driven Floquet buffer on the output branch. It produces "
            "truth-table outputs, but it is not yet a full MPS/HEOM structured-bath solver.\n\n"
        )
        handle.write("Summary:\n")
        for row in summary_rows:
            handle.write(
                "- "
                f"{row['architecture']}: accuracy={row['truth_table_accuracy']:.3f} "
                f"({row['truth_table_correct']}/{row['truth_table_total']}), "
                f"D_final={row['final_trace_distance']:.6f}, "
                f"W_proxy={row['integrated_drive_work_proxy']:.6f}\n"
            )
        handle.write("\nTruth table:\n")
        for row in final_rows:
            handle.write(
                "- "
                f"{row['architecture']}, input={row['input_bit']} -> "
                f"decoded={row['decoded_output_bit']} "
                f"(expected={row['expected_output_bit']}), "
                f"beta_out={row['output_beta_effective']:.6f}, "
                f"target_beta={row['target_output_beta']:.6f}, "
                f"correct={row['is_correct']}\n"
            )
        handle.write("\nNext backend upgrade:\n")
        handle.write(
            "Replace the GKSL output bath by OQuPy/PT-MPO or HEOM structured-bath "
            "dynamics, then rerun the same truth-table interface and convergence checks.\n"
        )
    return output_path


def main():
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    params = FullNotGateParameters()
    rows, summary_rows = run_truth_table(params)
    final_rows = final_truth_rows(rows, params)

    dynamics_csv = save_csv(
        rows,
        output_dir / "full_not_gate_dynamics.csv",
        [
            "architecture",
            "input_bit",
            "beta1",
            "expected_output_bit",
            "time",
            "beta_virtual",
            "target_output_beta",
            "output_excited_population",
            "output_beta_effective",
            "decoded_output_bit",
            "is_correct",
            "drive_power",
        ],
    )
    truth_csv = save_csv(
        final_rows,
        output_dir / "full_not_gate_truth_table.csv",
        [
            "architecture",
            "input_bit",
            "beta1",
            "expected_output_bit",
            "time",
            "beta_virtual",
            "target_output_beta",
            "output_excited_population",
            "output_beta_effective",
            "decoded_output_bit",
            "is_correct",
            "drive_power",
        ],
    )
    summary_csv = save_csv(
        summary_rows,
        output_dir / "full_not_gate_summary.csv",
        [
            "architecture",
            "truth_table_correct",
            "truth_table_total",
            "truth_table_accuracy",
            "final_trace_distance",
            "integrated_drive_work_proxy",
        ],
    )
    report_path = save_report(summary_rows, final_rows, output_dir / "full_not_gate_report.txt")
    plot_path = plot_truth_dynamics(rows, output_dir / "full_not_gate_dynamics.png")

    print("Full-ladder Floquet NOT prototype complete.")
    print(f"Saved dynamics CSV: {dynamics_csv}")
    print(f"Saved truth table CSV: {truth_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved report: {report_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
