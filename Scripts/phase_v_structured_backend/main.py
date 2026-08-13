import os
import argparse
from pathlib import Path

from parameters import quick_parameters, research_parameters
from src.oqupy_truth_table import run_phase_v_truth_table, save_csv
from visualization.plots import plot_phase_v

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))


DYNAMICS_HEADERS = [
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
    "decoder_threshold",
    "decoding_margin",
    "trace",
    "hermiticity_error",
    "purity",
    "elapsed_seconds",
]


SUMMARY_HEADERS = [
    "architecture",
    "truth_table_correct",
    "truth_table_total",
    "truth_table_accuracy",
    "final_trace_distance",
    "max_trace_deviation",
    "max_hermiticity_error",
    "elapsed_seconds_total",
]


def save_report(summary_rows, final_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Phase V PT-MPO/TEMPO structured-bath truth-table report\n")
        handle.write("=======================================================\n\n")
        handle.write("Status:\n")
        handle.write(
            "This replaces the output bath by an OQuPy TEMPO/PT-MPO structured-bath "
            "backend while preserving the same NOT truth-table decoder. The buffered "
            "case treats output qubit + driven Floquet buffer as the system and couples "
            "the structured bath to the buffer.\n\n"
        )
        handle.write("Summary:\n")
        for row in summary_rows:
            handle.write(
                "- "
                f"{row['architecture']}: accuracy={row['truth_table_accuracy']:.3f} "
                f"({row['truth_table_correct']}/{row['truth_table_total']}), "
                f"D_final={row['final_trace_distance']:.6f}, "
                f"trace_dev={row['max_trace_deviation']:.3e}, "
                f"herm_err={row['max_hermiticity_error']:.3e}\n"
            )
        handle.write("\nTruth table:\n")
        for row in final_rows:
            handle.write(
                "- "
                f"{row['architecture']}, input={row['input_bit']} -> "
                f"decoded={row['decoded_output_bit']} "
                f"(expected={row['expected_output_bit']}), "
                f"beta_eff={row['output_beta_effective']:.6f}, "
                f"threshold={row['decoder_threshold']:.6f}, "
                f"margin={row['decoding_margin']:.3e}, "
                f"correct={row['is_correct']}\n"
            )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run Phase V structured-bath backend.")
    parser.add_argument(
        "--mode",
        choices=["quick", "research"],
        default="quick",
        help="quick is a usable PT-MPO smoke run; research is heavier tuning.",
    )
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    params = quick_parameters() if args.mode == "quick" else research_parameters()
    rows, final_rows, summary_rows = run_phase_v_truth_table(params)
    prefix = f"phase_v_{args.mode}"
    dynamics_csv = save_csv(rows, output_dir / f"{prefix}_dynamics.csv", DYNAMICS_HEADERS)
    truth_csv = save_csv(final_rows, output_dir / f"{prefix}_truth_table.csv", DYNAMICS_HEADERS)
    summary_csv = save_csv(summary_rows, output_dir / f"{prefix}_summary.csv", SUMMARY_HEADERS)
    report_path = save_report(summary_rows, final_rows, output_dir / f"{prefix}_report.txt")
    plot_path = plot_phase_v(rows, output_dir / f"{prefix}_truth_table.png")
    print("Phase V structured-bath backend complete.")
    print(f"Mode: {args.mode}")
    print(f"Saved dynamics CSV: {dynamics_csv}")
    print(f"Saved truth table CSV: {truth_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved report: {report_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
