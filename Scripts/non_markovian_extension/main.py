import os
import argparse
from pathlib import Path

from parameters import (
    convergence_parameter_sets,
    dt_convergence_parameter_sets,
    memory_convergence_parameter_sets,
    research_parameters,
    smoke_parameters,
)
from src.convergence import run_convergence_scan, save_convergence_csv, save_convergence_report
from src.single_qubit_benchmark import (
    run_oqupy_single_qubit_benchmark,
    save_benchmark_csv,
    save_summary_csv,
)
from visualization.plots import plot_convergence_scan, plot_regime_comparison

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))


def main():
    parser = argparse.ArgumentParser(description="Run non-Markovian extension tasks.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "research", "convergence", "dt-scan", "memory-scan"],
        default="smoke",
        help="smoke is quick; research is heavier; convergence modes run Phase 3 scans.",
    )
    parser.add_argument(
        "--convergence-level",
        choices=["quick", "strong"],
        default="quick",
        help="quick is fast; strong adds heavier rows for a better convergence decision.",
    )
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "convergence":
        rows = run_convergence_scan(convergence_parameter_sets(args.convergence_level))
        prefix = f"convergence_scan_{args.convergence_level}"
        csv_path = save_convergence_csv(rows, output_dir / f"{prefix}.csv")
        report_path = save_convergence_report(rows, output_dir / f"{prefix}_report.txt")
        plot_path = plot_convergence_scan(rows, output_dir / f"{prefix}.png")
        print("Phase 3 convergence scan complete.")
        print(f"Level: {args.convergence_level}")
        print(f"Saved CSV: {csv_path}")
        print(f"Saved report: {report_path}")
        print(f"Saved plot: {plot_path}")
        return

    if args.mode in {"dt-scan", "memory-scan"}:
        parameter_sets = (
            dt_convergence_parameter_sets()
            if args.mode == "dt-scan"
            else memory_convergence_parameter_sets()
        )
        rows = run_convergence_scan(parameter_sets)
        prefix = args.mode.replace("-", "_")
        csv_path = save_convergence_csv(rows, output_dir / f"{prefix}.csv")
        report_path = save_convergence_report(rows, output_dir / f"{prefix}_report.txt")
        plot_path = plot_convergence_scan(rows, output_dir / f"{prefix}.png")
        print(f"Phase 3 {args.mode} complete.")
        print(f"Saved CSV: {csv_path}")
        print(f"Saved report: {report_path}")
        print(f"Saved plot: {plot_path}")
        return

    params = smoke_parameters() if args.mode == "smoke" else research_parameters()
    regimes = {"ohmic": 1.0} if args.mode == "smoke" else None
    rows, summary_rows = run_oqupy_single_qubit_benchmark(params, regimes=regimes)
    name = f"single_qubit_{args.mode}"
    dynamics_csv = save_benchmark_csv(rows, output_dir / f"{name}_dynamics.csv")
    summary_csv = save_summary_csv(summary_rows, output_dir / f"{name}_summary.csv")
    plot_path = plot_regime_comparison(rows, output_dir / f"{name}_regime_comparison.png")

    print("Single-qubit non-Markovian benchmark complete.")
    print(f"Mode: {args.mode}")
    print(f"Default parameters: {params}")
    print(f"Saved dynamics CSV: {dynamics_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
