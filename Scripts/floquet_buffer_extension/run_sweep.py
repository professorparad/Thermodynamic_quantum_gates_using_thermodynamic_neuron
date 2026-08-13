import os
from pathlib import Path

from sweeps.floquet_parameter_sweep import (
    default_sweep_rows,
    save_sweep_csv,
    save_sweep_report,
)
from sweeps.plot_sweep import plot_sweep

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))


def main():
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = default_sweep_rows()
    csv_path = save_sweep_csv(rows, output_dir / "floquet_parameter_sweep.csv")
    report_path = save_sweep_report(rows, output_dir / "floquet_parameter_sweep_report.txt")
    plot_path = plot_sweep(rows, output_dir / "floquet_parameter_sweep.png")

    print("Floquet-buffer parameter sweep complete.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
