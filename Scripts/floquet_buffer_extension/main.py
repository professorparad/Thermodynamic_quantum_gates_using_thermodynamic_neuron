import os
from pathlib import Path

from parameters import FloquetBufferParameters
from src.floquet_model import run_comparison, save_rows_csv
from visualization.plots import plot_comparison

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))


def main():
    params = FloquetBufferParameters()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    dynamics_rows, summary_rows = run_comparison(params)
    dynamics_csv = save_rows_csv(
        dynamics_rows,
        output_dir / "floquet_buffer_dynamics.csv",
        [
            "architecture",
            "initial_state",
            "time",
            "system_excited_population",
            "system_purity",
            "drive_power",
        ],
    )
    summary_csv = save_rows_csv(
        summary_rows,
        output_dir / "floquet_buffer_summary.csv",
        ["architecture", "final_trace_distance", "integrated_drive_work"],
    )
    plot_path = plot_comparison(
        dynamics_rows,
        summary_rows,
        output_dir / "floquet_buffer_comparison.png",
    )

    print("Floquet-buffer bridge model complete.")
    print(f"Parameters: {params}")
    print(f"Saved dynamics CSV: {dynamics_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()

