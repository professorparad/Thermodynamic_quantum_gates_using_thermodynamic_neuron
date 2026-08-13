import os
from pathlib import Path

from sweeps.ablation_study import (
    plot_ablation,
    run_ablation_study,
    save_ablation_csv,
    save_ablation_report,
)

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))


def main():
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_ablation_study()
    csv_path = save_ablation_csv(rows, output_dir / "floquet_ablation_study.csv")
    report_path = save_ablation_report(rows, output_dir / "floquet_ablation_study_report.txt")
    plot_path = plot_ablation(rows, output_dir / "floquet_ablation_study.png")

    print("Floquet-buffer ablation study complete.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
