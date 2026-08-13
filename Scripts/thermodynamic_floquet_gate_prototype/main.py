import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "Scripts" / "baseline_paper_reconstruction"
FLOQUET_OUTPUTS = ROOT / "Scripts" / "floquet_buffer_extension" / "outputs"
DECISION_OUTPUTS = ROOT / "Scripts" / "project_decision" / "outputs"
AUDIT_OUTPUTS = ROOT / "Scripts" / "thermodynamic_audit" / "outputs"

sys.path.insert(0, str(BASELINE_DIR))

from parameters import NotGateParameters, epsilon1_values  # noqa: E402
from src.not_gate import bounded_not_response, virtual_temperature  # noqa: E402

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "Scripts" / "thermodynamic_floquet_gate_prototype" / "outputs" / ".matplotlib"),
)


def _read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def _floquet_gain():
    rows = _read_csv(FLOQUET_OUTPUTS / "floquet_buffer_summary.csv")
    by_arch = {row.get("architecture"): row for row in rows}
    direct = _as_float(by_arch.get("direct", {}), "final_trace_distance")
    buffered = _as_float(by_arch.get("floquet_buffer", {}), "final_trace_distance")
    return buffered - direct, _as_float(by_arch.get("floquet_buffer", {}), "integrated_drive_work")


def _best_ablation_gain():
    rows = _read_csv(FLOQUET_OUTPUTS / "floquet_ablation_study.csv")
    if not rows:
        return 0.0, "", 0.0
    best = max(rows, key=lambda row: _as_float(row, "trace_distance_gain"))
    return (
        _as_float(best, "trace_distance_gain"),
        best.get("ablation", ""),
        _as_float(best, "integrated_drive_work"),
    )


def logical_margin_for_epsilon(epsilon1, params):
    """Compute baseline NOT output separation between hot and cold logical inputs."""

    hot_input = params.beta_hot
    cold_input = params.beta_cold
    beta_v_hot_input = virtual_temperature(params.beta0, hot_input, epsilon1, params.epsilon_z)
    beta_v_cold_input = virtual_temperature(params.beta0, cold_input, epsilon1, params.epsilon_z)
    hot_input_output = bounded_not_response(beta_v_hot_input, params)
    cold_input_output = bounded_not_response(beta_v_cold_input, params)
    return {
        "epsilon1": float(epsilon1),
        "hot_input_output_beta": float(hot_input_output),
        "cold_input_output_beta": float(cold_input_output),
        "baseline_output_margin": abs(float(hot_input_output - cold_input_output)),
    }


def design_rows():
    params = NotGateParameters()
    gain, work = _floquet_gain()
    best_ablation_gain, best_ablation_name, best_ablation_work = _best_ablation_gain()
    rows = []
    for epsilon1 in epsilon1_values():
        row = logical_margin_for_epsilon(epsilon1, params)
        row["floquet_trace_distance_gain"] = gain
        row["floquet_drive_work_proxy"] = work
        row["best_ablation_trace_distance_gain"] = best_ablation_gain
        row["best_ablation_name"] = best_ablation_name
        row["best_ablation_drive_work_proxy"] = best_ablation_work
        row["combined_screening_score"] = row["baseline_output_margin"] * max(gain, 0.0)
        row["optimistic_ablation_score"] = row["baseline_output_margin"] * max(
            best_ablation_gain,
            0.0,
        )
        row["recommended_for_full_gate"] = (
            row["baseline_output_margin"] > 0.5 and gain > 0.0 and abs(work) < 2.0
        )
        rows.append(row)
    return rows


def save_design_csv(rows, output_path):
    headers = [
        "epsilon1",
        "hot_input_output_beta",
        "cold_input_output_beta",
        "baseline_output_margin",
        "floquet_trace_distance_gain",
        "floquet_drive_work_proxy",
        "best_ablation_trace_distance_gain",
        "best_ablation_name",
        "best_ablation_drive_work_proxy",
        "combined_screening_score",
        "optimistic_ablation_score",
        "recommended_for_full_gate",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in headers) + "\n")
    return output_path


def save_design_report(rows, output_path):
    best = max(rows, key=lambda row: row["combined_screening_score"])
    optimistic = max(rows, key=lambda row: row["optimistic_ablation_score"])
    recommended = [row for row in rows if row["recommended_for_full_gate"]]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Thermodynamic Floquet NOT-gate prototype design report\n")
        handle.write("======================================================\n\n")
        handle.write(
            "This is a screening report that combines the baseline analytical NOT "
            "response with the current Floquet-buffer bridge metric.\n\n"
        )
        handle.write(f"Best epsilon1 by combined score: {best['epsilon1']}\n")
        handle.write(f"Baseline output margin: {best['baseline_output_margin']:.6f}\n")
        handle.write(f"Floquet trace-distance gain: {best['floquet_trace_distance_gain']:.6f}\n")
        handle.write(f"Drive-work proxy: {best['floquet_drive_work_proxy']:.6f}\n")
        handle.write(f"Combined screening score: {best['combined_screening_score']:.6f}\n\n")
        handle.write("Best ablation-supported optimistic setting:\n")
        handle.write(f"- epsilon1={optimistic['epsilon1']}\n")
        handle.write(f"- ablation case={optimistic['best_ablation_name']}\n")
        handle.write(
            f"- optimistic score={optimistic['optimistic_ablation_score']:.6f}\n\n"
        )
        handle.write("Recommended epsilon1 values for first full-gate attempt:\n")
        if recommended:
            for row in recommended:
                handle.write(f"- epsilon1={row['epsilon1']}\n")
        else:
            handle.write("- none yet; improve buffer gain or reduce drive-work proxy first\n")
        handle.write("\nNext programming task:\n")
        handle.write(
            "Implement a minimal NOT gate with one structured output bath and compare "
            "direct versus Floquet-buffered output distinguishability. Keep heat/work "
            "claims provisional until the thermodynamic audit is implemented.\n"
        )
        handle.write("\nMathematical audit rule:\n")
        handle.write(
            "Use trace distance and logical output margin as thesis-safe evidence. "
            "Use drive work as a proxy until explicit bath-energy accounting is added.\n"
        )
    return output_path


def save_minimal_not_experiment(rows, output_path):
    """Write a compact thesis-facing minimal NOT experiment specification."""

    recommended = [row for row in rows if row["recommended_for_full_gate"]]
    chosen = max(recommended or rows, key=lambda row: row["combined_screening_score"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Minimal Floquet-buffered thermodynamic NOT experiment\n")
        handle.write("====================================================\n\n")
        handle.write("Objective:\n")
        handle.write(
            "Compare a direct output-reservoir contact with a Floquet-buffered "
            "output-reservoir contact for the smallest NOT-style gate prototype.\n\n"
        )
        handle.write("Chosen baseline setting:\n")
        handle.write(f"- epsilon1={chosen['epsilon1']}\n")
        handle.write(f"- hot-input output beta={chosen['hot_input_output_beta']:.6f}\n")
        handle.write(f"- cold-input output beta={chosen['cold_input_output_beta']:.6f}\n")
        handle.write(f"- baseline output margin={chosen['baseline_output_margin']:.6f}\n\n")
        handle.write("Buffered/direct evidence to carry into the prototype:\n")
        handle.write(f"- default Floquet gain={chosen['floquet_trace_distance_gain']:.6f}\n")
        handle.write(f"- default drive-work proxy={chosen['floquet_drive_work_proxy']:.6f}\n")
        handle.write(
            f"- best ablation case={chosen['best_ablation_name']} "
            f"with gain={chosen['best_ablation_trace_distance_gain']:.6f}\n\n"
        )
        handle.write("Required observables:\n")
        handle.write("- final trace distance between logical outputs\n")
        handle.write("- output excited-state population or output beta proxy\n")
        handle.write("- purity and positivity/minimum eigenvalue\n")
        handle.write("- trace and Hermiticity error\n")
        handle.write("- drive-work proxy, labelled provisional\n\n")
        handle.write("Stop criterion:\n")
        handle.write(
            "Proceed toward the full three-qubit gate only if the buffered prototype "
            "improves distinguishability over direct contact and remains converged under "
            "time-step and memory/bath-contact checks.\n"
        )
    return output_path


def plot_design(rows, output_path):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = [row["epsilon1"] for row in rows]
    margin = [row["baseline_output_margin"] for row in rows]
    score = [row["combined_screening_score"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(x, margin, marker="o", label="baseline output margin")
    ax1.plot(x, score, marker="s", label="combined screening score")
    ax1.set_xlabel("epsilon1")
    ax1.set_ylabel("screening metric")
    ax1.grid(True, alpha=0.25)
    ax1.legend()
    fig.suptitle("Thermodynamic Floquet NOT Prototype Screening")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = design_rows()
    csv_path = save_design_csv(rows, output_dir / "not_floquet_design_table.csv")
    report_path = save_design_report(rows, output_dir / "not_floquet_design_report.txt")
    experiment_path = save_minimal_not_experiment(
        rows,
        output_dir / "minimal_not_experiment_spec.txt",
    )
    plot_path = plot_design(rows, output_dir / "not_floquet_design.png")

    print("Thermodynamic Floquet NOT prototype screening complete.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved experiment spec: {experiment_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
