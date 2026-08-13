from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "Scripts" / "thermodynamic_audit" / "outputs"


AUDIT_ROWS = [
    {
        "quantity": "Thermal qubit occupation",
        "formula": "g(beta epsilon) = 1 / (1 + exp(beta epsilon))",
        "status": "safe",
        "regime": "Baseline Markovian reset model",
        "caveat": "This is a two-level thermal occupation, not a bosonic nbar factor.",
    },
    {
        "quantity": "Virtual inverse temperature",
        "formula": "beta_v = (epsilon0 / epsilon_z) beta0 - (epsilon1 / epsilon_z) beta1",
        "status": "safe",
        "regime": "Baseline thermodynamic neuron",
        "caveat": "Requires the collector resonance and weak-coupling interpretation.",
    },
    {
        "quantity": "Reset-model output dynamics",
        "formula": "dot(beta_z) = - beta_z^2 (j_C + j_M) / C",
        "status": "safe",
        "regime": "Analytical Markovian reconstruction",
        "caveat": "Not a microscopic strong-coupling bath simulation.",
    },
    {
        "quantity": "Reset-model entropy production",
        "formula": "Sigma = integral dot(Sigma) dt with dot(Sigma) = - sum_k beta_k j_k",
        "status": "safe-baseline-only",
        "regime": "Weak coupling, local detailed balance",
        "caveat": "Do not reuse as strong-coupling entropy production without bath audit.",
    },
    {
        "quantity": "Trace preservation",
        "formula": "|Tr rho(t) - 1|",
        "status": "safe",
        "regime": "All numerical reduced-state simulations",
        "caveat": "Numerical-health diagnostic, not a thermodynamic observable.",
    },
    {
        "quantity": "Hermiticity error",
        "formula": "||rho - rho^dagger||",
        "status": "safe",
        "regime": "All numerical reduced-state simulations",
        "caveat": "Numerical-health diagnostic.",
    },
    {
        "quantity": "Trace distance",
        "formula": "D = 1/2 ||rho_0 - rho_1||_1",
        "status": "safe",
        "regime": "Markovian and non-Markovian logic-performance studies",
        "caveat": "Operational distinguishability, not direct thermodynamic cost.",
    },
    {
        "quantity": "Floquet drive-work proxy",
        "formula": "W = integral Tr[rho(t) partial_t H_F(t)] dt",
        "status": "provisional",
        "regime": "Weak open-system Floquet bridge",
        "caveat": "Useful for screening; final work requires full energy accounting.",
    },
    {
        "quantity": "Reduced-system heat current beyond weak coupling",
        "formula": "J_alpha = Tr[D_alpha(rho) H_S]",
        "status": "unsafe-beyond-baseline",
        "regime": "Invalid as stated for strong coupling or long memory",
        "caveat": "Interaction energy and bath memory must be included explicitly.",
    },
    {
        "quantity": "Strong-coupling entropy production",
        "formula": "requires explicit bath, reaction coordinate, or Hamiltonian-of-mean-force audit",
        "status": "not-yet-implemented",
        "regime": "Full non-Markovian thermodynamic claim",
        "caveat": "This is a thesis extension target, not current evidence.",
    },
]


def save_audit_csv(rows, output_path):
    headers = ["quantity", "formula", "status", "regime", "caveat"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(_csv_cell(row[key]) for key in headers) + "\n")
    return output_path


def _csv_cell(value):
    value = str(value).replace('"', '""')
    return f'"{value}"'


def save_audit_report(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Thermodynamic mathematics and interpretation audit\n")
        handle.write("=================================================\n\n")
        handle.write("Purpose:\n")
        handle.write(
            "Separate thesis-safe claims from provisional screening metrics and "
            "quantities that require a stronger thermodynamic audit.\n\n"
        )
        for status in [
            "safe",
            "safe-baseline-only",
            "provisional",
            "unsafe-beyond-baseline",
            "not-yet-implemented",
        ]:
            selected = [row for row in rows if row["status"] == status]
            if not selected:
                continue
            handle.write(f"{status.upper()}:\n")
            for row in selected:
                handle.write(f"- {row['quantity']}: {row['formula']}\n")
                handle.write(f"  Regime: {row['regime']}\n")
                handle.write(f"  Caveat: {row['caveat']}\n")
            handle.write("\n")
        handle.write("Thesis rule:\n")
        handle.write(
            "Use trace distance, output population, purity, trace preservation, and "
            "Hermiticity as the main non-Markovian evidence. Use heat/work language "
            "only inside the baseline reset model or after an explicit bath-energy audit.\n"
        )
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = save_audit_csv(AUDIT_ROWS, OUTPUT_DIR / "thermodynamic_audit_table.csv")
    report_path = save_audit_report(AUDIT_ROWS, OUTPUT_DIR / "thermodynamic_audit_report.txt")
    print("Thermodynamic audit complete.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
