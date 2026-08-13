import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import qutip as qt
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver

from parameters import HEOMValidationParameters


def _trace_distance(rho_a, rho_b):
    return float(0.5 * (rho_a - rho_b).norm("tr"))


def _excited_population(rho):
    proj = qt.basis(2, 1) * qt.basis(2, 1).dag()
    return float(np.real((proj * rho).tr()))


def _system_state(state, buffered):
    return state.ptrace(0) if buffered else state


def _drive_power(state, time, params):
    sx = qt.sigmax()
    ident = qt.qeye(2)
    sx_f = qt.tensor(ident, sx)
    op = (
        -params.drive_amplitude
        * params.drive_frequency
        * np.sin(params.drive_frequency * time + params.drive_phase)
        * sx_f
    )
    return float(np.real((op * state).tr()))


def _bath(coupling_operator, params):
    return DrudeLorentzBath(
        coupling_operator,
        lam=params.reorganization_energy,
        gamma=params.bath_cutoff,
        T=params.bath_temperature,
        Nk=params.matsubara_terms,
    )


def _initial_ket(label):
    ground = qt.basis(2, 0)
    excited = qt.basis(2, 1)
    states = {
        "ground": ground,
        "excited": excited,
        "plus": (ground + excited).unit(),
        "minus": (ground - excited).unit(),
        "plus_i": (ground + 1j * excited).unit(),
        "minus_i": (ground - 1j * excited).unit(),
    }
    try:
        return states[label]
    except KeyError as exc:
        raise ValueError(f"Unknown initial-state label: {label}") from exc


def _run_direct(params, initial_label, depth):
    sx, sz = qt.sigmax(), qt.sigmaz()
    hamiltonian = 0.5 * params.epsilon_s * sz
    bath = _bath(sx, params)
    rho0_ket = _initial_ket(initial_label)
    rho0 = rho0_ket * rho0_ket.dag()
    tlist = np.linspace(0.0, params.t_end, params.num_steps + 1)
    solver = HEOMSolver(
        hamiltonian,
        bath,
        max_depth=depth,
        options={"progress_bar": "", "store_states": True},
    )
    return tlist, solver.run(rho0, tlist).states


def _run_buffered(params, initial_label, depth):
    sx, sz, ident = qt.sigmax(), qt.sigmaz(), qt.qeye(2)
    sx_s = qt.tensor(sx, ident)
    sz_s = qt.tensor(sz, ident)
    sx_f = qt.tensor(ident, sx)
    sz_f = qt.tensor(ident, sz)

    h_static = (
        0.5 * params.epsilon_s * sz_s
        + 0.5 * params.epsilon_f * sz_f
        + params.system_buffer_coupling * sx_s * sx_f
    )
    h_drive = params.drive_amplitude * sx_f

    def drive_coeff(t, **kwargs):
        return np.cos(params.drive_frequency * t + params.drive_phase)

    hamiltonian = qt.QobjEvo([h_static, [h_drive, drive_coeff]])
    bath = _bath(sx_f, params)
    system_ket = _initial_ket(initial_label)
    buffer_ket = qt.basis(2, 0)
    psi0 = qt.tensor(system_ket, buffer_ket)
    rho0 = psi0 * psi0.dag()
    tlist = np.linspace(0.0, params.t_end, params.num_steps + 1)
    solver = HEOMSolver(
        hamiltonian,
        bath,
        max_depth=depth,
        options={"progress_bar": "", "store_states": True},
    )
    return tlist, solver.run(rho0, tlist).states


def _collect_dynamics(params, architecture, initial_label, tlist, states):
    buffered = architecture == "heom_floquet_buffer"
    rows = []
    for time, state in zip(tlist, states):
        system_state = _system_state(state, buffered)
        rows.append(
            {
                "architecture": architecture,
                "initial_state": initial_label,
                "time": float(time),
                "system_excited_population": _excited_population(system_state),
                "system_purity": float((system_state * system_state).tr().real),
                "drive_power": _drive_power(state, time, params) if buffered else 0.0,
            }
        )
    return rows


def run_heom_validation(params):
    all_rows = []
    final_states = {}
    convergence = {}
    for architecture, runner, depth in [
        ("heom_direct", _run_direct, params.direct_depth),
        ("heom_floquet_buffer", _run_buffered, params.buffered_depth),
    ]:
        for initial in ["ground", "excited"]:
            tlist, states = runner(params, initial, depth)
            all_rows.extend(_collect_dynamics(params, architecture, initial, tlist, states))
            final_states[(architecture, initial)] = _system_state(
                states[-1], architecture == "heom_floquet_buffer"
            )

            _, refined_states = runner(params, initial, params.convergence_depth)
            coarse_final = _system_state(states[-1], architecture == "heom_floquet_buffer")
            refined_final = _system_state(
                refined_states[-1], architecture == "heom_floquet_buffer"
            )
            convergence[(architecture, initial)] = _trace_distance(coarse_final, refined_final)

    summary_rows = []
    for architecture in ["heom_direct", "heom_floquet_buffer"]:
        distance = _trace_distance(
            final_states[(architecture, "ground")],
            final_states[(architecture, "excited")],
        )
        arch_rows = [row for row in all_rows if row["architecture"] == architecture]
        work = _integrate_work(arch_rows)
        pop_ground = _excited_population(final_states[(architecture, "ground")])
        pop_excited = _excited_population(final_states[(architecture, "excited")])
        summary_rows.append(
            {
                "architecture": architecture,
                "final_trace_distance": distance,
                "final_ground_input_excited_population": pop_ground,
                "final_excited_input_excited_population": pop_excited,
                "population_contrast": pop_excited - pop_ground,
                "integrated_drive_work_proxy": work,
                "max_depth_convergence_trace_distance": max(
                    convergence[(architecture, "ground")],
                    convergence[(architecture, "excited")],
                ),
            }
        )
    direct_d = summary_rows[0]["final_trace_distance"]
    buffered_d = summary_rows[1]["final_trace_distance"]
    for row in summary_rows:
        row["buffered_minus_direct_trace_distance"] = buffered_d - direct_d
    return all_rows, summary_rows


def _integrate_work(rows):
    rows = [row for row in rows if row["initial_state"] == "excited"]
    if len(rows) < 2:
        return 0.0
    times = np.array([row["time"] for row in rows], dtype=float)
    power = np.array([row["drive_power"] for row in rows], dtype=float)
    return float(np.trapz(power, times))


def _integrate_power_components(rows, initial_state="excited"):
    selected = [row for row in rows if row["initial_state"] == initial_state]
    if len(selected) < 2:
        return {"net": 0.0, "positive": 0.0, "absolute": 0.0}
    times = np.array([row["time"] for row in selected], dtype=float)
    power = np.array([row["drive_power"] for row in selected], dtype=float)
    return {
        "net": float(np.trapz(power, times)),
        "positive": float(np.trapz(np.maximum(power, 0.0), times)),
        "absolute": float(np.trapz(np.abs(power), times)),
    }


def _write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot(rows, summary_rows, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {
        ("heom_direct", "ground"): ("#1f77b4", "-"),
        ("heom_direct", "excited"): ("#1f77b4", "--"),
        ("heom_floquet_buffer", "ground"): ("#d62728", "-"),
        ("heom_floquet_buffer", "excited"): ("#d62728", "--"),
    }
    for key, (color, linestyle) in styles.items():
        architecture, initial = key
        selected = [
            row for row in rows
            if row["architecture"] == architecture and row["initial_state"] == initial
        ]
        axes[0].plot(
            [row["time"] for row in selected],
            [row["system_excited_population"] for row in selected],
            color=color,
            linestyle=linestyle,
            label=f"{architecture.replace('heom_', '')}, {initial}",
        )
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("system excited population")
    axes[0].legend(fontsize=8)
    axes[0].set_title("HEOM structured-bath dynamics")

    labels = [row["architecture"].replace("heom_", "") for row in summary_rows]
    values = [row["final_trace_distance"] for row in summary_rows]
    axes[1].bar(labels, values, color=["#1f77b4", "#d62728"])
    axes[1].set_ylabel("final trace distance")
    axes[1].set_title("Logical-state distinguishability")
    for idx, value in enumerate(values):
        axes[1].text(idx, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_report(path, params, summary_rows):
    direct, buffered = summary_rows
    gain = buffered["final_trace_distance"] - direct["final_trace_distance"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("HEOM structured-bath validation report\n")
        handle.write("======================================\n\n")
        handle.write(f"Parameters: {params}\n\n")
        for row in summary_rows:
            handle.write(f"{row['architecture']}:\n")
            handle.write(f"- final trace distance: {row['final_trace_distance']:.6f}\n")
            handle.write(f"- population contrast: {row['population_contrast']:.6f}\n")
            handle.write(
                "- depth convergence trace distance: "
                f"{row['max_depth_convergence_trace_distance']:.6e}\n"
            )
            handle.write(f"- drive-work proxy: {row['integrated_drive_work_proxy']:.6f}\n\n")
        handle.write(f"Buffered minus direct trace-distance gain: {gain:.6f}\n")
        if gain > 0:
            handle.write("Conclusion: buffered HEOM branch improves distinguishability.\n")
        else:
            handle.write("Conclusion: buffered HEOM branch does not improve this parameter point.\n")
    return path


def main():
    params = HEOMValidationParameters()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, summary_rows = run_heom_validation(params)
    dynamics_path = _write_csv(
        OUTPUT_DIR / "heom_validation_dynamics.csv",
        rows,
        [
            "architecture",
            "initial_state",
            "time",
            "system_excited_population",
            "system_purity",
            "drive_power",
        ],
    )
    summary_path = _write_csv(
        OUTPUT_DIR / "heom_validation_summary.csv",
        summary_rows,
        [
            "architecture",
            "final_trace_distance",
            "final_ground_input_excited_population",
            "final_excited_input_excited_population",
            "population_contrast",
            "integrated_drive_work_proxy",
            "max_depth_convergence_trace_distance",
            "buffered_minus_direct_trace_distance",
        ],
    )
    plot_path = _plot(rows, summary_rows, OUTPUT_DIR / "heom_validation.png")
    report_path = _write_report(OUTPUT_DIR / "heom_validation_report.txt", params, summary_rows)
    print("HEOM structured-bath validation complete.")
    print(f"Saved dynamics CSV: {dynamics_path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved report: {report_path}")
    for row in summary_rows:
        print(
            row["architecture"],
            f"D={row['final_trace_distance']:.6f}",
            f"contrast={row['population_contrast']:.6f}",
            f"conv={row['max_depth_convergence_trace_distance']:.3e}",
        )


if __name__ == "__main__":
    main()
