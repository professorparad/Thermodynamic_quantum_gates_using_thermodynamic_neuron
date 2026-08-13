import csv
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import qutip as qt

from main import (
    _drive_power,
    _excited_population,
    _run_buffered,
    _run_direct,
    _system_state,
    _trace_distance,
)
from parameters import HEOMValidationParameters


def _write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_heom_depth_scaling(base_params):
    rows = []
    reference = {}
    ref_params = replace(base_params, t_end=6.0, num_steps=100)
    for architecture, runner in [
        ("heom_direct", _run_direct),
        ("heom_floquet_buffer", _run_buffered),
    ]:
        for initial in ["ground", "excited"]:
            _, states = runner(ref_params, initial, 5)
            reference[(architecture, initial)] = _system_state(
                states[-1], architecture == "heom_floquet_buffer"
            )

    for depth in [2, 3, 4, 5]:
        for architecture, runner in [
            ("heom_direct", _run_direct),
            ("heom_floquet_buffer", _run_buffered),
        ]:
            final_states = {}
            elapsed = 0.0
            drift = 0.0
            for initial in ["ground", "excited"]:
                start = time.perf_counter()
                _, states = runner(ref_params, initial, depth)
                elapsed += time.perf_counter() - start
                final = _system_state(states[-1], architecture == "heom_floquet_buffer")
                final_states[initial] = final
                drift = max(drift, _trace_distance(final, reference[(architecture, initial)]))
            rows.append(
                {
                    "backend": architecture,
                    "depth": depth,
                    "runtime_seconds": elapsed,
                    "final_trace_distance": _trace_distance(
                        final_states["ground"], final_states["excited"]
                    ),
                    "depth5_final_state_drift": drift,
                }
            )
    return rows


def _tensor_on_site(op, site, total_sites):
    ident = qt.qeye(2)
    factors = [ident] * total_sites
    factors[site] = op
    return qt.tensor(factors)


def _chain_hamiltonian(architecture, chain_length, params):
    sx, sz = qt.sigmax(), qt.sigmaz()
    total_sites = chain_length + (2 if architecture == "mps_floquet_buffer" else 1)
    hamiltonian = 0
    hamiltonian += 0.5 * params.epsilon_s * _tensor_on_site(sz, 0, total_sites)
    first_bath_site = 1
    if architecture == "mps_floquet_buffer":
        first_bath_site = 2
        hamiltonian += 0.5 * params.epsilon_f * _tensor_on_site(sz, 1, total_sites)
        hamiltonian += params.system_buffer_coupling * (
            _tensor_on_site(sx, 0, total_sites) * _tensor_on_site(sx, 1, total_sites)
        )
    for site in range(first_bath_site, total_sites):
        hamiltonian += 0.5 * 0.9 * _tensor_on_site(sz, site, total_sites)
    contact_site = 1 if architecture == "mps_floquet_buffer" else 0
    hamiltonian += params.reorganization_energy * 2.0 * (
        _tensor_on_site(sx, contact_site, total_sites)
        * _tensor_on_site(sx, first_bath_site, total_sites)
    )
    for site in range(first_bath_site, total_sites - 1):
        coupling = 0.10 / (1.0 + 0.25 * (site - first_bath_site))
        hamiltonian += coupling * (
            _tensor_on_site(sx, site, total_sites)
            * _tensor_on_site(sx, site + 1, total_sites)
        )
    return hamiltonian, total_sites


def _initial_state(architecture, chain_length, initial):
    total_sites = chain_length + (2 if architecture == "mps_floquet_buffer" else 1)
    states = [qt.basis(2, 1) if initial == "excited" else qt.basis(2, 0)]
    if architecture == "mps_floquet_buffer":
        states.append(qt.basis(2, 0))
    states.extend(qt.basis(2, 0) for _ in range(chain_length))
    return qt.tensor(states)


def _reduced_system_dm(state, architecture):
    return state.ptrace(0)


def _bond_stats(state, total_sites, eps=1e-8):
    vector = np.asarray(state.full()).reshape(-1)
    max_rank = 1
    max_entropy = 0.0
    for cut in range(1, total_sites):
        left_dim = 2**cut
        right_dim = 2 ** (total_sites - cut)
        matrix = vector.reshape(left_dim, right_dim)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        probs = singular_values**2
        probs = probs / probs.sum()
        cumulative_tail = np.cumsum(probs[::-1])[::-1]
        rank = len(probs)
        for idx, tail in enumerate(cumulative_tail):
            if tail <= eps:
                rank = idx
                break
        rank = max(1, rank)
        entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
        max_rank = max(max_rank, rank)
        max_entropy = max(max_entropy, entropy)
    return max_rank, max_entropy


def run_mps_dimension_scaling(params):
    rows = []
    tlist = np.linspace(0.0, 6.0, 81)
    for chain_length in [2, 3, 4, 5, 6]:
        for architecture in ["mps_direct_subsystem", "mps_floquet_buffer"]:
            hamiltonian, total_sites = _chain_hamiltonian(architecture, chain_length, params)
            final_states = {}
            max_rank = 1
            max_entropy = 0.0
            elapsed = 0.0
            for initial in ["ground", "excited"]:
                psi0 = _initial_state(architecture, chain_length, initial)
                start = time.perf_counter()
                result = qt.sesolve(
                    hamiltonian, psi0, tlist, options={"progress_bar": "", "store_states": True}
                )
                elapsed += time.perf_counter() - start
                for state in result.states[::10]:
                    rank, entropy = _bond_stats(state, total_sites)
                    max_rank = max(max_rank, rank)
                    max_entropy = max(max_entropy, entropy)
                final_states[initial] = _reduced_system_dm(result.states[-1], architecture)
            rows.append(
                {
                    "backend": architecture,
                    "chain_length": chain_length,
                    "total_sites": total_sites,
                    "runtime_seconds": elapsed,
                    "max_required_bond_dimension_eps1e-8": max_rank,
                    "max_bond_entropy_bits": max_entropy,
                    "final_trace_distance": _trace_distance(
                        final_states["ground"], final_states["excited"]
                    ),
                }
            )
    return rows


def energy_comparison(base_params, heom_summary_rows, mps_rows):
    heom_direct = next(row for row in heom_summary_rows if row["backend"] == "heom_direct" and row["depth"] == 4)
    heom_buffer = next(
        row for row in heom_summary_rows if row["backend"] == "heom_floquet_buffer" and row["depth"] == 4
    )
    mps_direct = next(
        row for row in mps_rows
        if row["backend"] == "mps_direct_subsystem" and row["chain_length"] == 6
    )
    mps_buffer = next(
        row for row in mps_rows
        if row["backend"] == "mps_floquet_buffer" and row["chain_length"] == 6
    )
    work_params = replace(base_params, t_end=6.0, num_steps=100)
    times, states = _run_buffered(work_params, "excited", depth=4)
    power = np.array(
        [_drive_power(state, current_time, work_params) for current_time, state in zip(times, states)]
    )
    drive_work_net = float(np.trapz(power, times))
    drive_work_positive = float(np.trapz(np.maximum(power, 0.0), times))
    drive_work_absolute = float(np.trapz(np.abs(power), times))
    return [
        {
            "architecture": "direct_structured_subsystem",
            "drive_work_net": 0.0,
            "drive_work_positive": 0.0,
            "drive_work_absolute": 0.0,
            "heom_trace_distance_depth4": heom_direct["final_trace_distance"],
            "mps_trace_distance_chain6": mps_direct["final_trace_distance"],
            "absolute_work_per_heom_trace_distance": 0.0,
            "practical_energy_note": "No active drive, lowest external energy, but weaker output distinguishability.",
        },
        {
            "architecture": "floquet_buffered_subsystem",
            "drive_work_net": drive_work_net,
            "drive_work_positive": drive_work_positive,
            "drive_work_absolute": drive_work_absolute,
            "heom_trace_distance_depth4": heom_buffer["final_trace_distance"],
            "mps_trace_distance_chain6": mps_buffer["final_trace_distance"],
            "absolute_work_per_heom_trace_distance": (
                drive_work_absolute / heom_buffer["final_trace_distance"]
            ),
            "practical_energy_note": "Consumes clock/drive work, but gives stronger isolation and distinguishability.",
        },
    ]


def _plot_scaling(heom_rows, mps_rows, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for backend, color in [("heom_direct", "#1f77b4"), ("heom_floquet_buffer", "#d62728")]:
        selected = [row for row in heom_rows if row["backend"] == backend]
        axes[0].plot(
            [row["depth"] for row in selected],
            [row["depth5_final_state_drift"] for row in selected],
            marker="o",
            color=color,
            label=backend.replace("heom_", ""),
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("HEOM hierarchy depth")
    axes[0].set_ylabel("trace-distance drift vs depth 5")
    axes[0].set_title("HEOM convergence")
    axes[0].legend()

    for backend, color in [("mps_direct_subsystem", "#1f77b4"), ("mps_floquet_buffer", "#d62728")]:
        selected = [row for row in mps_rows if row["backend"] == backend]
        axes[1].plot(
            [row["chain_length"] for row in selected],
            [row["max_required_bond_dimension_eps1e-8"] for row in selected],
            marker="s",
            color=color,
            label=backend.replace("mps_", "").replace("_", " "),
        )
    axes[1].set_xlabel("bath-chain length")
    axes[1].set_ylabel("max required bond dimension")
    axes[1].set_title("MPS bond-dimension scaling")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_report(path, heom_rows, mps_rows, energy_rows):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("HEOM and MPS scaling report\n")
        handle.write("===========================\n\n")
        handle.write("HEOM depth scaling:\n")
        for row in heom_rows:
            handle.write(
                f"- {row['backend']} depth={row['depth']}: "
                f"D={row['final_trace_distance']:.6f}, "
                f"drift={row['depth5_final_state_drift']:.3e}, "
                f"runtime={row['runtime_seconds']:.3f}s\n"
            )
        handle.write("\nExact-state Schmidt-rank scaling (MPS representability proxy):\n")
        for row in mps_rows:
            handle.write(
                f"- {row['backend']} L={row['chain_length']}: "
                f"chi={row['max_required_bond_dimension_eps1e-8']}, "
                f"Smax={row['max_bond_entropy_bits']:.4f} bits, "
                f"D={row['final_trace_distance']:.6f}\n"
            )
        handle.write("\nEnergy/work comparison:\n")
        for row in energy_rows:
            handle.write(
                f"- {row['architecture']}: Wnet={row['drive_work_net']:.6f}, "
                f"Win+={row['drive_work_positive']:.6f}, "
                f"Wabs={row['drive_work_absolute']:.6f}, "
                f"D_HEOM={row['heom_trace_distance_depth4']:.6f}, "
                f"Wabs/D={row['absolute_work_per_heom_trace_distance']:.6f}. "
                f"{row['practical_energy_note']}\n"
            )
    return path


def main():
    params = HEOMValidationParameters()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heom_rows = run_heom_depth_scaling(params)
    mps_rows = run_mps_dimension_scaling(params)
    energy_rows = energy_comparison(params, heom_rows, mps_rows)

    heom_path = _write_csv(
        OUTPUT_DIR / "heom_depth_scaling.csv",
        heom_rows,
        ["backend", "depth", "runtime_seconds", "final_trace_distance", "depth5_final_state_drift"],
    )
    mps_path = _write_csv(
        OUTPUT_DIR / "mps_dimension_scaling.csv",
        mps_rows,
        [
            "backend",
            "chain_length",
            "total_sites",
            "runtime_seconds",
            "max_required_bond_dimension_eps1e-8",
            "max_bond_entropy_bits",
            "final_trace_distance",
        ],
    )
    energy_path = _write_csv(
        OUTPUT_DIR / "energy_comparison.csv",
        energy_rows,
        [
            "architecture",
            "drive_work_net",
            "drive_work_positive",
            "drive_work_absolute",
            "heom_trace_distance_depth4",
            "mps_trace_distance_chain6",
            "absolute_work_per_heom_trace_distance",
            "practical_energy_note",
        ],
    )
    plot_path = _plot_scaling(heom_rows, mps_rows, OUTPUT_DIR / "heom_mps_scaling.png")
    report_path = _write_report(OUTPUT_DIR / "heom_mps_scaling_report.txt", heom_rows, mps_rows, energy_rows)

    print("HEOM + MPS scaling complete.")
    print(f"Saved HEOM scaling: {heom_path}")
    print(f"Saved MPS scaling: {mps_path}")
    print(f"Saved energy comparison: {energy_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved report: {report_path}")
    for row in energy_rows:
        print(
            row["architecture"],
            f"Wabs={row['drive_work_absolute']:.6f}",
            f"D_HEOM={row['heom_trace_distance_depth4']:.6f}",
            f"Wabs/D={row['absolute_work_per_heom_trace_distance']:.6f}",
        )


if __name__ == "__main__":
    main()
