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

from main import (
    _collect_dynamics,
    _integrate_work,
    _run_buffered,
    _run_direct,
    _system_state,
    _trace_distance,
)
from parameters import HEOMValidationParameters
from scaling import run_mps_dimension_scaling


def _write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _trace_pair(states, buffered):
    return _trace_distance(_system_state(states["ground"], buffered), _system_state(states["excited"], buffered))


def _run_pair(runner, params, depth, buffered):
    states = {}
    rows = []
    elapsed = 0.0
    for initial in ["ground", "excited"]:
        start = time.perf_counter()
        tlist, run_states = runner(params, initial, depth)
        elapsed += time.perf_counter() - start
        states[initial] = run_states[-1]
        rows.extend(
            _collect_dynamics(
                params,
                "heom_floquet_buffer" if buffered else "heom_direct",
                initial,
                tlist,
                run_states,
            )
        )
    return _trace_pair(states, buffered), rows, elapsed


def run_heom_phase_sweep():
    base = HEOMValidationParameters()
    base = replace(base, t_end=5.0, num_steps=80, matsubara_terms=2)
    depth = 3
    rows = []
    direct_cache = {}
    grid = {
        "drive_amplitude": [0.0, 0.2, 0.35, 0.55],
        "drive_frequency": [0.8, 1.2, 1.8],
        "system_buffer_coupling": [0.04, 0.08, 0.14],
        "reorganization_energy": [0.025, 0.045, 0.075],
        "bath_cutoff": [0.8, 1.8, 3.0],
    }

    for reorg in grid["reorganization_energy"]:
        for cutoff in grid["bath_cutoff"]:
            direct_params = replace(base, reorganization_energy=reorg, bath_cutoff=cutoff)
            cache_key = (reorg, cutoff)
            direct_d, _, direct_runtime = _run_pair(_run_direct, direct_params, depth, buffered=False)
            direct_cache[cache_key] = (direct_d, direct_runtime)
            for amp in grid["drive_amplitude"]:
                for freq in grid["drive_frequency"]:
                    for coupling in grid["system_buffer_coupling"]:
                        params = replace(
                            direct_params,
                            drive_amplitude=amp,
                            drive_frequency=freq,
                            system_buffer_coupling=coupling,
                        )
                        buffer_d, dynamics_rows, buffer_runtime = _run_pair(
                            _run_buffered, params, depth, buffered=True
                        )
                        work = abs(_integrate_work(dynamics_rows))
                        gain = buffer_d - direct_d
                        efficiency = gain / work if work > 1e-12 else (np.inf if gain > 0 else 0.0)
                        rows.append(
                            {
                                "drive_amplitude": amp,
                                "drive_frequency": freq,
                                "system_buffer_coupling": coupling,
                                "reorganization_energy": reorg,
                                "bath_cutoff": cutoff,
                                "heom_depth": depth,
                                "direct_trace_distance": direct_d,
                                "buffered_trace_distance": buffer_d,
                                "trace_distance_gain": gain,
                                "drive_work_proxy_abs": work,
                                "gain_per_drive_work": efficiency,
                                "direct_runtime_seconds": direct_runtime,
                                "buffered_runtime_seconds": buffer_runtime,
                                "label": _label(gain, work),
                            }
                        )
    return rows


def _label(gain, work):
    if gain <= 0:
        return "buffer_hurts"
    if work < 1e-10:
        return "free_gain"
    if gain / work >= 1.0:
        return "efficient_gain"
    return "costly_gain"


def summarize_phase_sweep(rows):
    positive = [row for row in rows if row["trace_distance_gain"] > 0]
    best_gain = max(rows, key=lambda row: row["trace_distance_gain"])
    finite_eff = [row for row in positive if np.isfinite(row["gain_per_drive_work"])]
    best_eff = max(finite_eff, key=lambda row: row["gain_per_drive_work"]) if finite_eff else best_gain
    robust = [row for row in rows if row["buffered_trace_distance"] >= 0.75 and row["trace_distance_gain"] > 0]
    return {
        "total_points": len(rows),
        "positive_gain_points": len(positive),
        "robust_buffer_points": len(robust),
        "best_gain": best_gain,
        "best_efficiency": best_eff,
    }


def run_mps_best_operating_point(best_row):
    params = replace(
        HEOMValidationParameters(),
        system_buffer_coupling=best_row["system_buffer_coupling"],
        reorganization_energy=best_row["reorganization_energy"],
        bath_cutoff=best_row["bath_cutoff"],
        drive_amplitude=best_row["drive_amplitude"],
        drive_frequency=best_row["drive_frequency"],
    )
    return run_mps_dimension_scaling(params)


def _plot_phase_sweep(rows, mps_rows, output_path):
    import matplotlib.pyplot as plt

    gains = np.array([row["trace_distance_gain"] for row in rows], dtype=float)
    works = np.array([row["drive_work_proxy_abs"] for row in rows], dtype=float)
    amps = np.array([row["drive_amplitude"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    sc = axes[0].scatter(works, gains, c=amps, cmap="viridis", s=32, alpha=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("|drive-work proxy|")
    axes[0].set_ylabel("buffered - direct trace distance")
    axes[0].set_title("HEOM gain versus work")
    fig.colorbar(sc, ax=axes[0], label="drive amplitude")

    labels = sorted(set(row["label"] for row in rows))
    counts = [sum(row["label"] == label for row in rows) for label in labels]
    axes[1].bar(labels, counts, color="#4c78a8")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].set_ylabel("count")
    axes[1].set_title("Operating-point classes")

    for backend, color in [("mps_direct_subsystem", "#1f77b4"), ("mps_floquet_buffer", "#d62728")]:
        selected = [row for row in mps_rows if row["backend"] == backend]
        axes[2].plot(
            [row["chain_length"] for row in selected],
            [row["max_required_bond_dimension_eps1e-8"] for row in selected],
            marker="o",
            color=color,
            label=backend.replace("mps_", "").replace("_", " "),
        )
    axes[2].set_xlabel("bath-chain length")
    axes[2].set_ylabel("max required bond dimension")
    axes[2].set_title("MPS dimension at best point")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_report(path, summary, mps_rows):
    best_gain = summary["best_gain"]
    best_eff = summary["best_efficiency"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("HEOM Floquet phase-sweep report\n")
        handle.write("===================================\n\n")
        handle.write(f"Total HEOM points: {summary['total_points']}\n")
        handle.write(f"Positive-gain points: {summary['positive_gain_points']}\n")
        handle.write(f"Robust buffered points (D_buffer >= 0.75 and gain > 0): {summary['robust_buffer_points']}\n\n")
        handle.write("Best absolute HEOM gain:\n")
        _write_row(handle, best_gain)
        handle.write("\nBest finite gain per drive-work:\n")
        _write_row(handle, best_eff)
        handle.write("\nMPS dimension at best-gain point:\n")
        for row in mps_rows:
            handle.write(
                f"- {row['backend']} L={row['chain_length']}: "
                f"chi={row['max_required_bond_dimension_eps1e-8']}, "
                f"Smax={row['max_bond_entropy_bits']:.4f}, "
                f"D={row['final_trace_distance']:.6f}\n"
            )
        handle.write("\nInterpretation:\n")
        handle.write(
            "The direct branch is still the lower external-energy architecture because it has no "
            "AC drive. The Floquet buffer is valuable when distinguishability/noise margin is the "
            "priority: many HEOM points give positive gain, and the best-gain point remains modest "
            "in MPS bond dimension. Practically this is a power-for-fidelity tradeoff, not a free lunch.\n"
        )
    return path


def _write_row(handle, row):
    handle.write(
        f"- A={row['drive_amplitude']}, Omega={row['drive_frequency']}, "
        f"gSF={row['system_buffer_coupling']}, lambda={row['reorganization_energy']}, "
        f"cutoff={row['bath_cutoff']}\n"
    )
    handle.write(
        f"  D_direct={row['direct_trace_distance']:.6f}, "
        f"D_buffer={row['buffered_trace_distance']:.6f}, "
        f"gain={row['trace_distance_gain']:.6f}, "
        f"|W|={row['drive_work_proxy_abs']:.6f}, "
        f"gain/|W|={row['gain_per_drive_work']:.6f}, label={row['label']}\n"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_heom_phase_sweep()
    summary = summarize_phase_sweep(rows)
    mps_rows = run_mps_best_operating_point(summary["best_gain"])

    sweep_path = _write_csv(
        OUTPUT_DIR / "heom_floquet_phase_sweep.csv",
        rows,
        [
            "drive_amplitude",
            "drive_frequency",
            "system_buffer_coupling",
            "reorganization_energy",
            "bath_cutoff",
            "heom_depth",
            "direct_trace_distance",
            "buffered_trace_distance",
            "trace_distance_gain",
            "drive_work_proxy_abs",
            "gain_per_drive_work",
            "direct_runtime_seconds",
            "buffered_runtime_seconds",
            "label",
        ],
    )
    mps_path = _write_csv(
        OUTPUT_DIR / "mps_best_operating_point.csv",
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
    plot_path = _plot_phase_sweep(rows, mps_rows, OUTPUT_DIR / "heom_floquet_phase_summary.png")
    report_path = _write_report(OUTPUT_DIR / "heom_floquet_phase_sweep_report.txt", summary, mps_rows)

    print("HEOM Floquet phase-sweep complete.")
    print(f"Saved HEOM sweep: {sweep_path}")
    print(f"Saved MPS best-point scan: {mps_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved report: {report_path}")
    print(f"Total points: {summary['total_points']}")
    print(f"Positive gain: {summary['positive_gain_points']}")
    print(f"Robust buffered: {summary['robust_buffer_points']}")
    print("Best gain:")
    _write_row(sys.stdout, summary["best_gain"])
    print("Best efficiency:")
    _write_row(sys.stdout, summary["best_efficiency"])


if __name__ == "__main__":
    main()
