from dataclasses import replace
from pathlib import Path
import csv

import numpy as np

from parameters import FloquetBufferParameters
from src.floquet_model import run_comparison


def ablation_parameter_sets():
    """Mechanism-by-mechanism ablations for the Floquet-buffer bridge."""

    base = FloquetBufferParameters(t_end=18.0, num_steps=220)
    return [
        (
            "full_buffer",
            "Driven buffer with system-buffer coupling and buffer-bath dissipation.",
            base,
        ),
        (
            "no_periodic_drive",
            "Buffer present, but the Floquet drive amplitude is set to zero.",
            replace(base, drive_amplitude=0.0),
        ),
        (
            "weak_drive",
            "Buffer present with a reduced drive amplitude.",
            replace(base, drive_amplitude=0.15),
        ),
        (
            "strong_drive",
            "Buffer present with a stronger drive amplitude.",
            replace(base, drive_amplitude=0.60),
        ),
        (
            "off_resonant_drive",
            "Buffer present, but the drive frequency is detuned from the qubit scale.",
            replace(base, drive_frequency=1.60),
        ),
        (
            "weak_system_buffer_coupling",
            "Driven buffer with weaker system-buffer coupling.",
            replace(base, coupling=0.02),
        ),
        (
            "strong_system_buffer_coupling",
            "Driven buffer with stronger system-buffer coupling.",
            replace(base, coupling=0.18),
        ),
        (
            "weak_buffer_bath_contact",
            "Driven buffer with weaker dissipation into the bath.",
            replace(base, bath_gamma=0.02),
        ),
        (
            "strong_buffer_bath_contact",
            "Driven buffer with stronger dissipation into the bath.",
            replace(base, bath_gamma=0.18),
        ),
    ]


def _by_architecture(summary_rows):
    return {row["architecture"]: row for row in summary_rows}


def run_ablation_study():
    rows = []
    for label, description, params in ablation_parameter_sets():
        _, summary_rows = run_comparison(params)
        summary = _by_architecture(summary_rows)
        direct_distance = summary["direct"]["final_trace_distance"]
        buffered_distance = summary["floquet_buffer"]["final_trace_distance"]
        work = summary["floquet_buffer"]["integrated_drive_work"]
        gain = buffered_distance - direct_distance
        rows.append(
            {
                "ablation": label,
                "description": description,
                "drive_amplitude": params.drive_amplitude,
                "drive_frequency": params.drive_frequency,
                "coupling": params.coupling,
                "bath_gamma": params.bath_gamma,
                "direct_trace_distance": direct_distance,
                "buffered_trace_distance": buffered_distance,
                "trace_distance_gain": gain,
                "relative_gain": gain / direct_distance if direct_distance > 0.0 else np.nan,
                "integrated_drive_work": work,
                "absolute_work": abs(work),
                "gain_per_absolute_work": gain / abs(work) if abs(work) > 1.0e-12 else np.nan,
            }
        )
    return rows


def save_ablation_csv(rows, output_path):
    headers = [
        "ablation",
        "description",
        "drive_amplitude",
        "drive_frequency",
        "coupling",
        "bath_gamma",
        "direct_trace_distance",
        "buffered_trace_distance",
        "trace_distance_gain",
        "relative_gain",
        "integrated_drive_work",
        "absolute_work",
        "gain_per_absolute_work",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def save_ablation_report(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_gain = max(rows, key=lambda row: row["trace_distance_gain"])
    best_efficiency_candidates = [
        row for row in rows if not np.isnan(row["gain_per_absolute_work"])
    ]
    best_efficiency = max(
        best_efficiency_candidates,
        key=lambda row: row["gain_per_absolute_work"],
    )
    baseline = next(row for row in rows if row["ablation"] == "full_buffer")
    no_drive = next(row for row in rows if row["ablation"] == "no_periodic_drive")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Floquet-buffer ablation study\n")
        handle.write("=============================\n\n")
        handle.write(f"Rows evaluated: {len(rows)}\n")
        handle.write(
            "Purpose: isolate which physical ingredient drives the distinguishability "
            "gain: periodic drive, system-buffer coupling, and buffer-bath contact.\n\n"
        )
        handle.write("Default full-buffer case:\n")
        handle.write(
            "- "
            f"gain={baseline['trace_distance_gain']:.6f}, "
            f"work={baseline['integrated_drive_work']:.6f}, "
            f"buffered D={baseline['buffered_trace_distance']:.6f}\n\n"
        )
        handle.write("Drive ablation:\n")
        handle.write(
            "- "
            f"no-drive gain={no_drive['trace_distance_gain']:.6f}; "
            "if this is below the driven cases, the periodic modulation is doing "
            "real work rather than the buffer acting as a passive spacer.\n\n"
        )
        handle.write("Best absolute trace-distance gain:\n")
        handle.write(
            "- "
            f"{best_gain['ablation']}: gain={best_gain['trace_distance_gain']:.6f}, "
            f"work={best_gain['integrated_drive_work']:.6f}\n\n"
        )
        handle.write("Best gain per absolute drive-work proxy:\n")
        handle.write(
            "- "
            f"{best_efficiency['ablation']}: "
            f"gain/|work|={best_efficiency['gain_per_absolute_work']:.6f}, "
            f"gain={best_efficiency['trace_distance_gain']:.6f}\n\n"
        )
        handle.write("Interpretation caveat:\n")
        handle.write(
            "This ablation is still a Markovian buffer-bath bridge. It supports or "
            "rejects design choices, but full thermodynamic heat/work claims require "
            "an explicit bath or reaction-coordinate audit.\n"
        )
    return output_path


def plot_ablation(rows, output_path):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["ablation"] for row in rows]
    gain = [row["trace_distance_gain"] for row in rows]
    work = [row["absolute_work"] for row in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    ax1.bar(labels, gain, color="tab:blue")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_ylabel("trace-distance gain")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.bar(labels, work, color="tab:red")
    ax2.set_ylabel("|drive-work proxy|")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.tick_params(axis="x", labelrotation=35)
    for label in ax2.get_xticklabels():
        label.set_ha("right")

    fig.suptitle("Floquet Buffer Ablation Study")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
