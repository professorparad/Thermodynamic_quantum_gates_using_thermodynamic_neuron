from dataclasses import replace
from pathlib import Path

import numpy as np

from parameters import FloquetBufferParameters
from src.floquet_model import run_comparison, save_rows_csv


def _summary_by_architecture(summary_rows):
    return {row["architecture"]: row for row in summary_rows}


def sweep_parameter(base_params, parameter_name, values):
    """Sweep one Floquet-buffer parameter and collect decision metrics."""

    rows = []
    for value in values:
        params = replace(base_params, **{parameter_name: float(value)})
        _, summary_rows = run_comparison(params)
        summary = _summary_by_architecture(summary_rows)
        direct = summary["direct"]
        buffered = summary["floquet_buffer"]
        direct_distance = direct["final_trace_distance"]
        buffered_distance = buffered["final_trace_distance"]
        work = buffered["integrated_drive_work"]
        rows.append(
            {
                "swept_parameter": parameter_name,
                "swept_value": float(value),
                "direct_trace_distance": direct_distance,
                "buffered_trace_distance": buffered_distance,
                "trace_distance_gain": buffered_distance - direct_distance,
                "relative_trace_distance_gain": (
                    (buffered_distance - direct_distance) / direct_distance
                    if direct_distance > 0.0
                    else np.nan
                ),
                "integrated_drive_work": work,
                "gain_per_work": (
                    (buffered_distance - direct_distance) / abs(work)
                    if abs(work) > 1.0e-12
                    else np.nan
                ),
            }
        )
    return rows


def default_sweep_rows():
    """Small, thesis-friendly sweep around the current bridge parameters."""

    base = FloquetBufferParameters(num_steps=220, t_end=18.0)
    rows = []
    rows.extend(sweep_parameter(base, "drive_amplitude", [0.0, 0.15, 0.3, 0.45, 0.6]))
    rows.extend(sweep_parameter(base, "drive_frequency", [0.5, 0.8, 1.0, 1.2, 1.6]))
    rows.extend(sweep_parameter(base, "coupling", [0.02, 0.05, 0.08, 0.12, 0.18]))
    return rows


def save_sweep_csv(rows, output_path):
    headers = [
        "swept_parameter",
        "swept_value",
        "direct_trace_distance",
        "buffered_trace_distance",
        "trace_distance_gain",
        "relative_trace_distance_gain",
        "integrated_drive_work",
        "gain_per_work",
    ]
    return save_rows_csv(rows, Path(output_path), headers)


def best_rows(rows):
    """Return useful best-case rows for a compact report."""

    positive = [row for row in rows if row["trace_distance_gain"] > 0.0]
    if not positive:
        return {}
    return {
        "max_gain": max(positive, key=lambda row: row["trace_distance_gain"]),
        "max_gain_per_work": max(
            [row for row in positive if not np.isnan(row["gain_per_work"])],
            key=lambda row: row["gain_per_work"],
        ),
    }


def save_sweep_report(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best = best_rows(rows)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Floquet-buffer parameter sweep report\n")
        handle.write("=====================================\n\n")
        handle.write(f"Rows evaluated: {len(rows)}\n")
        positive_count = sum(row["trace_distance_gain"] > 0.0 for row in rows)
        handle.write(f"Rows with improved trace distance: {positive_count}\n\n")
        if best:
            max_gain = best["max_gain"]
            max_eff = best["max_gain_per_work"]
            handle.write("Best absolute trace-distance gain:\n")
            handle.write(
                "- "
                f"{max_gain['swept_parameter']}={max_gain['swept_value']}: "
                f"gain={max_gain['trace_distance_gain']:.6f}, "
                f"work={max_gain['integrated_drive_work']:.6f}\n\n"
            )
            handle.write("Best gain per unit drive-work proxy:\n")
            handle.write(
                "- "
                f"{max_eff['swept_parameter']}={max_eff['swept_value']}: "
                f"gain/work={max_eff['gain_per_work']:.6f}, "
                f"gain={max_eff['trace_distance_gain']:.6f}\n\n"
            )
            handle.write(
                "Caveat: this is a Markovian buffer-bath bridge. Treat gain/work as "
                "a screening metric, not a final thermodynamic proof.\n"
            )
        else:
            handle.write(
                "No positive gain found. The next step would be a wider drive-frequency "
                "and coupling sweep before attempting a full gate.\n"
            )
    return output_path
