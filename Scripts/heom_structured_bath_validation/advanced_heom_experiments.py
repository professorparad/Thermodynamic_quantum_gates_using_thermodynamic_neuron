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
    _run_buffered,
    _run_direct,
    _system_state,
    _trace_distance,
)
from parameters import HEOMValidationParameters


STATE_PAIRS = {
    "x": ("plus", "minus"),
    "y": ("plus_i", "minus_i"),
    "z": ("ground", "excited"),
}
STATE_LABELS = tuple(label for pair in STATE_PAIRS.values() for label in pair)
ARCHITECTURES = {
    "direct": (_run_direct, False),
    "floquet_buffer": (_run_buffered, True),
}


def _best_gain_params():
    return replace(
        HEOMValidationParameters(),
        drive_amplitude=0.55,
        drive_frequency=1.2,
        system_buffer_coupling=0.04,
        reorganization_energy=0.075,
        bath_cutoff=0.8,
        t_end=5.0,
        num_steps=100,
        matsubara_terms=2,
    )


def _write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run_labels(params, architecture, labels, depth):
    runner, buffered = ARCHITECTURES[architecture]
    runs = {}
    elapsed = 0.0
    for label in labels:
        start = time.perf_counter()
        tlist, states = runner(params, label, depth)
        elapsed += time.perf_counter() - start
        runs[label] = {
            "time": np.asarray(tlist, dtype=float),
            "full": states,
            "system": [_system_state(state, buffered) for state in states],
        }
    return runs, elapsed


def _pauli_transfer_matrix(final_states):
    identity, sx, sy, sz = qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()
    paulis = [identity, sx, sy, sz]
    image_ops = [
        final_states["ground"] + final_states["excited"],
        final_states["plus"] - final_states["minus"],
        final_states["plus_i"] - final_states["minus_i"],
        final_states["ground"] - final_states["excited"],
    ]
    ptm = np.zeros((4, 4), dtype=float)
    for row, pauli in enumerate(paulis):
        for col, image in enumerate(image_ops):
            ptm[row, col] = float(np.real(0.5 * (pauli * image).tr()))

    choi = np.zeros((4, 4), dtype=complex)
    for image, pauli in zip(image_ops, paulis):
        choi += 0.25 * np.kron(image.full(), pauli.full().T)
    choi = 0.5 * (choi + choi.conj().T)
    return ptm, np.linalg.eigvalsh(choi)


def run_channel_tomography(params, depth=3):
    metric_rows = []
    ptm_rows = []
    cached_runs = {}
    pauli_labels = ["I", "X", "Y", "Z"]
    target_x = np.diag([1.0, 1.0, -1.0, -1.0])

    for architecture in ARCHITECTURES:
        runs, elapsed = _run_labels(params, architecture, STATE_LABELS, depth)
        cached_runs[architecture] = runs
        finals = {label: run["system"][-1] for label, run in runs.items()}
        ptm, choi_eigenvalues = _pauli_transfer_matrix(finals)
        bloch_block = ptm[1:, 1:]
        left, singular_values, right_transpose = np.linalg.svd(bloch_block)
        orientation = np.linalg.det(left @ right_transpose)
        optimal_rotation = left @ np.diag([1.0, 1.0, orientation]) @ right_transpose
        calibrated_overlap = float(np.sum(optimal_rotation * bloch_block))
        axis_distances = {
            axis: _trace_distance(finals[positive], finals[negative])
            for axis, (positive, negative) in STATE_PAIRS.items()
        }
        metrics = {
            "architecture": architecture,
            "heom_depth": depth,
            "runtime_seconds": elapsed,
            "average_fidelity_identity": (float(np.trace(ptm)) + 2.0) / 6.0,
            "average_fidelity_optimal_unitary_frame": (3.0 + calibrated_overlap) / 6.0,
            "average_fidelity_x_diagnostic": (float(np.sum(target_x * ptm)) + 2.0) / 6.0,
            "trace_distance_x": axis_distances["x"],
            "trace_distance_y": axis_distances["y"],
            "trace_distance_z": axis_distances["z"],
            "worst_axis_trace_distance": min(axis_distances.values()),
            "mean_axis_trace_distance": float(np.mean(list(axis_distances.values()))),
            "channel_unitarity": float(np.sum(bloch_block**2) / 3.0),
            "bloch_volume_contraction_abs": float(abs(np.linalg.det(bloch_block))),
            "largest_bloch_singular_value": float(singular_values[0]),
            "smallest_bloch_singular_value": float(singular_values[-1]),
            "choi_min_eigenvalue": float(choi_eigenvalues[0]),
            "trace_preservation_residual": float(
                np.linalg.norm(ptm[0, :] - np.array([1.0, 0.0, 0.0, 0.0]))
            ),
        }
        metric_rows.append(metrics)
        for row in range(4):
            for col in range(4):
                ptm_rows.append(
                    {
                        "architecture": architecture,
                        "output_pauli": pauli_labels[row],
                        "input_pauli": pauli_labels[col],
                        "ptm_value": ptm[row, col],
                    }
                )
        metrics["_ptm"] = ptm
        metrics["_choi_eigenvalues"] = choi_eigenvalues
    return metric_rows, ptm_rows, cached_runs


def _cumulative_integral(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(times)
    return np.concatenate(([0.0], np.cumsum(increments)))


def run_backflow_and_energy(params, cached_runs):
    rows = []
    summary = []
    for architecture, runs in cached_runs.items():
        axis_backflow = {}
        for axis, (positive, negative) in STATE_PAIRS.items():
            times = runs[positive]["time"]
            distances = np.array(
                [
                    _trace_distance(rho_a, rho_b)
                    for rho_a, rho_b in zip(
                        runs[positive]["system"], runs[negative]["system"]
                    )
                ]
            )
            increments = np.diff(distances)
            cumulative_backflow = np.concatenate(
                ([0.0], np.cumsum(np.maximum(increments, 0.0)))
            )
            axis_backflow[axis] = float(cumulative_backflow[-1])
            for idx, current_time in enumerate(times):
                rows.append(
                    {
                        "architecture": architecture,
                        "axis": axis,
                        "time": current_time,
                        "trace_distance": distances[idx],
                        "cumulative_blp_backflow": cumulative_backflow[idx],
                        "instantaneous_drive_power": 0.0,
                        "cumulative_positive_drive_work": 0.0,
                        "cumulative_absolute_drive_work": 0.0,
                    }
                )

        if architecture == "floquet_buffer":
            power_times = runs["excited"]["time"]
            power = np.array(
                [
                    _drive_power(state, current_time, params)
                    for current_time, state in zip(
                        power_times, runs["excited"]["full"]
                    )
                ]
            )
            positive_work = _cumulative_integral(np.maximum(power, 0.0), power_times)
            absolute_work = _cumulative_integral(np.abs(power), power_times)
            z_rows = [
                row
                for row in rows
                if row["architecture"] == architecture and row["axis"] == "z"
            ]
            for row, current_power, supplied, throughput in zip(
                z_rows, power, positive_work, absolute_work
            ):
                row["instantaneous_drive_power"] = current_power
                row["cumulative_positive_drive_work"] = supplied
                row["cumulative_absolute_drive_work"] = throughput
        else:
            positive_work = np.array([0.0])
            absolute_work = np.array([0.0])

        z_rows = [
            row
            for row in rows
            if row["architecture"] == architecture and row["axis"] == "z"
        ]
        times = np.array([row["time"] for row in z_rows])
        distances = np.array([row["trace_distance"] for row in z_rows])
        summary.append(
            {
                "architecture": architecture,
                "blp_lower_bound_x": axis_backflow["x"],
                "blp_lower_bound_y": axis_backflow["y"],
                "blp_lower_bound_z": axis_backflow["z"],
                "sampled_blp_lower_bound": max(axis_backflow.values()),
                "integrated_z_distinguishability": float(np.trapz(distances, times)),
                "final_z_trace_distance": float(distances[-1]),
                "positive_supplied_work": float(positive_work[-1]),
                "absolute_drive_work_throughput": float(absolute_work[-1]),
            }
        )
    return rows, summary


def _run_z_pair(params, architecture, depth):
    runs, elapsed = _run_labels(params, architecture, STATE_PAIRS["z"], depth)
    distance = _trace_distance(
        runs["ground"]["system"][-1], runs["excited"]["system"][-1]
    )
    positive_work = 0.0
    absolute_work = 0.0
    net_work = 0.0
    if architecture == "floquet_buffer":
        times = runs["excited"]["time"]
        power = np.array(
            [
                _drive_power(state, current_time, params)
                for current_time, state in zip(times, runs["excited"]["full"])
            ]
        )
        net_work = float(_cumulative_integral(power, times)[-1])
        positive_work = float(_cumulative_integral(np.maximum(power, 0.0), times)[-1])
        absolute_work = float(_cumulative_integral(np.abs(power), times)[-1])
    return distance, net_work, positive_work, absolute_work, elapsed, runs


def run_stroboscopic_phase_sensitivity(params, depth=3):
    direct_distance, _, _, _, direct_runtime, _ = _run_z_pair(params, "direct", depth)
    rows = []
    for phase in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        phase_params = replace(params, drive_phase=float(phase), num_steps=80)
        distance, net_work, positive_work, absolute_work, elapsed, _ = _run_z_pair(
            phase_params, "floquet_buffer", depth
        )
        rows.append(
            {
                "drive_phase_rad": phase,
                "drive_phase_over_pi": phase / np.pi,
                "direct_trace_distance": direct_distance,
                "buffered_trace_distance": distance,
                "trace_distance_gain": distance - direct_distance,
                "drive_work_net": net_work,
                "drive_work_positive": positive_work,
                "drive_work_absolute": absolute_work,
                "gain_per_positive_work": (
                    (distance - direct_distance) / positive_work
                    if positive_work > 1e-12
                    else np.nan
                ),
                "direct_runtime_seconds": direct_runtime,
                "buffered_runtime_seconds": elapsed,
            }
        )
    return rows


def _latin_hypercube(sample_count, dimensions, seed):
    rng = np.random.default_rng(seed)
    samples = np.zeros((sample_count, dimensions), dtype=float)
    for dim in range(dimensions):
        strata = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
        samples[:, dim] = strata[rng.permutation(sample_count)]
    return 2.0 * samples - 1.0


def run_uncertainty_ensemble(params, sample_count=48, depth=3):
    names = [
        "drive_amplitude",
        "drive_frequency",
        "system_buffer_coupling",
        "reorganization_energy",
        "bath_cutoff",
        "bath_temperature",
        "drive_phase",
    ]
    spreads = np.array([0.10, 0.05, 0.15, 0.15, 0.15, 0.10, np.pi / 6.0])
    design = _latin_hypercube(sample_count, len(names), seed=20260814)
    rows = []
    regression_design = []
    gains = []

    for sample_index, unit_deviations in enumerate(design):
        values = {
            "drive_amplitude": params.drive_amplitude * (1.0 + spreads[0] * unit_deviations[0]),
            "drive_frequency": params.drive_frequency * (1.0 + spreads[1] * unit_deviations[1]),
            "system_buffer_coupling": params.system_buffer_coupling
            * (1.0 + spreads[2] * unit_deviations[2]),
            "reorganization_energy": params.reorganization_energy
            * (1.0 + spreads[3] * unit_deviations[3]),
            "bath_cutoff": params.bath_cutoff * (1.0 + spreads[4] * unit_deviations[4]),
            "bath_temperature": params.bath_temperature
            * (1.0 + spreads[5] * unit_deviations[5]),
            "drive_phase": spreads[6] * unit_deviations[6],
        }
        sample_params = replace(params, num_steps=80, **values)
        direct, _, _, _, direct_runtime, _ = _run_z_pair(sample_params, "direct", depth)
        buffered, net, positive, absolute, buffer_runtime, _ = _run_z_pair(
            sample_params, "floquet_buffer", depth
        )
        gain = buffered - direct
        row = {
            "sample": sample_index,
            **values,
            "direct_trace_distance": direct,
            "buffered_trace_distance": buffered,
            "trace_distance_gain": gain,
            "drive_work_net": net,
            "drive_work_positive": positive,
            "drive_work_absolute": absolute,
            "robust_pass": int(buffered >= 0.75 and gain > 0.0),
            "direct_runtime_seconds": direct_runtime,
            "buffered_runtime_seconds": buffer_runtime,
        }
        rows.append(row)
        regression_design.append(unit_deviations)
        gains.append(gain)

    design_matrix = np.column_stack([np.ones(sample_count), np.asarray(regression_design)])
    coefficients = np.linalg.lstsq(design_matrix, np.asarray(gains), rcond=None)[0][1:]
    sensitivity_rows = [
        {
            "parameter": name,
            "standardized_linear_effect_on_gain": coefficient,
            "absolute_effect_rank": rank,
        }
        for rank, (name, coefficient) in enumerate(
            sorted(zip(names, coefficients), key=lambda item: abs(item[1]), reverse=True),
            start=1,
        )
    ]
    return rows, sensitivity_rows


def run_convergence_matrix(params):
    depths = [2, 3, 4]
    matsubara_terms = [1, 2, 3]
    reference_runs = {}
    reference_params = replace(params, matsubara_terms=3, num_steps=80)
    for architecture in ARCHITECTURES:
        _, _, _, _, _, runs = _run_z_pair(reference_params, architecture, depth=4)
        reference_runs[architecture] = {
            label: runs[label]["system"][-1] for label in STATE_PAIRS["z"]
        }

    rows = []
    for architecture in ARCHITECTURES:
        for terms in matsubara_terms:
            for depth in depths:
                current_params = replace(params, matsubara_terms=terms, num_steps=80)
                distance, _, _, _, elapsed, runs = _run_z_pair(
                    current_params, architecture, depth
                )
                drift = max(
                    _trace_distance(
                        runs[label]["system"][-1], reference_runs[architecture][label]
                    )
                    for label in STATE_PAIRS["z"]
                )
                rows.append(
                    {
                        "architecture": architecture,
                        "heom_depth": depth,
                        "matsubara_terms": terms,
                        "final_trace_distance": distance,
                        "max_final_state_drift_vs_depth4_nk3": drift,
                        "runtime_seconds": elapsed,
                    }
                )
    return rows


def _plot_channel_tomography(metric_rows, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    for col, metrics in enumerate(metric_rows):
        image = axes[0, col].imshow(metrics["_ptm"], cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axes[0, col].set_xticks(range(4), ["I", "X", "Y", "Z"])
        axes[0, col].set_yticks(range(4), ["I", "X", "Y", "Z"])
        axes[0, col].set_xlabel("input Pauli")
        axes[0, col].set_ylabel("output Pauli")
        axes[0, col].set_title(f"{metrics['architecture']} PTM")
        for row in range(4):
            for column in range(4):
                value = metrics["_ptm"][row, column]
                axes[0, col].text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[0, :2], shrink=0.82, label="Pauli-transfer coefficient")

    labels = [row["architecture"] for row in metric_rows]
    x = np.arange(len(labels))
    width = 0.24
    for offset, axis, color in [(-width, "x", "#4c78a8"), (0.0, "y", "#f58518"), (width, "z", "#54a24b")]:
        axes[0, 2].bar(
            x + offset,
            [row[f"trace_distance_{axis}"] for row in metric_rows],
            width,
            label=axis.upper(),
            color=color,
        )
    axes[0, 2].set_xticks(x, labels, rotation=12)
    axes[0, 2].set_ylim(0.0, 1.05)
    axes[0, 2].set_ylabel("antipodal trace distance")
    axes[0, 2].set_title("Directional state retention")
    axes[0, 2].legend()

    axes[1, 0].bar(
        x - 0.24,
        [row["average_fidelity_identity"] for row in metric_rows],
        0.24,
        label="identity target",
        color="#4c78a8",
    )
    axes[1, 0].bar(
        x,
        [row["average_fidelity_optimal_unitary_frame"] for row in metric_rows],
        0.24,
        label="calibrated frame",
        color="#54a24b",
    )
    axes[1, 0].bar(
        x + 0.24,
        [row["average_fidelity_x_diagnostic"] for row in metric_rows],
        0.24,
        label="X diagnostic",
        color="#e45756",
    )
    axes[1, 0].set_xticks(x, labels, rotation=12)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylabel("average channel fidelity")
    axes[1, 0].set_title("Gate-channel diagnostics")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(
        x - 0.16,
        [row["channel_unitarity"] for row in metric_rows],
        0.32,
        label="unitarity",
        color="#72b7b2",
    )
    axes[1, 1].bar(
        x + 0.16,
        [row["bloch_volume_contraction_abs"] for row in metric_rows],
        0.32,
        label="Bloch volume",
        color="#b279a2",
    )
    axes[1, 1].set_xticks(x, labels, rotation=12)
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("Channel contraction")
    axes[1, 1].legend(fontsize=8)

    for idx, row in enumerate(metric_rows):
        axes[1, 2].plot(
            range(4),
            row["_choi_eigenvalues"],
            marker="o",
            label=row["architecture"],
        )
    axes[1, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set_xlabel("ordered Choi eigenvalue")
    axes[1, 2].set_ylabel("eigenvalue")
    axes[1, 2].set_title("Complete-positivity audit")
    axes[1, 2].legend(fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_backflow(rows, summary, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    colors = {"direct": "#4c78a8", "floquet_buffer": "#e45756"}
    for architecture in ARCHITECTURES:
        selected = [row for row in rows if row["architecture"] == architecture and row["axis"] == "z"]
        axes[0, 0].plot(
            [row["time"] for row in selected],
            [row["trace_distance"] for row in selected],
            color=colors[architecture],
            label=architecture,
        )
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("D_z(t)")
    axes[0, 0].set_title("Logical distinguishability dynamics")
    axes[0, 0].legend()

    width = 0.35
    axis_names = list(STATE_PAIRS)
    x = np.arange(3)
    for idx, item in enumerate(summary):
        axes[0, 1].bar(
            x + (idx - 0.5) * width,
            [item[f"blp_lower_bound_{axis}"] for axis in axis_names],
            width,
            color=colors[item["architecture"]],
            label=item["architecture"],
        )
    axes[0, 1].set_xticks(x, [axis.upper() for axis in axis_names])
    axes[0, 1].set_ylabel("sum of positive D increments")
    axes[0, 1].set_title("Sampled BLP information backflow")
    axes[0, 1].legend(fontsize=8)

    buffered = [row for row in rows if row["architecture"] == "floquet_buffer" and row["axis"] == "z"]
    axes[1, 0].plot(
        [row["time"] for row in buffered],
        [row["instantaneous_drive_power"] for row in buffered],
        color="#f58518",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("drive power")
    axes[1, 0].set_title("Instantaneous AC power")

    axes[1, 1].plot(
        [row["time"] for row in buffered],
        [row["cumulative_positive_drive_work"] for row in buffered],
        label="positive supplied work",
        color="#54a24b",
    )
    axes[1, 1].plot(
        [row["time"] for row in buffered],
        [row["cumulative_absolute_drive_work"] for row in buffered],
        label="absolute throughput",
        color="#b279a2",
    )
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("cumulative work proxy")
    axes[1, 1].set_title("Cancellation-safe energy accounting")
    axes[1, 1].legend(fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_phase(rows, output_path):
    import matplotlib.pyplot as plt

    phases = np.array([row["drive_phase_over_pi"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    axes[0].plot(phases, [row["buffered_trace_distance"] for row in rows], marker="o", label="buffer")
    axes[0].plot(phases, [row["direct_trace_distance"] for row in rows], linestyle="--", label="direct")
    axes[0].set_xlabel("readout phase / pi")
    axes[0].set_ylabel("final trace distance")
    axes[0].set_title("Stroboscopic phase sensitivity")
    axes[0].legend(fontsize=8)
    axes[1].plot(phases, [row["trace_distance_gain"] for row in rows], marker="o", color="#54a24b")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("readout phase / pi")
    axes[1].set_ylabel("buffer gain")
    axes[1].set_title("Gain around one drive cycle")
    axes[2].plot(phases, [row["drive_work_positive"] for row in rows], marker="o", label="Win+")
    axes[2].plot(phases, [row["drive_work_absolute"] for row in rows], marker="s", label="Wabs")
    axes[2].set_xlabel("readout phase / pi")
    axes[2].set_ylabel("work proxy")
    axes[2].set_title("Phase-dependent drive cost")
    axes[2].legend(fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_uncertainty(rows, sensitivity_rows, output_path):
    import matplotlib.pyplot as plt

    gains = np.array([row["trace_distance_gain"] for row in rows])
    buffered = np.array([row["buffered_trace_distance"] for row in rows])
    work = np.array([row["drive_work_positive"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    axes[0, 0].hist(gains, bins=12, color="#4c78a8", alpha=0.85)
    axes[0, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_xlabel("trace-distance gain")
    axes[0, 0].set_ylabel("samples")
    axes[0, 0].set_title("Calibration-uncertainty ensemble")

    sorted_gain = np.sort(gains)
    axes[0, 1].plot(sorted_gain, np.arange(1, len(sorted_gain) + 1) / len(sorted_gain), color="#e45756")
    axes[0, 1].set_xlabel("trace-distance gain")
    axes[0, 1].set_ylabel("empirical CDF")
    axes[0, 1].set_title("Gain quantiles")

    ordered = list(reversed(sensitivity_rows))
    effects = [row["standardized_linear_effect_on_gain"] for row in ordered]
    axes[1, 0].barh(
        [row["parameter"].replace("_", " ") for row in ordered],
        effects,
        color=["#54a24b" if value >= 0 else "#e45756" for value in effects],
    )
    axes[1, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("linear effect across stated tolerance")
    axes[1, 0].set_title("Local sensitivity ranking")

    scatter = axes[1, 1].scatter(work, gains, c=buffered, cmap="viridis", s=42, alpha=0.85)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("positive supplied work")
    axes[1, 1].set_ylabel("trace-distance gain")
    axes[1, 1].set_title("Robust energy-fidelity cloud")
    fig.colorbar(scatter, ax=axes[1, 1], label="buffered D")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _matrix_from_rows(rows, architecture, key):
    return np.array(
        [
            [
                next(
                    row[key]
                    for row in rows
                    if row["architecture"] == architecture
                    and row["matsubara_terms"] == terms
                    and row["heom_depth"] == depth
                )
                for depth in [2, 3, 4]
            ]
            for terms in [1, 2, 3]
        ]
    )


def _plot_convergence(rows, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    for col, architecture in enumerate(ARCHITECTURES):
        drift = _matrix_from_rows(rows, architecture, "max_final_state_drift_vs_depth4_nk3")
        runtime = _matrix_from_rows(rows, architecture, "runtime_seconds")
        image = axes[0, col].imshow(np.log10(np.maximum(drift, 1e-14)), origin="lower", cmap="magma_r")
        axes[0, col].set_title(f"{architecture}: log10 final-state drift")
        axes[1, col].imshow(runtime, origin="lower", cmap="viridis")
        axes[1, col].set_title(f"{architecture}: runtime (s)")
        for row in range(3):
            for column in range(3):
                log_value = np.log10(max(drift[row, column], 1e-14))
                drift_midpoint = 0.5 * (
                    np.log10(max(drift.min(), 1e-14))
                    + np.log10(max(drift.max(), 1e-14))
                )
                axes[0, col].text(
                    column,
                    row,
                    f"{drift[row, column]:.1e}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if log_value > drift_midpoint else "black",
                )
                axes[1, col].text(column, row, f"{runtime[row, column]:.2f}", ha="center", va="center", fontsize=8, color="white" if runtime[row, column] > 0.5 * runtime.max() else "black")
        for axis in [axes[0, col], axes[1, col]]:
            axis.set_xticks(range(3), [2, 3, 4])
            axis.set_yticks(range(3), [1, 2, 3])
            axis.set_xlabel("HEOM hierarchy depth")
            axis.set_ylabel("Matsubara terms Nk")
        fig.colorbar(image, ax=axes[0, col], shrink=0.75)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _clean_metric_rows(metric_rows):
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in metric_rows
    ]


def _write_report(path, params, metric_rows, backflow_summary, phase_rows, uncertainty_rows, sensitivity_rows, convergence_rows):
    gains = np.array([row["trace_distance_gain"] for row in uncertainty_rows])
    robust_fraction = float(np.mean([row["robust_pass"] for row in uncertainty_rows]))
    phase_gains = np.array([row["trace_distance_gain"] for row in phase_rows])
    worst_convergence = {
        architecture: max(
            row["max_final_state_drift_vs_depth4_nk3"]
            for row in convergence_rows
            if row["architecture"] == architecture
            and row["heom_depth"] == 3
            and row["matsubara_terms"] == 2
        )
        for architecture in ARCHITECTURES
    }
    direct = next(row for row in metric_rows if row["architecture"] == "direct")
    buffered = next(row for row in metric_rows if row["architecture"] == "floquet_buffer")
    direct_memory = next(row for row in backflow_summary if row["architecture"] == "direct")
    buffer_memory = next(row for row in backflow_summary if row["architecture"] == "floquet_buffer")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("Advanced HEOM channel, memory, robustness, and convergence report\n")
        handle.write("================================================================\n\n")
        handle.write(f"Operating point: {params}\n\n")
        handle.write("Six-state channel tomography (output-stage channel):\n")
        for row in [direct, buffered]:
            handle.write(
                f"- {row['architecture']}: Favg(I)={row['average_fidelity_identity']:.6f}, "
                f"Dmin={row['worst_axis_trace_distance']:.6f}, "
                f"Dmean={row['mean_axis_trace_distance']:.6f}, "
                f"Favg(calibrated)={row['average_fidelity_optimal_unitary_frame']:.6f}, "
                f"unitarity={row['channel_unitarity']:.6f}, "
                f"min eig(Choi)={row['choi_min_eigenvalue']:.3e}, "
                f"TP residual={row['trace_preservation_residual']:.3e}\n"
            )
        handle.write("\nSampled BLP information-backflow lower bound:\n")
        handle.write(
            f"- direct={direct_memory['sampled_blp_lower_bound']:.6f}; "
            f"Floquet buffer={buffer_memory['sampled_blp_lower_bound']:.6f}\n"
        )
        handle.write(
            f"- buffer positive supplied work={buffer_memory['positive_supplied_work']:.6f}; "
            f"absolute throughput={buffer_memory['absolute_drive_work_throughput']:.6f}\n"
        )
        handle.write("\nStroboscopic phase sensitivity (16 phases):\n")
        handle.write(
            f"- gain range=[{phase_gains.min():.6f}, {phase_gains.max():.6f}], "
            f"median={np.median(phase_gains):.6f}\n"
        )
        handle.write("\nLatin-hypercube calibration uncertainty (48 points):\n")
        handle.write(
            f"- robust-pass fraction={robust_fraction:.3f}; gain median={np.median(gains):.6f}; "
            f"5--95% interval=[{np.quantile(gains, 0.05):.6f}, {np.quantile(gains, 0.95):.6f}]\n"
        )
        handle.write("- ranked local effects on gain:\n")
        for row in sensitivity_rows:
            handle.write(
                f"  {row['absolute_effect_rank']}. {row['parameter']}: "
                f"{row['standardized_linear_effect_on_gain']:+.6f}\n"
            )
        handle.write("\nTwo-axis convergence audit:\n")
        handle.write(
            f"- depth=3, Nk=2 drift vs depth=4, Nk=3: "
            f"direct={worst_convergence['direct']:.3e}, "
            f"buffer={worst_convergence['floquet_buffer']:.3e}\n"
        )
        handle.write("\nScope statement:\n")
        handle.write(
            "These experiments validate the reduced output/output-buffer quantum channel. "
            "They do not by themselves constitute a full six-qubit thermodynamic-NOT HEOM run. "
            "The X-target fidelity is a diagnostic only; the identity-target fidelity measures "
            "state retention through this output-stage channel. The BLP value is a lower bound "
            "from three antipodal Pauli pairs, not a global optimization over all state pairs.\n"
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    params = _best_gain_params()

    metric_rows, ptm_rows, cached_runs = run_channel_tomography(params)
    backflow_rows, backflow_summary = run_backflow_and_energy(params, cached_runs)
    phase_rows = run_stroboscopic_phase_sensitivity(params)
    uncertainty_rows, sensitivity_rows = run_uncertainty_ensemble(params)
    convergence_rows = run_convergence_matrix(params)

    clean_metrics = _clean_metric_rows(metric_rows)
    _write_csv(OUTPUT_DIR / "heom_channel_metrics.csv", clean_metrics, list(clean_metrics[0]))
    _write_csv(OUTPUT_DIR / "heom_pauli_transfer_matrices.csv", ptm_rows, list(ptm_rows[0]))
    _write_csv(OUTPUT_DIR / "heom_backflow_energy_dynamics.csv", backflow_rows, list(backflow_rows[0]))
    _write_csv(OUTPUT_DIR / "heom_backflow_summary.csv", backflow_summary, list(backflow_summary[0]))
    _write_csv(OUTPUT_DIR / "heom_stroboscopic_phase.csv", phase_rows, list(phase_rows[0]))
    _write_csv(OUTPUT_DIR / "heom_uncertainty_ensemble.csv", uncertainty_rows, list(uncertainty_rows[0]))
    _write_csv(OUTPUT_DIR / "heom_uncertainty_sensitivity.csv", sensitivity_rows, list(sensitivity_rows[0]))
    _write_csv(OUTPUT_DIR / "heom_convergence_matrix.csv", convergence_rows, list(convergence_rows[0]))

    _plot_channel_tomography(metric_rows, OUTPUT_DIR / "heom_channel_tomography.png")
    _plot_backflow(backflow_rows, backflow_summary, OUTPUT_DIR / "heom_information_backflow.png")
    _plot_phase(phase_rows, OUTPUT_DIR / "heom_stroboscopic_phase.png")
    _plot_uncertainty(uncertainty_rows, sensitivity_rows, OUTPUT_DIR / "heom_uncertainty_robustness.png")
    _plot_convergence(convergence_rows, OUTPUT_DIR / "heom_convergence_matrix.png")
    report_path = OUTPUT_DIR / "advanced_heom_experiments_report.txt"
    _write_report(
        report_path,
        params,
        clean_metrics,
        backflow_summary,
        phase_rows,
        uncertainty_rows,
        sensitivity_rows,
        convergence_rows,
    )

    print("Advanced HEOM experiments complete.")
    print(f"Saved report: {report_path}")
    for row in clean_metrics:
        print(
            row["architecture"],
            f"Favg(I)={row['average_fidelity_identity']:.6f}",
            f"Dmin={row['worst_axis_trace_distance']:.6f}",
            f"ChoiMin={row['choi_min_eigenvalue']:.3e}",
        )
    gains = np.array([row["trace_distance_gain"] for row in uncertainty_rows])
    print(
        "uncertainty",
        f"robust={np.mean([row['robust_pass'] for row in uncertainty_rows]):.3f}",
        f"gain_5_50_95={np.quantile(gains, [0.05, 0.5, 0.95])}",
    )


if __name__ == "__main__":
    main()
