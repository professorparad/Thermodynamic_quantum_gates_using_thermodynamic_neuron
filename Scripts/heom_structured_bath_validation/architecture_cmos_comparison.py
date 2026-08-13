import csv
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
BASELINE_CURVES = (
    ROOT / "Scripts" / "baseline_paper_reconstruction" / "outputs" / "not_gate_transfer_curves.csv"
)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _passive_lookup(rows):
    lookup = {}
    for row in rows:
        if row["drive_amplitude"] == 0.0:
            key = (
                row["system_buffer_coupling"],
                row["reorganization_energy"],
                row["bath_cutoff"],
            )
            lookup[key] = row
    return lookup


def build_architecture_data(rows):
    passive = _passive_lookup(rows)
    driven_increment_rows = []
    for row in rows:
        if row["drive_amplitude"] == 0.0:
            continue
        key = (
            row["system_buffer_coupling"],
            row["reorganization_energy"],
            row["bath_cutoff"],
        )
        passive_row = passive[key]
        driven_increment_rows.append(
            {
                "drive_amplitude": row["drive_amplitude"],
                "drive_frequency": row["drive_frequency"],
                "system_buffer_coupling": row["system_buffer_coupling"],
                "reorganization_energy": row["reorganization_energy"],
                "bath_cutoff": row["bath_cutoff"],
                "direct_trace_distance": row["direct_trace_distance"],
                "passive_buffer_trace_distance": passive_row["buffered_trace_distance"],
                "floquet_buffer_trace_distance": row["buffered_trace_distance"],
                "passive_gain_over_direct": (
                    passive_row["buffered_trace_distance"] - row["direct_trace_distance"]
                ),
                "floquet_increment_over_passive": (
                    row["buffered_trace_distance"] - passive_row["buffered_trace_distance"]
                ),
                "floquet_gain_over_direct": row["trace_distance_gain"],
                "drive_work_absolute": row["drive_work_absolute"],
                "increment_per_absolute_work": (
                    (row["buffered_trace_distance"] - passive_row["buffered_trace_distance"])
                    / row["drive_work_absolute"]
                    if row["drive_work_absolute"] > 1e-12
                    else np.nan
                ),
            }
        )

    reference_key = (0.04, 0.075, 0.8)
    reference_passive = passive[reference_key]
    reference_driven = [
        row
        for row in rows
        if (
            row["system_buffer_coupling"],
            row["reorganization_energy"],
            row["bath_cutoff"],
        )
        == reference_key
        and row["drive_frequency"] == 1.2
    ]
    best_efficiency = max(
        (row for row in rows if row["drive_amplitude"] > 0.0),
        key=lambda row: row["gain_per_absolute_work"],
    )
    best_gain = max(rows, key=lambda row: row["trace_distance_gain"])
    summary_rows = [
        {
            "architecture": "direct_autonomous_contact",
            "trace_distance": reference_passive["direct_trace_distance"],
            "absolute_drive_work": 0.0,
            "autonomous": "yes",
            "external_clock": "no",
            "physical_role": "logical node directly loaded by structured bath",
        },
        {
            "architecture": "integrated_passive_buffer",
            "trace_distance": reference_passive["buffered_trace_distance"],
            "absolute_drive_work": 0.0,
            "autonomous": "yes",
            "external_clock": "no",
            "physical_role": "static reaction coordinate / impedance filter",
        },
        {
            "architecture": "floquet_buffer_best_efficiency",
            "trace_distance": best_efficiency["buffered_trace_distance"],
            "absolute_drive_work": best_efficiency["drive_work_absolute"],
            "autonomous": "no",
            "external_clock": "yes",
            "physical_role": "actively tunable spectral filter",
        },
        {
            "architecture": "floquet_buffer_best_gain",
            "trace_distance": best_gain["buffered_trace_distance"],
            "absolute_drive_work": best_gain["drive_work_absolute"],
            "autonomous": "no",
            "external_clock": "yes",
            "physical_role": "maximum-distinguishability driven filter",
        },
    ]
    return summary_rows, driven_increment_rows, sorted(
        reference_driven, key=lambda row: row["drive_amplitude"]
    )


def _mos_current(overdrive, drain_source, channel_modulation=0.06):
    overdrive = np.maximum(overdrive, 0.0)
    drain_source = np.maximum(drain_source, 0.0)
    triode = drain_source < overdrive
    current = np.where(
        triode,
        overdrive * drain_source - 0.5 * drain_source**2,
        0.5 * overdrive**2,
    )
    return current * (1.0 + channel_modulation * drain_source)


def cmos_transfer_curve(points=501, threshold=0.22):
    vin_values = np.linspace(0.0, 1.0, points)
    vout_grid = np.linspace(0.0, 1.0, 3001)
    vout_values = []
    for vin in vin_values:
        n_current = _mos_current(vin - threshold, vout_grid)
        p_current = _mos_current(1.0 - vin - threshold, 1.0 - vout_grid)
        vout_values.append(vout_grid[np.argmin(np.abs(n_current - p_current))])
    return vin_values, np.asarray(vout_values)


def thermal_transfer_curve(rows, epsilon1=20.0, beta_hot=1.0, beta_cold=2.0):
    selected = sorted(
        (
            row
            for row in rows
            if row["epsilon1"] == epsilon1 and beta_hot <= row["beta1"] <= beta_cold
        ),
        key=lambda row: row["beta1"],
    )
    input_normalized = np.array(
        [(row["beta1"] - beta_hot) / (beta_cold - beta_hot) for row in selected]
    )
    output_normalized = np.array(
        [
            (row["beta_z_infinity"] - beta_hot) / (beta_cold - beta_hot)
            for row in selected
        ]
    )
    return input_normalized, output_normalized


def transfer_metrics(input_values, output_values):
    gain = np.abs(np.gradient(output_values, input_values))
    regenerative = np.flatnonzero(gain >= 1.0)
    if len(regenerative) == 0:
        low_threshold = high_threshold = float(input_values[np.argmax(gain)])
    else:
        low_threshold = float(input_values[regenerative[0]])
        high_threshold = float(input_values[regenerative[-1]])
    output_high = float(np.interp(low_threshold, input_values, output_values))
    output_low = float(np.interp(high_threshold, input_values, output_values))
    return {
        "input_low_threshold": low_threshold,
        "input_high_threshold": high_threshold,
        "output_high_at_low_threshold": output_high,
        "output_low_at_high_threshold": output_low,
        "low_noise_margin": low_threshold - output_low,
        "high_noise_margin": output_high - high_threshold,
        "maximum_small_signal_gain": float(np.max(gain)),
    }, gain


def _box(ax, x, y, width, height, text, color, edge="#222222", fontsize=9):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, start, end, text=None, style="->", dashed=False, color="#222222"):
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=1.25,
        linestyle="--" if dashed else "-",
        color=color,
    )
    ax.add_patch(arrow)
    if text:
        ax.text(
            0.5 * (start[0] + end[0]),
            0.5 * (start[1] + end[1]) + 0.12,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def make_architecture_schematic(output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), constrained_layout=True)
    titles = [
        "Direct autonomous",
        "Integrated passive buffer",
        "Floquet hybrid",
        "Complementary thermal inverter",
    ]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 4.2)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)

    _box(axes[0], 0.25, 1.55, 1.25, 0.75, "thermal\nmachine", "#f2cf5b")
    _box(axes[0], 2.55, 1.55, 1.15, 0.75, "structured\nbath", "#d9d9d9")
    _arrow(axes[0], (1.50, 1.93), (2.55, 1.93), "direct load", "<->")
    axes[0].text(2.0, 0.75, "fully autonomous\nminimum hardware", ha="center", fontsize=9)

    _box(axes[1], 0.05, 1.55, 1.15, 0.75, "thermal\nmachine", "#f2cf5b")
    _box(axes[1], 1.55, 1.55, 0.9, 0.75, "static F", "#9ad0c2")
    _box(axes[1], 2.80, 1.55, 1.10, 0.75, "structured\nbath", "#d9d9d9")
    _arrow(axes[1], (1.20, 1.93), (1.55, 1.93), "g", "<->")
    _arrow(axes[1], (2.45, 1.93), (2.80, 1.93), "k", "<->")
    axes[1].text(2.0, 0.75, "autonomous filter\nfixed after fabrication", ha="center", fontsize=9)

    _box(axes[2], 0.05, 1.55, 1.15, 0.75, "thermal\nmachine", "#f2cf5b")
    _box(axes[2], 1.55, 1.55, 0.9, 0.75, "F(t)", "#9ad0c2")
    _box(axes[2], 2.80, 1.55, 1.10, 0.75, "structured\nbath", "#d9d9d9")
    _box(axes[2], 1.45, 3.05, 1.10, 0.55, "clock + work", "#ef9a9a", fontsize=8)
    _arrow(axes[2], (1.20, 1.93), (1.55, 1.93), "g", "<->")
    _arrow(axes[2], (2.45, 1.93), (2.80, 1.93), "k", "<->")
    _arrow(axes[2], (2.0, 3.05), (2.0, 2.30), "A, Omega", dashed=True, color="#b63c3c")
    axes[2].text(2.0, 0.75, "hybrid active filter\nreconfigurable in operation", ha="center", fontsize=9)

    _box(axes[3], 1.45, 1.60, 1.10, 0.72, "output Cz", "#f2cf5b")
    _box(axes[3], 0.05, 3.05, 1.10, 0.58, "cold rail", "#9ecae1")
    _box(axes[3], 2.85, 3.05, 1.10, 0.58, "hot rail", "#ef9a9a")
    _box(axes[3], 0.10, 1.60, 0.95, 0.72, "cooling\npull-high", "#c6dbef", fontsize=8)
    _box(axes[3], 2.95, 1.60, 0.95, 0.72, "heating\npull-low", "#f7c6c6", fontsize=8)
    _arrow(axes[3], (0.60, 3.05), (0.60, 2.32), None, "->")
    _arrow(axes[3], (1.05, 1.96), (1.45, 1.96), None, "->")
    _arrow(axes[3], (2.55, 1.96), (2.95, 1.96), None, "->")
    _arrow(axes[3], (3.40, 2.32), (3.40, 3.05), None, "->")
    axes[3].text(2.0, 0.75, "closest CMOS analogue:\ncompeting state-restoring rates", ha="center", fontsize=9)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def make_performance_figure(summary_rows, increment_rows, reference_rows, output_path):
    import matplotlib.pyplot as plt

    labels = ["direct", "passive", "Floquet\nefficient", "Floquet\nmax D"]
    colors = ["#7f7f7f", "#4c78a8", "#54a24b", "#e45756"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)

    axes[0, 0].bar(
        labels,
        [row["trace_distance"] for row in summary_rows],
        color=colors,
    )
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_ylabel("final trace distance")
    axes[0, 0].set_title("Output-stage distinguishability")
    for index, row in enumerate(summary_rows):
        axes[0, 0].text(index, row["trace_distance"] + 0.02, f"{row['trace_distance']:.3f}", ha="center", fontsize=8)

    axes[0, 1].bar(
        labels,
        [row["absolute_drive_work"] for row in summary_rows],
        color=colors,
    )
    axes[0, 1].set_ylabel("absolute drive-work throughput")
    axes[0, 1].set_title("External control cost")

    amplitudes = np.array([row["drive_amplitude"] for row in reference_rows])
    distances = np.array([row["buffered_trace_distance"] for row in reference_rows])
    works = np.array([row["drive_work_absolute"] for row in reference_rows])
    axes[1, 0].plot(amplitudes, distances, marker="o", color="#4c78a8", label="D buffer")
    axes[1, 0].axhline(reference_rows[0]["direct_trace_distance"], color="#7f7f7f", linestyle="--", label="direct D")
    axes[1, 0].set_xlabel("drive amplitude A")
    axes[1, 0].set_ylabel("trace distance", color="#4c78a8")
    axes[1, 0].tick_params(axis="y", labelcolor="#4c78a8")
    work_axis = axes[1, 0].twinx()
    work_axis.plot(amplitudes, works, marker="s", color="#e45756", label="Wabs")
    work_axis.set_ylabel("absolute work", color="#e45756")
    work_axis.tick_params(axis="y", labelcolor="#e45756")
    axes[1, 0].set_title("Same bath and coupling: what drive adds")
    handles_a, labels_a = axes[1, 0].get_legend_handles_labels()
    handles_b, labels_b = work_axis.get_legend_handles_labels()
    axes[1, 0].legend(handles_a + handles_b, labels_a + labels_b, fontsize=8, loc="center right")

    amplitudes_for_box = [0.2, 0.35, 0.55]
    distributions = [
        [
            row["floquet_increment_over_passive"]
            for row in increment_rows
            if row["drive_amplitude"] == amplitude
        ]
        for amplitude in amplitudes_for_box
    ]
    axes[1, 1].boxplot(distributions, tick_labels=[str(value) for value in amplitudes_for_box], showfliers=True)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("drive amplitude A")
    axes[1, 1].set_ylabel("D(Floquet) - D(passive)")
    axes[1, 1].set_title("Drive-specific gain over 243 matched points")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _draw_cmos(ax):
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.0, 5.0)
    ax.set_axis_off()
    ax.set_title("CMOS inverter")
    _box(ax, 2.35, 3.25, 1.30, 0.58, "pMOS pull-up", "#f7c6c6")
    _box(ax, 2.35, 1.15, 1.30, 0.58, "nMOS pull-down", "#c6dbef")
    ax.plot([3.0, 3.0], [3.83, 4.45], color="#222222", linewidth=1.5)
    ax.plot([3.0, 3.0], [0.55, 1.15], color="#222222", linewidth=1.5)
    ax.plot([3.0, 3.0], [1.73, 3.25], color="#222222", linewidth=1.5)
    ax.plot([3.0, 4.65], [2.48, 2.48], color="#222222", linewidth=1.5)
    ax.plot([0.65, 2.35], [2.48, 2.48], color="#222222", linewidth=1.5)
    ax.plot([2.35, 2.35], [1.44, 3.54], color="#222222", linewidth=1.1)
    ax.text(3.0, 4.62, "VDD", ha="center", fontsize=9)
    ax.text(3.0, 0.30, "GND", ha="center", fontsize=9)
    ax.text(0.55, 2.48, "Vin", ha="right", va="center", fontsize=9)
    ax.text(4.78, 2.48, "Vout", ha="left", va="center", fontsize=9)
    ax.text(3.0, 4.90, "electrical supply rails", ha="center", fontsize=8)


def _draw_thermal_inverter(ax):
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.0, 5.0)
    ax.set_axis_off()
    ax.set_title("Floquet thermodynamic inverter")
    _box(ax, 2.35, 2.15, 1.30, 0.70, "output Cz", "#f2cf5b")
    _box(ax, 0.15, 3.55, 1.10, 0.58, "cold bath\nbeta high", "#9ecae1", fontsize=8)
    _box(ax, 4.75, 3.55, 1.10, 0.58, "hot bath\nbeta low", "#ef9a9a", fontsize=8)
    _box(ax, 0.35, 1.70, 1.35, 0.70, "cooling\npull-high", "#c6dbef", fontsize=8)
    _box(ax, 4.30, 1.70, 1.35, 0.70, "heating\npull-low", "#f7c6c6", fontsize=8)
    _box(ax, 2.15, 3.65, 1.70, 0.55, "input-controlled rates", "#d8c3e8", fontsize=8)
    _arrow(ax, (0.70, 3.55), (1.00, 2.40), None)
    _arrow(ax, (1.70, 2.05), (2.35, 2.40), None)
    _arrow(ax, (3.65, 2.40), (4.30, 2.05), None)
    _arrow(ax, (5.00, 2.40), (5.30, 3.55), None)
    _arrow(ax, (3.00, 3.65), (3.00, 2.85), "beta in", dashed=True)
    ax.text(2.02, 1.93, r"Gamma down", ha="center", fontsize=8)
    ax.text(3.98, 1.93, r"Gamma up", ha="center", fontsize=8)
    ax.text(3.0, 0.75, "Floquet modulation may tune both channels,\nbut is not itself complementary pull-up/pull-down", ha="center", fontsize=8)


def make_cmos_comparison_figure(cmos_curve, thermal_curve, output_path):
    import matplotlib.pyplot as plt

    cmos_x, cmos_y, cmos_gain = cmos_curve
    thermal_x, thermal_y, thermal_gain = thermal_curve
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    _draw_cmos(axes[0, 0])
    _draw_thermal_inverter(axes[0, 1])

    axes[1, 0].plot(cmos_x, cmos_y, label="CMOS square-law model", color="#4c78a8")
    axes[1, 0].plot(thermal_x, thermal_y, label="thermodynamic NOT, epsilon1=20", color="#e45756")
    axes[1, 0].plot([0.0, 1.0], [1.0, 0.0], color="#777777", linestyle=":", label="ideal inversion guide")
    axes[1, 0].set_xlabel("normalized input")
    axes[1, 0].set_ylabel("normalized output")
    axes[1, 0].set_xlim(0.0, 1.0)
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title("Static transfer characteristics")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(cmos_x, cmos_gain, label="CMOS |dVout/dVin|", color="#4c78a8")
    axes[1, 1].plot(thermal_x, thermal_gain, label="thermal |d betaout/d betain|", color="#e45756")
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="regeneration boundary")
    axes[1, 1].set_xlabel("normalized input")
    axes[1, 1].set_ylabel("small-signal gain magnitude")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(0.05, max(float(cmos_gain.max()), float(thermal_gain.max())) * 1.25)
    axes[1, 1].set_title("Threshold gain and regeneration")
    axes[1, 1].legend(fontsize=8)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _write_report(path, summary_rows, increment_rows, cmos_metrics, thermal_metrics):
    increments = np.array([row["floquet_increment_over_passive"] for row in increment_rows])
    best_increment = max(increment_rows, key=lambda row: row["floquet_increment_over_passive"])
    direct, passive, efficient, best = summary_rows
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Architecture and CMOS-inverter comparison\n")
        handle.write("==========================================\n\n")
        handle.write("Same-point HEOM architecture comparison:\n")
        handle.write(
            f"- direct autonomous contact: D={direct['trace_distance']:.6f}, Wabs=0\n"
        )
        handle.write(
            f"- integrated passive buffer: D={passive['trace_distance']:.6f}, Wabs=0\n"
        )
        handle.write(
            f"- Floquet best-gain buffer: D={best['trace_distance']:.6f}, "
            f"Wabs={best['absolute_drive_work']:.6f}\n"
        )
        handle.write(
            f"- drive-specific increment at best-gain point: "
            f"{best['trace_distance'] - passive['trace_distance']:.6f}\n\n"
        )
        handle.write("Matched Floquet versus passive-buffer sweep:\n")
        handle.write(f"- driven points: {len(increments)}\n")
        handle.write(f"- positive incremental points: {np.sum(increments > 0)}/{len(increments)}\n")
        handle.write(f"- median increment: {np.median(increments):.6f}\n")
        handle.write(
            f"- 5--95% interval: [{np.quantile(increments, 0.05):.6f}, "
            f"{np.quantile(increments, 0.95):.6f}]\n"
        )
        handle.write(f"- maximum increment: {best_increment['floquet_increment_over_passive']:.6f}\n\n")
        handle.write("Normalized inverter transfer metrics:\n")
        for name, metrics in [("CMOS", cmos_metrics), ("thermodynamic NOT", thermal_metrics)]:
            handle.write(
                f"- {name}: gain_max={metrics['maximum_small_signal_gain']:.6f}, "
                f"VIL/xL={metrics['input_low_threshold']:.6f}, "
                f"VIH/xH={metrics['input_high_threshold']:.6f}, "
                f"NML={metrics['low_noise_margin']:.6f}, "
                f"NMH={metrics['high_noise_margin']:.6f}\n"
            )
        handle.write("\nDecision:\n")
        handle.write(
            "The integrated passive buffer is the strongest architecture supported by the current "
            "fixed-parameter output-stage data: it retains autonomy and captures most of the "
            "distinguishability gain. Floquet driving is justified when in-operation spectral "
            "tunability, calibration, or rescue of poorly matched static parameters is worth its "
            "clock, work, and control overhead. A CMOS analogy is useful for transfer gain, noise "
            "margin, and load isolation, but a genuine complementary thermal inverter requires "
            "separate input-controlled heating and cooling channels; the present single buffer is "
            "closer to an active spectral filter than to a complete CMOS inverter.\n"
        )
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_rows = _read_rows(OUTPUT_DIR / "heom_floquet_phase_sweep.csv")
    baseline_rows = _read_rows(BASELINE_CURVES)
    summary_rows, increment_rows, reference_rows = build_architecture_data(phase_rows)

    cmos_x, cmos_y = cmos_transfer_curve()
    thermal_x, thermal_y = thermal_transfer_curve(baseline_rows)
    cmos_metrics, cmos_gain = transfer_metrics(cmos_x, cmos_y)
    thermal_metrics, thermal_gain = transfer_metrics(thermal_x, thermal_y)
    metric_rows = [
        {"architecture": "CMOS", **cmos_metrics},
        {"architecture": "thermodynamic_NOT", **thermal_metrics},
    ]

    _write_csv(
        OUTPUT_DIR / "architecture_quantitative_comparison.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    _write_csv(
        OUTPUT_DIR / "floquet_increment_over_passive.csv",
        increment_rows,
        list(increment_rows[0]),
    )
    _write_csv(
        OUTPUT_DIR / "cmos_thermal_transfer_metrics.csv",
        metric_rows,
        list(metric_rows[0]),
    )
    make_architecture_schematic(OUTPUT_DIR / "thermal_architecture_schematic.png")
    make_performance_figure(
        summary_rows,
        increment_rows,
        reference_rows,
        OUTPUT_DIR / "architecture_performance_tradeoff.png",
    )
    make_cmos_comparison_figure(
        (cmos_x, cmos_y, cmos_gain),
        (thermal_x, thermal_y, thermal_gain),
        OUTPUT_DIR / "cmos_floquet_inverter_comparison.png",
    )
    report = _write_report(
        OUTPUT_DIR / "architecture_cmos_comparison_report.txt",
        summary_rows,
        increment_rows,
        cmos_metrics,
        thermal_metrics,
    )
    print(report.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
