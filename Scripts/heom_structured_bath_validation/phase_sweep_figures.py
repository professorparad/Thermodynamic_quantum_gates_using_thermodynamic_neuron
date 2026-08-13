import csv
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def _grid(rows, x_key, y_key, value_fn):
    xs = sorted(set(row[x_key] for row in rows))
    ys = sorted(set(row[y_key] for row in rows))
    values = np.full((len(ys), len(xs)), np.nan)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            selected = [row for row in rows if row[x_key] == x and row[y_key] == y]
            values[yi, xi] = value_fn(selected)
    return xs, ys, values


def make_operating_region_figures(rows, output_path):
    import matplotlib.pyplot as plt

    def best_finite_efficiency(selected):
        values = [
            min(row["gain_per_absolute_work"], 10.0)
            for row in selected
            if np.isfinite(row["gain_per_absolute_work"])
        ]
        return max(values) if values else 0.0

    xs, ys, best_gain = _grid(
        rows,
        "drive_amplitude",
        "system_buffer_coupling",
        lambda selected: max(row["trace_distance_gain"] for row in selected),
    )
    _, _, robust_fraction = _grid(
        rows,
        "drive_amplitude",
        "system_buffer_coupling",
        lambda selected: sum(
            row["buffered_trace_distance"] >= 0.75 and row["trace_distance_gain"] > 0
            for row in selected
        )
        / len(selected),
    )
    _, _, best_efficiency = _grid(
        rows,
        "drive_amplitude",
        "system_buffer_coupling",
        best_finite_efficiency,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    panels = [
        (best_gain, "Best HEOM gain", "buffered - direct D"),
        (robust_fraction, "Robust-pass fraction", "fraction"),
        (best_efficiency, "Best gain/work throughput (clipped)", "gain / Wabs"),
    ]
    for ax, (values, title, cbar_label) in zip(axes, panels):
        im = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(xs)), [str(x) for x in xs])
        ax.set_yticks(range(len(ys)), [str(y) for y in ys])
        ax.set_xlabel("drive amplitude A")
        ax.set_ylabel("system-buffer coupling gSF")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=cbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_energy_frontier(rows, output_path):
    import matplotlib.pyplot as plt

    works = np.array([row["drive_work_absolute"] for row in rows], dtype=float)
    gains = np.array([row["trace_distance_gain"] for row in rows], dtype=float)
    buffered = np.array([row["buffered_trace_distance"] for row in rows], dtype=float)
    labels = np.array([row["label"] for row in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = {
        "buffer_hurts": "#8c8c8c",
        "costly_gain": "#f58518",
        "efficient_gain": "#54a24b",
        "passive_buffer_gain": "#4c78a8",
        "zero_resolved_work": "#b279a2",
    }
    for label in sorted(set(labels)):
        mask = labels == label
        ax.scatter(
            works[mask],
            gains[mask],
            s=22 + 45 * buffered[mask],
            alpha=0.75,
            color=colors.get(label, "#333333"),
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("absolute drive-work throughput")
    ax.set_ylabel("HEOM distinguishability gain")
    ax.set_title("Energy-fidelity frontier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    rows = _read_rows(OUTPUT_DIR / "heom_floquet_phase_sweep.csv")
    region = make_operating_region_figures(rows, OUTPUT_DIR / "heom_floquet_operating_regions.png")
    frontier = make_energy_frontier(rows, OUTPUT_DIR / "heom_energy_fidelity_frontier.png")
    print("Saved operating-region figure:", region)
    print("Saved energy-fidelity frontier:", frontier)


if __name__ == "__main__":
    main()
