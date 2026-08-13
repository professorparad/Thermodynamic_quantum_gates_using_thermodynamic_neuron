from pathlib import Path


def _grid(rows, drive_amplitude, field):
    import numpy as np

    selected = [row for row in rows if row["drive_amplitude"] == drive_amplitude]
    xs = sorted({row["buffer_coupling"] for row in selected})
    ys = sorted({row["coupling_scale"] for row in selected})
    values = np.full((len(ys), len(xs)), float("nan"))
    for row in selected:
        y = ys.index(row["coupling_scale"])
        x = xs.index(row["buffer_coupling"])
        values[y, x] = row[field]
    return xs, ys, values


def plot_phase_maps(rows, output_path):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drive = max({row["drive_amplitude"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    fields = [
        ("buffered_min_margin", "Buffered NOT margin"),
        ("margin_gain", "Buffered - direct margin"),
        ("trace_distance_gain", "Buffered - direct trace distance"),
    ]
    for ax, (field, title) in zip(axes, fields):
        xs, ys, values = _grid(rows, drive, field)
        image = ax.imshow(values, origin="lower", aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(xs)), [f"{x:.2f}" for x in xs])
        ax.set_yticks(range(len(ys)), [f"{y:.1f}" for y in ys])
        ax.set_xlabel("system-buffer coupling $g_{SF}$")
        ax.set_ylabel("bath-coupling scale")
        ax.set_title(f"{title}\nA={drive:.2f}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_phase_classification(rows, output_path):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drive = max({row["drive_amplitude"] for row in rows})
    selected = [row for row in rows if row["drive_amplitude"] == drive]
    xs = sorted({row["buffer_coupling"] for row in selected})
    ys = sorted({row["coupling_scale"] for row in selected})
    labels = {"fail": 0, "fragile_pass": 1, "robust_pass": 2}
    values = np.zeros((len(ys), len(xs)))
    for row in selected:
        values[ys.index(row["coupling_scale"]), xs.index(row["buffer_coupling"])] = labels[
            row["three_phase_label"]
        ]

    cmap = colors.ListedColormap(["#b2182b", "#fdae61", "#1a9850"])
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(len(xs)), [f"{x:.2f}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:.1f}" for y in ys])
    ax.set_xlabel("system-buffer coupling $g_{SF}$")
    ax.set_ylabel("bath-coupling scale")
    ax.set_title(f"Three-phase robustness classification, A={drive:.2f}")
    for label, value in labels.items():
        ax.scatter([], [], color=cmap(value / 2), label=label.replace("_", " "))
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_drive_slices(rows, output_path):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drives = sorted({row["drive_amplitude"] for row in rows})
    xs = sorted({row["buffer_coupling"] for row in rows})
    ys = sorted({row["coupling_scale"] for row in rows})
    labels = {"fail": 0, "fragile_pass": 1, "robust_pass": 2}
    cmap = colors.ListedColormap(["#b2182b", "#fdae61", "#1a9850"])

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)

    def values_for_drive(drive):
        selected = [row for row in rows if row["drive_amplitude"] == drive]
        values = np.zeros((len(ys), len(xs)))
        for row in selected:
            values[ys.index(row["coupling_scale"]), xs.index(row["buffer_coupling"])] = labels[
                row["three_phase_label"]
            ]
        return values

    image = ax.imshow(values_for_drive(drives[0]), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(len(xs)), [f"{x:.2f}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:.1f}" for y in ys])
    ax.set_xlabel("system-buffer coupling $g_{SF}$")
    ax.set_ylabel("bath-coupling scale")

    def update(frame):
        drive = drives[frame]
        image.set_data(values_for_drive(drive))
        ax.set_title(f"Robustness phase slice as drive changes: A={drive:.2f}")
        return (image,)

    animation = FuncAnimation(fig, update, frames=len(drives), interval=900, blit=False)
    try:
        animation.save(output_path, writer=PillowWriter(fps=1))
    except Exception:
        fallback = output_path.with_suffix(".png")
        update(len(drives) - 1)
        fig.savefig(fallback, dpi=180)
        output_path = fallback
    plt.close(fig)
    return output_path


def plot_phase_3d(rows, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = np.array([row["buffer_coupling"] for row in rows], dtype=float)
    ys = np.array([row["coupling_scale"] for row in rows], dtype=float)
    zs = np.array([row["drive_amplitude"] for row in rows], dtype=float)
    margins = np.array([row["buffered_min_margin"] for row in rows], dtype=float)
    labels = [row["three_phase_label"] for row in rows]
    colors = {
        "fail": "#b2182b",
        "fragile_pass": "#fdae61",
        "robust_pass": "#1a9850",
    }

    fig = plt.figure(figsize=(8.0, 6.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for label in ["fail", "fragile_pass", "robust_pass"]:
        mask = np.array([item == label for item in labels])
        if not np.any(mask):
            continue
        sizes = 45.0 + 180.0 * margins[mask] / max(float(np.max(margins)), 1.0e-12)
        ax.scatter(
            xs[mask],
            ys[mask],
            zs[mask],
            s=sizes,
            c=colors[label],
            alpha=0.82,
            edgecolor="black",
            linewidth=0.35,
            label=label.replace("_", " "),
        )

    ax.set_xlabel("system-buffer coupling $g_{SF}$")
    ax.set_ylabel("bath-coupling scale")
    ax.set_zlabel("drive amplitude $A$")
    ax.set_title("Three-dimensional robustness phase diagram")
    ax.view_init(elev=24, azim=-52)
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
