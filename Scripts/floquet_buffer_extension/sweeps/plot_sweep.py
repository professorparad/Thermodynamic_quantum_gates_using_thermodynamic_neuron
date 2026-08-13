from pathlib import Path


def plot_sweep(rows, output_path):
    """Plot Floquet-buffer sweep metrics grouped by swept parameter."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameters = []
    for row in rows:
        if row["swept_parameter"] not in parameters:
            parameters.append(row["swept_parameter"])

    fig, axes = plt.subplots(len(parameters), 2, figsize=(11.0, 3.6 * len(parameters)))
    if len(parameters) == 1:
        axes = [axes]

    for axis_row, parameter in zip(axes, parameters):
        selected = [row for row in rows if row["swept_parameter"] == parameter]
        x = [row["swept_value"] for row in selected]
        gain = [row["trace_distance_gain"] for row in selected]
        work = [row["integrated_drive_work"] for row in selected]

        axis_row[0].plot(x, gain, marker="o")
        axis_row[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis_row[0].set_xlabel(parameter)
        axis_row[0].set_ylabel("buffer trace-distance gain")
        axis_row[0].grid(True, alpha=0.25)

        axis_row[1].plot(x, work, marker="o", color="tab:red")
        axis_row[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis_row[1].set_xlabel(parameter)
        axis_row[1].set_ylabel("drive-work proxy")
        axis_row[1].grid(True, alpha=0.25)

    fig.suptitle("Floquet Buffer Parameter Screening", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
