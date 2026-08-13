from pathlib import Path


def plot_comparison(rows, summary_rows, output_path):
    """Plot direct versus Floquet-buffer logical dynamics."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for architecture in ["direct", "floquet_buffer"]:
        for initial in ["ground", "excited"]:
            selected = [
                row
                for row in rows
                if row["architecture"] == architecture and row["initial_state"] == initial
            ]
            label = f"{architecture}, {initial}"
            ax1.plot(
                [row["time"] for row in selected],
                [row["system_excited_population"] for row in selected],
                label=label,
            )
            ax2.plot(
                [row["time"] for row in selected],
                [row["system_purity"] for row in selected],
                label=label,
            )

    ax1.set_xlabel("time")
    ax1.set_ylabel("system excited population")
    ax1.set_title("Logical Qubit Population")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("time")
    ax2.set_ylabel("system purity")
    ax2.set_title("Logical Qubit Purity")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)

    title_parts = [
        f"{row['architecture']}: D={row['final_trace_distance']:.3f}, W={row['integrated_drive_work']:.3e}"
        for row in summary_rows
    ]
    fig.suptitle("Floquet Buffer Bridge Model\n" + " | ".join(title_parts), fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

