from collections import defaultdict
from pathlib import Path


def plot_transfer_curves(curves, output_path, beta_hot, beta_cold):
    """Plot beta_z infinity versus beta1 for the NOT-gate transfer curve."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in curves:
        grouped[row["epsilon1"]].append(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for epsilon1, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: item["beta1"])
        ax.plot(
            [row["beta1"] for row in rows],
            [row["beta_z_infinity"] for row in rows],
            label=fr"$\epsilon_1={epsilon1:g}$",
        )

    ax.plot(
        [beta_hot, beta_hot, beta_cold, beta_cold],
        [beta_cold, beta_hot, beta_hot, beta_hot],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="ideal NOT guide",
    )
    ax.set_xlabel(r"Input inverse temperature $\beta_1$")
    ax.set_ylabel(r"Output inverse temperature $\beta_z^\infty$")
    ax.set_title("Baseline Thermodynamic NOT Gate")
    ax.set_ylim(beta_hot - 0.15, beta_cold + 0.15)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

