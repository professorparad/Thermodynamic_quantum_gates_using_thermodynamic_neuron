from pathlib import Path


def plot_truth_dynamics(rows, output_path):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for architecture in ["direct_three_qubit", "floquet_buffered_three_qubit"]:
        for input_bit in [0, 1]:
            selected = [
                row
                for row in rows
                if row["architecture"] == architecture and row["input_bit"] == input_bit
            ]
            label = f"{architecture}, input={input_bit}"
            ax1.plot(
                [row["time"] for row in selected],
                [row["output_beta_effective"] for row in selected],
                label=label,
            )
            ax2.plot(
                [row["time"] for row in selected],
                [row["output_excited_population"] for row in selected],
                label=label,
            )
    ax1.axhline(1.5, color="black", linestyle="--", linewidth=0.8, label="decode threshold")
    ax1.set_xlabel("time")
    ax1.set_ylabel("effective output beta")
    ax1.set_title("Thermodynamic NOT Output Beta")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("time")
    ax2.set_ylabel("output excited population")
    ax2.set_title("Output Qubit Population")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
