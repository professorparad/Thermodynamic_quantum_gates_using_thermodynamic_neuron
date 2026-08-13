from pathlib import Path


def plot_phase_v(rows, output_path):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    threshold_styles = {
        "pt_mpo_direct_output": ("black", "direct threshold"),
        "pt_mpo_floquet_buffered_output": ("tab:gray", "buffered threshold"),
    }
    for architecture in ["pt_mpo_direct_output", "pt_mpo_floquet_buffered_output"]:
        for input_bit in [0, 1]:
            selected = [
                row for row in rows
                if row["architecture"] == architecture and row["input_bit"] == input_bit
            ]
            ax.plot(
                [row["time"] for row in selected],
                [row["output_beta_effective"] for row in selected],
                label=f"{architecture}, input={input_bit}",
            )
        arch_rows = [row for row in rows if row["architecture"] == architecture]
        if arch_rows and "decoder_threshold" in arch_rows[-1]:
            color, label = threshold_styles[architecture]
            ax.axhline(
                arch_rows[-1]["decoder_threshold"],
                color=color,
                linestyle="--",
                linewidth=0.8,
                label=label,
            )
    ax.set_xlabel("time")
    ax.set_ylabel("effective output beta")
    ax.set_title("Phase V PT-MPO/TEMPO Structured-Bath Truth Table")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
