from collections import defaultdict
from pathlib import Path

import numpy as np


def plot_fig2_virtual_temperature(rows, output_path, beta0, engine_threshold):
    """Plot virtual temperature regimes for the three-qubit machine."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    beta1 = np.array([row["beta1"] for row in rows], dtype=float)
    beta_v = np.array([row["beta_v"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(beta1, beta_v, color="#1f77b4", linewidth=2.0)
    ax.axvline(beta0, color="black", linestyle="--", linewidth=1.0, label=r"$\beta_1=\beta_0$")
    ax.axvline(
        engine_threshold,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=r"$\beta_1=(\epsilon_0/\epsilon_1)\beta_0$",
    )
    ax.axhline(beta0, color="#2ca02c", linestyle="--", linewidth=1.0, label=r"$\beta_v=\beta_0$")
    ax.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.0, label=r"$\beta_v=0$")
    ax.fill_between(beta1, beta_v, beta0, where=beta_v > beta0, color="#1f77b4", alpha=0.12)
    ax.set_xlabel(r"Input inverse temperature $\beta_1$")
    ax.set_ylabel(r"Virtual inverse temperature $\beta_v$")
    ax.set_title("Fig. 2 Reconstruction: Three-Qubit Machine Regimes")
    ax.set_ylim(min(-2.0, beta_v.min()), max(beta_v.max(), beta0) + 0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_fig3b_not_transfer(curves, output_path, beta_hot, beta_cold):
    """Named Fig. 3B transfer-characteristic plot."""

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
            linewidth=1.8,
            label=fr"$\epsilon_1={epsilon1:g}$",
        )
    ax.step(
        [0.5, beta_hot, beta_cold, 2.5],
        [beta_cold, beta_cold, beta_hot, beta_hot],
        where="post",
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="ideal NOT",
    )
    ax.set_xlabel(r"Input inverse temperature $\beta_1$")
    ax.set_ylabel(r"Output inverse temperature $\beta_z^\infty$")
    ax.set_title("Fig. 3B Reconstruction: NOT Transfer Curve")
    ax.set_ylim(beta_hot - 0.15, beta_cold + 0.15)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_fig6_nor_surface(rows, output_path, beta_hot, beta_cold):
    """Plot NOR response surface and truth-table corner points."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    beta1_values = sorted({row["beta1"] for row in rows})
    beta2_values = sorted({row["beta2"] for row in rows})
    z = np.array([row["beta_z_infinity"] for row in rows], dtype=float)
    z = z.reshape(len(beta2_values), len(beta1_values))

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(
        z,
        origin="lower",
        extent=[min(beta1_values), max(beta1_values), min(beta2_values), max(beta2_values)],
        aspect="auto",
        cmap="viridis",
        vmin=beta_hot,
        vmax=beta_cold,
    )
    corners = [
        (beta_hot, beta_hot, "1"),
        (beta_hot, beta_cold, "0"),
        (beta_cold, beta_hot, "0"),
        (beta_cold, beta_cold, "0"),
    ]
    for x, y, label in corners:
        ax.scatter([x], [y], color="white", edgecolor="black", s=80, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold", zorder=4)
    ax.set_xlabel(r"Input $\beta_1$")
    ax.set_ylabel(r"Input $\beta_2$")
    ax.set_title("Fig. 6 Reconstruction: NOR Response")
    fig.colorbar(image, ax=ax, label=r"$\beta_z^\infty$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_fig3c_tradeoff(rows, output_path):
    """Plot error versus dissipation proxy for the NOT gate."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row["dissipation_proxy"])

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(
        [row["dissipation_proxy"] for row in rows],
        [row["average_error"] for row in rows],
        marker="o",
        linewidth=1.8,
    )
    for row in rows:
        ax.annotate(
            fr"$\epsilon_1={row['epsilon1']:g}$",
            (row["dissipation_proxy"], row["average_error"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlabel(r"Average dissipation proxy $\langle\Sigma\rangle$")
    ax.set_ylabel(r"Average decoding error $\langle\xi\rangle$")
    ax.set_title("Fig. 3C Reconstruction: Error-Dissipation Trade-Off")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_fig7_majority_slices(rows, output_path, beta_hot, beta_cold):
    """Plot 3-majority response as beta3 slices."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    beta1_values = sorted({row["beta1"] for row in rows})
    beta2_values = sorted({row["beta2"] for row in rows})
    beta3_values = sorted({row["beta3"] for row in rows})
    selected_beta3 = [beta3_values[0], beta3_values[len(beta3_values) // 2], beta3_values[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharex=True, sharey=True)
    image = None
    for ax, beta3 in zip(axes, selected_beta3):
        slice_rows = [row for row in rows if abs(row["beta3"] - beta3) < 1e-12]
        z = np.array([row["beta_z_infinity"] for row in slice_rows], dtype=float)
        z = z.reshape(len(beta2_values), len(beta1_values))
        image = ax.imshow(
            z,
            origin="lower",
            extent=[min(beta1_values), max(beta1_values), min(beta2_values), max(beta2_values)],
            aspect="auto",
            cmap="viridis",
            vmin=beta_hot,
            vmax=beta_cold,
        )
        ax.set_title(fr"$\beta_3={beta3:.2f}$")
        ax.set_xlabel(r"$\beta_1$")
        ax.grid(False)
    axes[0].set_ylabel(r"$\beta_2$")
    fig.suptitle("Fig. 7 Reconstruction: 3-Majority Response Slices")
    fig.colorbar(image, ax=axes.ravel().tolist(), label=r"$\beta_z^\infty$")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fig8_xor_surface(rows, output_path, beta_hot, beta_cold):
    """Plot XOR network response surface."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    beta1_values = sorted({row["beta1"] for row in rows})
    beta2_values = sorted({row["beta2"] for row in rows})
    z = np.array([row["xor_beta_z"] for row in rows], dtype=float)
    z = z.reshape(len(beta2_values), len(beta1_values))

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(
        z,
        origin="lower",
        extent=[min(beta1_values), max(beta1_values), min(beta2_values), max(beta2_values)],
        aspect="auto",
        cmap="viridis",
        vmin=beta_hot,
        vmax=beta_cold,
    )
    corners = [
        (beta_hot, beta_hot, "0"),
        (beta_hot, beta_cold, "1"),
        (beta_cold, beta_hot, "1"),
        (beta_cold, beta_cold, "0"),
    ]
    for x, y, label in corners:
        ax.scatter([x], [y], color="white", edgecolor="black", s=80, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold", zorder=4)
    ax.set_xlabel(r"Input $\beta_1$")
    ax.set_ylabel(r"Input $\beta_2$")
    ax.set_title("Fig. 8 Reconstruction: XOR Network Response")
    fig.colorbar(image, ax=ax, label=r"$\beta_z^\infty$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
