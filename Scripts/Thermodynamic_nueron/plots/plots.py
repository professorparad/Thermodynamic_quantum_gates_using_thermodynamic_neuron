import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path

from ..config.parametres import sigma_tol

PLOTS_DIR = Path(__file__).resolve().parent

def _plot_path(filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR / Path(filename).name

def plot_transfer_curve(results, filename="transfer_curve.png", show=False):
    beta1 = np.array([r["beta1"] for r in results])
    output_key = "beta_out" if "beta_out" in results[0] else "beta_v"
    beta_out = np.array([r[output_key] for r in results])
    plt.figure(figsize=(8 , 5))
    plt.plot(beta1  , beta_out , linewidth = 3)
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_{out}$' if output_key == "beta_out" else r'$\beta_v$')
    plt.grid(True)
    fig = plt.gcf()
    output_path = _plot_path(filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig

def plot_gksl_analysis(results, filename="gksl_analysis.png", show=False):
    beta1 = np.array([r["beta1"] for r in results])
    beta_v = np.array([r["beta_v"] for r in results])
    beta_out = np.array([r["beta_out"] for r in results])
    J0 = np.array([r["J0"] for r in results])
    J1 = np.array([r["J1"] for r in results])
    Jz = np.array([r["Jz"] for r in results])
    sigma = np.array([r["sigma"] for r in results])
    logic = np.array([r["logic_output"] for r in results])
    threshold = results[0]["threshold"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Thermodynamic Neuron - Rigorous GKSL Analysis", fontsize=14)

    ax = axes[0, 0]
    ax.plot(beta1, beta_v, linewidth=2.5, color="steelblue")
    ax.set_xlabel(r"Input $\beta_1$")
    ax.set_ylabel(r"Virtual temperature $\beta_v$")
    ax.set_title("Collector transfer curve")
    ax.grid(True, alpha=0.4)

    ax = axes[0, 1]
    ax.plot(beta1, beta_out, linewidth=2.5, color="darkorange", label="ODE exact current")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold {threshold:.4f}")
    ax.set_xlabel(r"Input $\beta_1$")
    ax.set_ylabel(r"Output $\beta_z^\infty$")
    ax.set_title("Neuron transfer curve")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    ax = axes[0, 2]
    ax.step(beta1, logic, where="mid", linewidth=2.5, color="darkgreen")
    ax.set_xlabel(r"Input $\beta_1$")
    ax.set_ylabel("Logical output")
    ax.set_yticks([0, 1])
    ax.set_title("Thermodynamic NOT gate")
    ax.grid(True, alpha=0.4)

    ax = axes[1, 0]
    ax.plot(beta1, J0, label=r"$J_0$", linewidth=2)
    ax.plot(beta1, J1, label=r"$J_1$", linewidth=2)
    ax.plot(beta1, Jz, label=r"$J_z$", linewidth=2)
    ax.set_xlabel(r"Input $\beta_1$")
    ax.set_ylabel("Heat current")
    ax.set_title("Collector heat currents")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    ax = axes[1, 1]
    ax.plot(beta1, sigma, linewidth=2.5, color="purple")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Input $\beta_1$")
    ax.set_ylabel(r"$\Sigma$")
    if np.any(sigma < -sigma_tol):
        ax.set_title("Second law violation detected", color="red")
    else:
        ax.set_title("Second law verification")
    ax.grid(True, alpha=0.4)

    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        "GKSL steady-state analysis\n\n"
        f"points: {len(results)}\n"
        f"threshold: {threshold:.6f}\n"
        f"beta_out range: [{beta_out.min():.6f}, {beta_out.max():.6f}]\n"
        f"sigma min: {sigma.min():.3e}\n"
        f"violations: {np.sum(sigma < -sigma_tol)}\n"
        f"max |sum J|: {np.max(np.abs(J0 + J1 + Jz)):.3e}"
    )
    ax.text(0.08, 0.5, summary, fontsize=10, family="monospace", va="center")

    fig.tight_layout()
    output_path = _plot_path(filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig
