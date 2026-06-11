import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

from ..collector.baths import global_thermal_bath
from ..collector.hamiltonian import H
from ..collector.heat_current import heat_current
from ..collector.operators import smz, spz
from ..config.parametres import betaz, ez, gammaz
from ..plots.plots import PLOTS_DIR


def low_rank_density_approximation(rho, bond_dimension=4):
    rho_array = rho.full()
    U, singular_values, Vh = np.linalg.svd(rho_array, full_matrices=False)
    rank = min(bond_dimension, len(singular_values))
    approx_array = (U[:, :rank] * singular_values[:rank]) @ Vh[:rank, :]
    return Qobj(approx_array, dims=rho.dims), singular_values, rank


def tensor_train_scaling(n_values=None, bond_dimensions=(2, 4)):
    if n_values is None:
        n_values = np.arange(3, 11)

    rows = []
    for n_qubits in n_values:
        dim = 2 ** n_qubits
        gksl_memory_mb = dim ** 2 * 16 / (1024 ** 2)
        row = {
            "N": n_qubits,
            "dim": dim,
            "gksl_memory_mb": gksl_memory_mb,
        }
        for bond_dimension in bond_dimensions:
            tt_memory_mb = n_qubits * bond_dimension ** 2 * 4 * 16 / (1024 ** 2)
            row[f"tt_D{bond_dimension}_memory_mb"] = tt_memory_mb
            row[f"tt_D{bond_dimension}_speedup"] = gksl_memory_mb / (tt_memory_mb + 1e-30)
        rows.append(row)
    return rows


def tensor_network_report(results, rho_ss, bond_dimension=8):
    rho_approx, singular_values, rank = low_rank_density_approximation(rho_ss, bond_dimension)
    bathz = global_thermal_bath(H, smz + spz, betaz, gammaz)
    Jz_approx = heat_current(H, rho_approx, bathz)
    reference_Jz = results[0]["Jz"]
    error_pct = abs(Jz_approx - reference_Jz) / (abs(reference_Jz) + 1e-30) * 100

    return {
        "N": 3,
        "dim": 8,
        "bond_dimension": rank,
        "Jz_approx": Jz_approx,
        "Jz_reference": reference_Jz,
        "Jz_error_pct": error_pct,
        "singular_values": singular_values,
        "scaling": tensor_train_scaling(),
    }


def plot_tensor_network_scaling(report, filename="tensor_network_scaling.png", show=False):
    rows = report["scaling"]
    n_values = np.array([r["N"] for r in rows])
    gksl_memory = np.array([r["gksl_memory_mb"] for r in rows])
    tt_d2 = np.array([r["tt_D2_memory_mb"] for r in rows])
    tt_d4 = np.array([r["tt_D4_memory_mb"] for r in rows])
    speedup_d4 = np.array([r["tt_D4_speedup"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(n_values, gksl_memory, "o-", linewidth=3, label="GKSL full", color="red")
    ax1.semilogy(n_values, tt_d2, "s--", linewidth=2, label="TT D=2", color="blue")
    ax1.semilogy(n_values, tt_d4, "^--", linewidth=2, label="TT D=4", color="green")
    ax1.set_xlabel("Number of qubits")
    ax1.set_ylabel("Memory (MB)")
    ax1.set_title("Memory scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    ax2.semilogy(n_values, speedup_d4, "o-", linewidth=3, color="darkgreen", label="GKSL / TT D=4")
    ax2.set_xlabel("Number of qubits")
    ax2.set_ylabel("Speedup factor")
    ax2.set_title("Tensor-train speedup")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig
