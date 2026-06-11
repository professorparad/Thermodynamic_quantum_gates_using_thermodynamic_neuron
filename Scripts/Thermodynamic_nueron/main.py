import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Thermodynamic_nueron.sweeps.rigorous_sweep import run_rigorous_sweep
from Thermodynamic_nueron.collector.collector_solver import run_collector_global
import numpy as np 

from Thermodynamic_nueron.config.parametres import (
    e0, e1, ez, g, gamma0, gamma1, gammaz, 
    beta0, betaz, mu, mu_p, beta_r, C_out, sigma_tol
)
from Thermodynamic_nueron.collector.operators import (
    I, sm, sp, n, n0, n1, nz, sm0, sm1, smz
)
from Thermodynamic_nueron.collector.hamiltonian import (
    H0, H_int, H, ket101, ket010
)
from Thermodynamic_nueron.Data_generator.dataset_generator import (
    save_dataset,
    save_structured_datasets,
)
from Thermodynamic_nueron.plots.plots import plot_gksl_analysis, plot_transfer_curve
from Thermodynamic_nueron.tensor_network import (
    plot_tensor_network_scaling,
    tensor_network_report,
)

def main():
    beta1_values = np.linspace(0.5, 4.0, 50)
    results = run_rigorous_sweep(beta1_values, run_collector_global)
    dataset = save_dataset(results, "beta_sweep.csv")
    saved_datasets = save_structured_datasets(results)
    plot_transfer_curve(results, "beta_sweep_transfer_curve.png")
    plot_gksl_analysis(results, "gksl_analysis.png")
    tn_report = tensor_network_report(results, results[0]["rho_ss"])
    plot_tensor_network_scaling(tn_report, "tensor_network_scaling.png")
    second_law_violations = [r for r in results if not r["second_law_valid"]]
    print(f"Generated {len(results)} points")
    print(f"Saved data to {dataset.attrs['saved_to']}")
    print(f"Saved structured datasets: {len(saved_datasets)} CSV files")
    print("Saved plot to Scripts/Thermodynamic_nueron/plots/beta_sweep_transfer_curve.png")
    print("Saved plot to Scripts/Thermodynamic_nueron/plots/gksl_analysis.png")
    print("Saved plot to Scripts/Thermodynamic_nueron/plots/tensor_network_scaling.png")
    print(f"Tensor-network Jz error estimate: {tn_report['Jz_error_pct']:.4f}%")
    if second_law_violations:
        min_sigma = min(r["sigma"] for r in results)
        print(
            "Second law check FAILED: "
            f"{len(second_law_violations)} / {len(results)} points have sigma < -sigma_tol "
            f"(min sigma={min_sigma:.3e})."
        )
    else:
        print("Second law check passed for all sweep points.")
    print("Thermodynamic Neuron package loaded successfully.")
    print(f"  e0={e0}, e1={e1}, ez={ez}, g={g}")
    print(f"  H0 =\n{H0}")
    print(f"  H_int =\n{H_int}")
    print("Operators available: n0, n1, nz, sm0, sm1, smz")


if __name__ == "__main__":
    main()
