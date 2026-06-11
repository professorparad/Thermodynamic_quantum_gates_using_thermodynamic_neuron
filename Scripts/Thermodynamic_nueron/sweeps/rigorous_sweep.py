import numpy as np
from scipy.integrate import solve_ivp

from ..Nueron.currents import collector_current_exact
from ..Nueron.fixed_point import find_fixed_point
from ..Nueron.nueron_ode import nueron_ode_exact
from ..collector.collector_solver import run_collector_virtual
from ..config.parametres import sigma_tol


def run_rigorous_sweep(beta1_values, collector_function=run_collector_virtual):
    results = []

    for beta1 in beta1_values:
        collector = collector_function(beta1)
        Jc_exact = collector_current_exact(collector["Jz"])
        beta_out_fixed = find_fixed_point(Jc_exact)

        sol = solve_ivp(
            nueron_ode_exact,
            [0, 2000],
            [1.0],
            args=(Jc_exact,),
            rtol=1e-9,
            atol=1e-9,
            method="Radau",
        )

        if sol.y.shape[1] > 1:
            final_error = abs(sol.y[0, -1] - sol.y[0, -2])
        else:
            final_error = 0.0
        result = {
            "beta1": beta1,
            "beta_v": collector["beta_v"],
            "J0": collector["J0"],
            "J1": collector["J1"],
            "Jz": collector["Jz"],
            "Jc_exact": Jc_exact,
            "sigma": collector["sigma"],
            "sum_J": collector["J0"] + collector["J1"] + collector["Jz"],
            "beta_out": sol.y[0, -1],
            "beta_out_fixed": beta_out_fixed,
            "convergence_steps": len(sol.t),
            "final_error": final_error,
            "P01": collector.get("P01"),
            "P10": collector.get("P10"),
            "rho_ss": collector.get("rho_ss"),
        }
        results.append(result)

    add_logic_metrics(results)
    return results


def add_logic_metrics(results):
    beta_out = np.array([r["beta_out"] for r in results], dtype=float)
    beta_min = np.nanmin(beta_out)
    beta_max = np.nanmax(beta_out)
    beta_threshold = 0.5 * (beta_min + beta_max)
    beta_range = beta_max - beta_min
    if beta_range == 0:
        beta_range = 1.0

    for r in results:
        margin = abs(r["beta_out"] - beta_threshold)
        r["threshold"] = beta_threshold
        r["margin"] = margin
        r["confidence_score"] = margin / beta_range
        r["logic_output"] = int(r["beta_out"] > beta_threshold)
        r["energy_conserved"] = abs(r["sum_J"]) < 1e-12
        r["second_law_valid"] = r["sigma"] >= -sigma_tol

    return results
