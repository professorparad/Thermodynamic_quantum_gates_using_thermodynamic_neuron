import numpy as np
from qutip import steadystate

from ..baths import global_thermal_bath
from ..hamiltonian import H
from ..heat_current import heat_current
from ..operators import sm0, sm1, smz, sp0, sp1, spz
from ...Nueron.Virtual_temp import virtual_temp
from ...config.parametres import (
    beta0,
    betaz,
    e0,
    e1,
    ez,
    gamma0,
    gamma1,
    gammaz,
)


def run_collector_global(beta1):
    bath0 = global_thermal_bath(H, sm0 + sp0, beta0, gamma0)
    bath1 = global_thermal_bath(H, sm1 + sp1, beta1, gamma1)
    bathz = global_thermal_bath(H, smz + spz, betaz, gammaz)
    c_ops = bath0 + bath1 + bathz

    rho_ss = steadystate(H, c_ops)
    J0 = heat_current(H, rho_ss, bath0)
    J1 = heat_current(H, rho_ss, bath1)
    Jz = heat_current(H, rho_ss, bathz)
    sigma = -(beta0 * J0 + beta1 * J1 + betaz * Jz)

    rho01 = rho_ss.ptrace([0, 1])
    P01 = float(np.real(rho01[1, 1]))
    P10 = float(np.real(rho01[2, 2]))
    beta_v = virtual_temp(P01, P10, ez)

    return {
        "beta_v": beta_v,
        "J0": J0,
        "J1": J1,
        "Jz": Jz,
        "P01": P01,
        "P10": P10,
        "sigma": sigma,
        "rho_ss": rho_ss,
    }
