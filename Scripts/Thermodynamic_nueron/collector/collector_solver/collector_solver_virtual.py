from qutip import steadystate 
import numpy as np 
from ..hamiltonian import H 
from ..operators import sm0 , sm1 , smz
from ..baths import thermal_bath 
from ..heat_current import heat_current
from ...config.parametres import * 
from ...Nueron.Virtual_temp import virtual_temp



def run_collector_virtual(beta1):
    bath0 = thermal_bath(sm0 , e0 , beta0 , gamma0)
    bath1 = thermal_bath(sm1 , e1 , beta1 , gamma1)
    bathz = thermal_bath(smz , ez , betaz , gammaz)
    c_ops = bath0 + bath1 + bathz 
    rho_ss = steadystate(H , c_ops )
    J0 = heat_current(H , rho_ss , bath0 )
    J1 = heat_current(H , rho_ss  , bath1)
    Jz = heat_current(H , rho_ss , bathz)
    sigma = -(beta0*J0 + beta1*J1 + betaz*Jz)
    rho01 = rho_ss.ptrace([0  , 1])
    P01  = float(np.real(rho01[1  , 1]))
    P10 = float(np.real(rho01[2 , 2 ]))
    beta_v = virtual_temp(P01 , P10 , ez)
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
