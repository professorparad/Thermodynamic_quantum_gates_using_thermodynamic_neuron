from qutip import steadystate 
import numpy as np 
from ..hamiltonian import H 
from ..operators import sm0 , sm1 , smz 
from ..baths import thermal_bath 
from ..heat_current import heat_current
from ...config.parametres import * 

def run_collector_exact(beta1):
    bath0 = thermal_bath(sm0 , e0 , beta0 , gamma0)
    bath1 = thermal_bath(sm1 , e1 , beta1 , gamma1)
    bathz = thermal_bath(smz , ez , betaz , gammaz)
    c_ops = bath0 + bath1 + bathz 
    rho_ss = steadystate(H , c_ops )
    J0 = heat_current(H , rho_ss , bath0 )
    J1 = heat_current(H , rho_ss  , bath1)
    Jz = heat_current(H , rho_ss , bathz)
    sigma = -(beta0*J0 + beta1*J1 + betaz*Jz)
    return {"Jz_exact": Jz ,  "J0": J0 , "J1":J1 , "Jz":Jz , "sigma" : sigma  , "rho_ss": rho_ss }
