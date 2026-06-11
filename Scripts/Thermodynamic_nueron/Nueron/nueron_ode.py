from .currents import * 
from ..config.parametres import *

def nueron_ode_virtual( t , y  , beta_v):
    beta_z = y[0]
    Jc = collector_current_virtual(beta_z , beta_v)
    Jm = modulator_current(beta_z)
    dbeta = (-(beta_z ** 2)/C_out) * (Jc + Jm)
    return [dbeta]
def nueron_ode_exact( t , y  , Jc_exact):
    beta_z = y[0]
    Jm = modulator_current(beta_z)
    dbeta = (-(beta_z ** 2)/C_out) * (Jc_exact + Jm)
    return [dbeta]
