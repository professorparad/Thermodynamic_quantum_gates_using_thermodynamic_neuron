import numpy as np 
def Dissipator(L , rho):
    return (L* rho * L.dag() - 0.5*(L.dag()* L * rho + rho* L.dag()*L))

def tot_dissipator(bath_ops , rho):
    D = 0 * rho 
    for L in bath_ops:
        D+= Dissipator(L , rho)
    return D
def heat_current(H , rho , bath_ops):
    D = tot_dissipator(bath_ops , rho )
    return float(np.real((H*D).tr()))