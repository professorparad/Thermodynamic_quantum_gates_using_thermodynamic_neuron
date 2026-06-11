import numpy as np 
from ..config.parametres import *
def gfun(beta):

    return 1.0/(1.0+np.exp(beta * ez))
def collector_current_virtual(beta_z , beta_v):
    return mu * ez * (gfun(beta_z) - gfun(beta_v))
def collector_current_exact(Jz):
    return Jz
def modulator_current(beta_z):
    return (mu_p * ez * (gfun(beta_z)- gfun(beta_r)))
