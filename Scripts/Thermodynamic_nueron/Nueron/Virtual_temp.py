import numpy as np 
def virtual_temp(P01 , P10 , ez):
    ratio =  P10 / (P01 + 1e-300)
    return -np.log(ratio)/ez