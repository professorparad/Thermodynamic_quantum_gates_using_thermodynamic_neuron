import numpy as np 
def beta_sweep(beta1_values , collector_function):
    results = []
    for beta1 in beta1_values:
        r = collector_function(beta1)
        if isinstance(r, dict):
            r["beta1"] = beta1
        results.append(r)
    return results
