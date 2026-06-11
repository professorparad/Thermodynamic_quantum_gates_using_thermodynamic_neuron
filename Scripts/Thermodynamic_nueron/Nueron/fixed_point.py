from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from .currents import * 

def find_fixed_point(Jc):
    def equation(beta_z):
        return (Jc + modulator_current(beta_z))
    try :
        return brentq(equation, 0.01, 20.0, xtol=1e-10)
    except ValueError :
        sol = solve_ivp(
            lambda t, y: [-(y[0] ** 2 / C_out) * equation(y[0])],
            [0, 5000],
            [1.0],
            rtol=1e-10,
            atol=1e-10,
            method="Radau",
        )
        return sol.y[0, -1]
    
