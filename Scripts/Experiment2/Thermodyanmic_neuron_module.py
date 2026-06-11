from qutip import basis , destroy , qeye , steadystate , tensor 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import  solve_ivp 
from scipy.optimize import brentq


e0 = 2.0 
e1 = 1.5 
ez = e0 - e1 
g = 0.005 
gamma0 = 0.001
gamma1 = 0.001 
gammaz = 0.001 
beta0 = 1.5
betaz = 1.0 
## output reservoir 
mu = 1.0
mu_p = 0.3 
beta_r = 2.0 
C_out = 10.0 
sigma_tol = 1e-10
## operators in qutip 
I = qeye(2)
sm = destroy(2)
sp = sm.dag()
n =sp*sm 
## tensor operators 
n0 = tensor( n , I , I )
n1 = tensor(I , n , I)
nz = tensor (I , I  , n)
sm0 = tensor(sm ,  I , I )
sm1 = tensor(I , sm , I )
smz = tensor(I , I , sm )
## Hamiltonian 
H0 = e0*n0 + e1*n1 + ez*nz
ket101 = tensor(basis( 2, 1 ) , basis(2 , 0 ) , basis(2 , 1 ) )
ket010 = tensor(basis(2 , 0) , basis( 2, 1 ) , basis( 2, 0))
H_int = g*(ket010*ket101.dag()+ket101*ket010.dag())
H = H0 + H_int 
### bath parametres 
def be_ocuupation( beta , omega):
    x = beta * omega 
    if x > 100 :
        return 0.0 
    return 1.0 / (np.exp(x)- 1.0)
def thermal_bath(sm_op , omega , beta , gamma):
    nb = be_ocuupation(beta , omega)
    return [ np.sqrt(gamma * (nb+1) )* sm_op , np.sqrt(gamma * nb ) * sm_op.dag() , ]

def dissipator(L , rho):
    return L*rho*L.dag() -0.5*(L.dag()*L*rho + rho*L.dag()*L)
def heat_current(H_sys , rho , bath_ops):
    D = 0 
    for L in bath_ops:
        D = D + dissipator(L  ,rho)
    return np.real((H_sys *  D).tr())
def gfun(beta):
    return 1.0 / (1.0 + np.exp( beta * ez))
def run_collector(beta1):
    bath0 = thermal_bath(sm0 , e0 , beta0 , gamma0)
    bath1 = thermal_bath(sm1 , e1 , beta1 , gamma1)
    bathz = thermal_bath(smz , ez , betaz , gammaz)
    c_ops = bath0 +bath1 +bathz 
    rho_ss = steadystate(H , c_ops)
    J0 = heat_current(H , rho_ss , bath0)
    J1 = heat_current(H , rho_ss , bath1)
    Jz = heat_current(H , rho_ss , bathz)
    sigma = beta0 * (-J0) + beta1 * (-J1) + betaz*(-Jz)
    rho01 = rho_ss.ptrace([0 , 1])
    P01 = np.real(rho01[1 , 1])
    P10 = np.real(rho01[2 , 2])
    beta_v = -np.log(P10 /P01 )/ez
    return {'beta_v': beta_v , 'J0' : J0 , 'J1' : J1 , 'Jz': Jz , 'P01': P01 , 'P10': P10 , 'sigma ' :sigma }
def collector_current(Jz):
    return -Jz

def collector_current_1(beta_z , beta_v):
    return mu* ez * (gfun(beta_z) -gfun(beta_v))
def modulator_current(beta_z):
    return mu_p*ez*(gfun(beta_z) - gfun(beta_r))

def ode(t  , y , Jc):
    beta_z = y[0]
    Jm = modulator_current(beta_z)
    dbeta = -(beta_z ** 2 / C_out) * (Jc + Jm)
    return [dbeta]

def find_fixed_point(Jc):
    def fp_eq(beta_z):
        return Jc+modulator_current(beta_z)
    try:
        return brentq(fp_eq  , 0.01 , 20.0  , xtol = 1e-10)
    except ValueError:
        sol = solve_ivp(ode , [0 , 5000 ] , [1 , 0 ] , args = (Jc,) , rtol = 1e-10 , atol = 1e-10 , method = 'Radau')
        return sol.y[0 , -1]
def sweep(beta1_values):
    results = []
    for beta1 in beta1_values:
        r  = run_collector(beta1)
        Jc = collector_current_exact(r['Jz'])
        sol = solve_ivp(ode , [0 , 2000 ] , [1 , 0] , args = (Jc_exact) , rtol = 1e-9 , atol = 1e-9 , method = 'Radau') 
        beta_out_ode = sol.y[0 , -1]
        return results.append({''beta1':    beta1,
            'beta_v':   r['beta_v'],
            'J0':       r['J0'],
            'J1':       r['J1'],
            'Jz':       r['Jz'],
            'Jc_exact': Jc_exact,
            'sigma':    r['sigma'],
            'beta_out': beta_out_ode,
            'P01':      r['P01'],
            'P10':      r['P10']})
return results 



 
