import numpy as np 

def nbar(beta  , omega):
    x = beta * omega 
    if x > 100:
        return 0.0 
    if x < -100 :
        return 1e10 
    return 1.0 / (np.exp(x)- 1.0)

def thermal_bath(sm_op, omega, beta, gamma):
    nb = nbar(beta, omega)
    return [np.sqrt(gamma * (nb + 1.0)) * sm_op, np.sqrt(gamma * nb) * sm_op.dag()]

def global_thermal_bath(H, coupling_op, beta, gamma, transition_tol=1e-10):
    energies, states = H.eigenstates()
    transitions = {}

    for high_index, high_energy in enumerate(energies):
        high = states[high_index]
        for low_index, low_energy in enumerate(energies):
            omega = float(np.real(high_energy - low_energy))
            if omega <= transition_tol:
                continue

            low = states[low_index]
            amplitude = low.dag() * coupling_op * high
            if abs(amplitude) <= transition_tol:
                continue

            key = round(omega / transition_tol) * transition_tol
            transitions.setdefault(key, 0 * H)
            transitions[key] += amplitude * low * high.dag()

    c_ops = []
    for omega, jump_down in transitions.items():
        nb = nbar(beta, omega)
        if jump_down.norm() <= transition_tol:
            continue
        c_ops.append(np.sqrt(gamma * (nb + 1.0)) * jump_down)
        if nb > 0.0:
            c_ops.append(np.sqrt(gamma * nb) * jump_down.dag())
    return c_ops
