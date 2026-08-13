from pathlib import Path

import numpy as np

from .observables import purity, trace_distance
from .thermal_rates import thermal_jump_rates


def _sigmas():
    import qutip as qt

    return qt.sigmax(), qt.sigmay(), qt.sigmaz(), qt.sigmam(), qt.sigmap(), qt.qeye(2)


def _partial_system_state(state):
    return state.ptrace(0)


def _system_excited_population(system_state):
    import qutip as qt

    projector = qt.basis(2, 1) * qt.basis(2, 1).dag()
    return float(np.real((projector * system_state).tr()))


def run_direct_baseline(params, initial_label):
    """Direct Markovian damping of S by B."""

    import qutip as qt

    sx, _, sz, sm, sp, _ = _sigmas()
    hamiltonian = 0.5 * params.epsilon_s * sz
    down, up = thermal_jump_rates(params.bath_beta, params.epsilon_s, params.direct_gamma)
    c_ops = [np.sqrt(down) * sm, np.sqrt(up) * sp]
    rho0 = qt.basis(2, 0) if initial_label == "ground" else qt.basis(2, 1)
    rho0 = rho0 * rho0.dag()
    tlist = np.linspace(0.0, params.t_end, params.num_steps + 1)
    result = qt.mesolve(hamiltonian, rho0, tlist, c_ops=c_ops, options={"progress_bar": ""})
    rows = []
    for time, state in zip(tlist, result.states):
        rows.append(
            {
                "architecture": "direct",
                "initial_state": initial_label,
                "time": float(time),
                "system_excited_population": _system_excited_population(state),
                "system_purity": purity(state),
                "drive_power": 0.0,
            }
        )
    return rows, result.states


def run_floquet_buffer(params, initial_label):
    """Driven buffer between S and B."""

    import qutip as qt

    sx, _, sz, sm, sp, ident = _sigmas()
    sx_s = qt.tensor(sx, ident)
    sz_s = qt.tensor(sz, ident)
    sx_f = qt.tensor(ident, sx)
    sz_f = qt.tensor(ident, sz)
    sm_f = qt.tensor(ident, sm)
    sp_f = qt.tensor(ident, sp)

    h_static = (
        0.5 * params.epsilon_s * sz_s
        + 0.5 * params.epsilon_f * sz_f
        + params.coupling * sx_s * sx_f
    )
    h_drive = params.drive_amplitude * sx_f
    def drive_coefficient(t, omega):
        return np.cos(omega * t)

    hamiltonian = [h_static, [h_drive, drive_coefficient]]

    down, up = thermal_jump_rates(params.bath_beta, params.epsilon_f, params.bath_gamma)
    c_ops = [np.sqrt(down) * sm_f, np.sqrt(up) * sp_f]

    system_ket = qt.basis(2, 0) if initial_label == "ground" else qt.basis(2, 1)
    buffer_ket = qt.basis(2, 0)
    rho0 = qt.tensor(system_ket, buffer_ket)
    rho0 = rho0 * rho0.dag()
    tlist = np.linspace(0.0, params.t_end, params.num_steps + 1)
    result = qt.mesolve(
        hamiltonian,
        rho0,
        tlist,
        c_ops=c_ops,
        args={"omega": params.drive_frequency},
        options={"progress_bar": ""},
    )

    rows = []
    reduced_states = []
    for time, state in zip(tlist, result.states):
        system_state = _partial_system_state(state)
        reduced_states.append(system_state)
        drive_power_operator = (
            -params.drive_amplitude
            * params.drive_frequency
            * np.sin(params.drive_frequency * time)
            * sx_f
        )
        drive_power = float(np.real((drive_power_operator * state).tr()))
        rows.append(
            {
                "architecture": "floquet_buffer",
                "initial_state": initial_label,
                "time": float(time),
                "system_excited_population": _system_excited_population(system_state),
                "system_purity": purity(system_state),
                "drive_power": drive_power,
            }
        )
    return rows, reduced_states


def run_comparison(params):
    """Run direct and Floquet-buffer comparisons for two logical states."""

    all_rows = []
    final_states = {}
    for architecture_runner in [run_direct_baseline, run_floquet_buffer]:
        for initial_label in ["ground", "excited"]:
            rows, states = architecture_runner(params, initial_label)
            all_rows.extend(rows)
            architecture = rows[0]["architecture"]
            final_states[(architecture, initial_label)] = states[-1]

    summary_rows = []
    for architecture in ["direct", "floquet_buffer"]:
        distance = trace_distance(
            final_states[(architecture, "ground")],
            final_states[(architecture, "excited")],
        )
        arch_rows = [row for row in all_rows if row["architecture"] == architecture]
        work = integrate_drive_work(arch_rows)
        summary_rows.append(
            {
                "architecture": architecture,
                "final_trace_distance": distance,
                "integrated_drive_work": work,
            }
        )
    return all_rows, summary_rows


def integrate_drive_work(rows):
    """Integrate drive power over time for one architecture."""

    rows = [row for row in rows if row["initial_state"] == "excited"]
    if len(rows) < 2:
        return 0.0
    times = np.array([row["time"] for row in rows], dtype=float)
    power = np.array([row["drive_power"] for row in rows], dtype=float)
    return float(np.trapz(power, times))


def save_rows_csv(rows, output_path, headers):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in headers) + "\n")
    return output_path
