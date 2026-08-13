from pathlib import Path
import csv

import numpy as np


def _sigmas():
    import qutip as qt

    return qt.sigmax(), qt.sigmay(), qt.sigmaz(), qt.sigmam(), qt.sigmap(), qt.qeye(2)


def beta_from_excited_population(population, epsilon):
    """Invert the two-level thermal occupation into an effective inverse temperature."""

    clipped = min(max(float(population), 1.0e-12), 1.0 - 1.0e-12)
    return float(np.log((1.0 - clipped) / clipped) / float(epsilon))


def trace_distance(rho_a, rho_b):
    """Trace distance between two reduced density matrices."""

    a = rho_a.full() if hasattr(rho_a, "full") else np.asarray(rho_a, dtype=complex)
    b = rho_b.full() if hasattr(rho_b, "full") else np.asarray(rho_b, dtype=complex)
    singular_values = np.linalg.svd(a - b, compute_uv=False)
    return 0.5 * float(np.sum(singular_values))


def thermal_jump_rates(beta, epsilon, gamma):
    """Thermal qubit jump rates satisfying detailed balance."""

    x = max(min(float(beta) * float(epsilon), 700.0), -700.0)
    p_excited = 1.0 / (1.0 + np.exp(x))
    down = float(gamma) * (1.0 - p_excited)
    up = float(gamma) * p_excited
    return down, up


def fermi_occupation(beta, epsilon):
    x = max(min(float(beta) * float(epsilon), 700.0), -700.0)
    return 1.0 / (1.0 + np.exp(x))


def inverse_fermi_occupation(population, epsilon):
    p = min(max(float(population), 1.0e-12), 1.0 - 1.0e-12)
    return float(np.log((1.0 - p) / p) / float(epsilon))


def virtual_temperature(beta0, beta1, epsilon1, epsilon_z):
    epsilon0 = float(epsilon1) + float(epsilon_z)
    return (epsilon0 / epsilon_z) * beta0 - (float(epsilon1) / epsilon_z) * beta1


def bounded_not_response(beta_v, params):
    gz_hot = fermi_occupation(params.beta_hot, params.epsilon_z)
    gz_cold = fermi_occupation(params.beta_cold, params.epsilon_z)
    gz_virtual = fermi_occupation(beta_v, params.epsilon_z)
    q_value = gz_hot * gz_virtual + gz_cold * (1.0 - gz_virtual)
    return inverse_fermi_occupation(q_value, params.epsilon_z)


def excited_population(state, subsystem=2, total_subsystems=3):
    """Excited-state population for one qubit inside a tensor-product state."""

    import qutip as qt

    projector = qt.basis(2, 1) * qt.basis(2, 1).dag()
    if total_subsystems == 1:
        reduced = state
    else:
        reduced = state.ptrace(subsystem)
    return float(np.real((projector * reduced).tr()))


def target_output_beta(beta1, params):
    """Baseline thermodynamic-neuron target beta for the output reservoir."""

    beta_v = virtual_temperature(params.beta0, beta1, params.epsilon1, params.epsilon_z)
    return float(bounded_not_response(beta_v, params)), float(beta_v)


def decode_output_bit(beta_out, params):
    """Decode output beta using the midpoint between hot and cold logical levels."""

    threshold = 0.5 * (params.beta_hot + params.beta_cold)
    return int(beta_out > threshold)


def _thermal_qubit_density(beta, epsilon):
    import qutip as qt

    p_excited = 1.0 / (1.0 + np.exp(float(beta) * float(epsilon)))
    return (1.0 - p_excited) * qt.basis(2, 0) * qt.basis(2, 0).dag() + p_excited * (
        qt.basis(2, 1) * qt.basis(2, 1).dag()
    )


def run_direct_three_qubit_case(params, case):
    """Three-qubit direct GKSL surrogate for thermodynamic NOT logic."""

    import qutip as qt

    sx, _, sz, sm, sp, ident = _sigmas()
    beta_target, beta_v = target_output_beta(case["beta1"], params)

    sz0 = qt.tensor(sz, ident, ident)
    sz1 = qt.tensor(ident, sz, ident)
    szz = qt.tensor(ident, ident, sz)
    lower0 = qt.tensor(sp, ident, ident)
    raise0 = qt.tensor(sm, ident, ident)
    lower1 = qt.tensor(ident, sp, ident)
    raise1 = qt.tensor(ident, sm, ident)
    lowerz = qt.tensor(ident, ident, sp)
    raisez = qt.tensor(ident, ident, sm)

    hamiltonian = (
        0.5 * (params.epsilon1 + params.epsilon_z) * sz0
        + 0.5 * params.epsilon1 * sz1
        + 0.5 * params.epsilon_z * szz
    )

    down0, up0 = thermal_jump_rates(params.beta0, params.epsilon1 + params.epsilon_z, params.input_gamma)
    down1, up1 = thermal_jump_rates(case["beta1"], params.epsilon1, params.input_gamma)
    downz, upz = thermal_jump_rates(beta_target, params.epsilon_z, params.output_gamma)
    c_ops = [
        np.sqrt(down0) * lower0,
        np.sqrt(up0) * raise0,
        np.sqrt(down1) * lower1,
        np.sqrt(up1) * raise1,
        np.sqrt(downz) * lowerz,
        np.sqrt(upz) * raisez,
    ]

    rho0 = qt.tensor(
        _thermal_qubit_density(params.beta0, params.epsilon1 + params.epsilon_z),
        _thermal_qubit_density(case["beta1"], params.epsilon1),
        _thermal_qubit_density(0.5 * (params.beta_hot + params.beta_cold), params.epsilon_z),
    )
    tlist = np.linspace(0.0, params.t_end, params.num_steps + 1)
    result = qt.mesolve(hamiltonian, rho0, tlist, c_ops=c_ops, options={"progress_bar": ""})
    return _rows_from_result(
        "direct_three_qubit",
        case,
        params,
        result.states,
        tlist,
        beta_target,
        beta_v,
        output_subsystem=2,
        total_subsystems=3,
        drive_power_values=None,
    )


def run_buffered_three_qubit_case(params, case):
    """Three-qubit GKSL surrogate with Floquet buffer on the output branch."""

    import qutip as qt

    sx, _, sz, sm, sp, ident = _sigmas()
    beta_target, beta_v = target_output_beta(case["beta1"], params)

    ops = []
    for op in [sz, sm, sp, sx]:
        ops.append([qt.tensor(*[op if i == target else ident for i in range(4)]) for target in range(4)])
    sz_ops, sm_ops, sp_ops, sx_ops = ops
    lower_ops = sp_ops
    raise_ops = sm_ops

    h_static = (
        0.5 * (params.epsilon1 + params.epsilon_z) * sz_ops[0]
        + 0.5 * params.epsilon1 * sz_ops[1]
        + 0.5 * params.epsilon_z * sz_ops[2]
        + 0.5 * params.epsilon_buffer * sz_ops[3]
        + params.buffer_coupling * sx_ops[2] * sx_ops[3]
    )
    h_drive = params.drive_amplitude * sx_ops[3]

    def drive_coefficient(t, omega):
        return np.cos(omega * t)

    hamiltonian = [h_static, [h_drive, drive_coefficient]]

    down0, up0 = thermal_jump_rates(params.beta0, params.epsilon1 + params.epsilon_z, params.input_gamma)
    down1, up1 = thermal_jump_rates(case["beta1"], params.epsilon1, params.input_gamma)
    downz, upz = thermal_jump_rates(beta_target, params.epsilon_z, params.output_gamma)
    downf, upf = thermal_jump_rates(beta_target, params.epsilon_buffer, params.buffer_gamma)
    c_ops = [
        np.sqrt(down0) * lower_ops[0],
        np.sqrt(up0) * raise_ops[0],
        np.sqrt(down1) * lower_ops[1],
        np.sqrt(up1) * raise_ops[1],
        np.sqrt(downz) * lower_ops[2],
        np.sqrt(upz) * raise_ops[2],
        np.sqrt(downf) * lower_ops[3],
        np.sqrt(upf) * raise_ops[3],
    ]

    rho0 = qt.tensor(
        _thermal_qubit_density(params.beta0, params.epsilon1 + params.epsilon_z),
        _thermal_qubit_density(case["beta1"], params.epsilon1),
        _thermal_qubit_density(0.5 * (params.beta_hot + params.beta_cold), params.epsilon_z),
        _thermal_qubit_density(params.beta_hot, params.epsilon_buffer),
    )
    tlist = np.linspace(0.0, params.t_end, params.num_steps + 1)
    result = qt.mesolve(
        hamiltonian,
        rho0,
        tlist,
        c_ops=c_ops,
        args={"omega": params.drive_frequency},
        options={"progress_bar": ""},
    )
    drive_power_values = []
    for time, state in zip(tlist, result.states):
        drive_power_operator = (
            -params.drive_amplitude
            * params.drive_frequency
            * np.sin(params.drive_frequency * time)
            * sx_ops[3]
        )
        drive_power_values.append(float(np.real((drive_power_operator * state).tr())))

    return _rows_from_result(
        "floquet_buffered_three_qubit",
        case,
        params,
        result.states,
        tlist,
        beta_target,
        beta_v,
        output_subsystem=2,
        total_subsystems=4,
        drive_power_values=drive_power_values,
    )


def _rows_from_result(
    architecture,
    case,
    params,
    states,
    tlist,
    beta_target,
    beta_v,
    output_subsystem,
    total_subsystems,
    drive_power_values,
):
    rows = []
    for idx, (time, state) in enumerate(zip(tlist, states)):
        pop = excited_population(state, output_subsystem, total_subsystems)
        beta_eff = beta_from_excited_population(pop, params.epsilon_z)
        output_bit = decode_output_bit(beta_eff, params)
        rows.append(
            {
                "architecture": architecture,
                "input_bit": case["input_bit"],
                "beta1": case["beta1"],
                "expected_output_bit": case["expected_output_bit"],
                "time": float(time),
                "beta_virtual": beta_v,
                "target_output_beta": beta_target,
                "output_excited_population": pop,
                "output_beta_effective": beta_eff,
                "decoded_output_bit": output_bit,
                "is_correct": output_bit == case["expected_output_bit"],
                "drive_power": 0.0 if drive_power_values is None else drive_power_values[idx],
            }
        )
    return rows, states


def run_truth_table(params):
    """Run direct and buffered three-qubit truth-table simulations."""

    from parameters import logical_inputs

    all_rows = []
    final_states = {}
    for case in logical_inputs(params):
        direct_rows, direct_states = run_direct_three_qubit_case(params, case)
        buffered_rows, buffered_states = run_buffered_three_qubit_case(params, case)
        all_rows.extend(direct_rows)
        all_rows.extend(buffered_rows)
        final_states[("direct_three_qubit", case["input_bit"])] = direct_states[-1].ptrace(2)
        final_states[("floquet_buffered_three_qubit", case["input_bit"])] = buffered_states[-1].ptrace(2)

    summary_rows = []
    for architecture in ["direct_three_qubit", "floquet_buffered_three_qubit"]:
        final_by_input = {
            row["input_bit"]: row
            for row in all_rows
            if row["architecture"] == architecture and abs(row["time"] - params.t_end) < 1.0e-9
        }
        distance = trace_distance(final_states[(architecture, 0)], final_states[(architecture, 1)])
        arch_rows = [row for row in all_rows if row["architecture"] == architecture]
        work = integrate_drive_work(arch_rows)
        correct = sum(1 for row in final_by_input.values() if row["is_correct"])
        summary_rows.append(
            {
                "architecture": architecture,
                "truth_table_correct": correct,
                "truth_table_total": len(final_by_input),
                "truth_table_accuracy": correct / max(len(final_by_input), 1),
                "final_trace_distance": distance,
                "integrated_drive_work_proxy": work,
            }
        )
    return all_rows, summary_rows


def integrate_drive_work(rows):
    selected = [row for row in rows if row["input_bit"] == 1]
    if len(selected) < 2:
        return 0.0
    times = np.array([row["time"] for row in selected], dtype=float)
    power = np.array([row["drive_power"] for row in selected], dtype=float)
    return float(np.trapz(power, times))


def final_truth_rows(rows, params):
    return [row for row in rows if abs(row["time"] - params.t_end) < 1.0e-9]


def save_csv(rows, output_path, headers):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path
