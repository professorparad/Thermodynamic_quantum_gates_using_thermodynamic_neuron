import os
import numpy as np 
import oqupy
import matplotlib.pyplot as plt

output_dir = "quantum_simulation_results"
os.makedirs(output_dir, exist_ok=True)
metrics_file_path = os.path.join(output_dir, "simulation_metrics.txt")

with open(metrics_file_path, "w") as f_out:
    f_out.write("       OPEN SYSTEM DYNAMICS VALIDATION METRICS           \n")

omega = 1.0
omega_cutoff = 3.0
alpha = 0.2
temp = 0.5

regimes = {
    "Sub-Ohmic (zeta=0.5)": 0.5,
    "Ohmic (zeta=1.0)": 1.0,
    "Super-Ohmic (zeta=3.0)": 3.0
}

H_s = 0.5 * omega * oqupy.operators.sigma("z")
sys = oqupy.System(H_s)
interaction_operator = 0.5 * oqupy.operators.sigma("x")
initial_density_matrix_state_qubit = oqupy.operators.spin_dm("z+")

dt = 0.1
end_time = 10.0
truncation_tolerance = 1.0e-5
temporary_config = oqupy.TempoParameters(dt=dt, epsrel=truncation_tolerance)

collected_times = {}
collected_populations = {}
collected_purities = {}

for name, zeta_val in regimes.items():
    print(f"Running simulation configuration: {name}...")
    
    spectral_density = oqupy.PowerLawSD(
        alpha=alpha, 
        cutoff=omega_cutoff, 
        temperature=temp, 
        zeta=zeta_val
    )
    bath = oqupy.Bath(interaction_operator, spectral_density)
    
    dynamics = oqupy.tempo_compute(
        system=sys, 
        bath=bath, 
        initial_state=initial_density_matrix_state_qubit, 
        start_time=0.0, 
        end_time=end_time, 
        parameters=temporary_config
    )
    
    times, states = dynamics.times, dynamics.states
    collected_times[name] = times
    
    traces = []
    hermitian_errors = []
    purities = []
    population_z = []
    
    for rho in states:
        traces.append(np.real(np.trace(rho)))
        hermitian_diff = rho - rho.conj().T
        hermitian_errors.append(np.linalg.norm(hermitian_diff))
        purities.append(np.real(np.trace(np.dot(rho, rho))))
        population_z.append(np.real(np.trace(np.dot(rho, oqupy.operators.sigma("z")))))
        
    collected_populations[name] = population_z
    collected_purities[name] = purities
    
    max_trace_deviation = np.max(np.abs(np.array(traces) - 1.0))
    max_hermitian_error = np.max(hermitian_errors)
    
    print(f" -> Max Trace Dev: {max_trace_deviation:.2e} | Max Hermiticity Error: {max_hermitian_error:.2e}\n")
    
    with open(metrics_file_path, "a") as f_out:
        f_out.write(f"--- ENVIRONMENT REGIME: {name} ---\n")
        f_out.write(f"Max Trace Deviation from 1.0: {max_trace_deviation:.2e} (Target: < 1e-4)\n")
        f_out.write(f"Max Hermiticity Metric Error: {max_hermitian_error:.2e} (Target: < 1e-4)\n")
        f_out.write(f"Final Step Calculated Purity: {purities[-1]:.4f}\n")
        f_out.write(f"Final Expected Population Z:  {population_z[-1]:.4f}\n")
        f_out.write("-" * 57 + "\n\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
color_map = ["#8E44AD", "#2980B9", "#27AE60"] 

for i, (name, _) in enumerate(regimes.items()):
    c = color_map[i]
    ax1.plot(collected_times[name], collected_populations[name], label=name, color=c, linewidth=2.2)
    ax2.plot(collected_times[name], collected_purities[name], label=name, color=c, linewidth=2.2)

ax1.set_xlabel('Time ($t$)', fontsize=12)
ax1.set_ylabel(r'Expectation Value $\langle \sigma_z \rangle$', fontsize=12)
ax1.set_title('Qubit Trajectories across Custom Environments', fontsize=13, fontweight='bold')
ax1.grid(True, linestyle=":")
ax1.legend()

ax2.set_xlabel('Time ($t$)', fontsize=12)
ax2.set_ylabel(r'Purity $\mathrm{Tr}(\rho^2)$', fontsize=12)
ax2.set_title('Thermodynamic Decoherence Comparison', fontsize=13, fontweight='bold')
ax2.grid(True, linestyle=":")
ax2.legend()

plt.tight_layout()

plot_output_path = os.path.join(output_dir, "regime_comparison_plot.png")
plt.savefig(plot_output_path, dpi=300)
plt.show()

print(f"'{output_dir}/'")