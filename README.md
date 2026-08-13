# Repository: Autonomous Quantum Thermal Machines & Thermodynamic Computing

This repository contains numerical implementations for studying open quantum systems, non-Markovian environmental dynamics, and autonomous thermodynamic logic gates. The code scales from standard spin-boson models to specialized tensor network architectures  using libraries like (**OQuPy/TEMPO , QUTIP , ITENSOR**) and exact replication of the "thermodynamic neuron" perceptron framework.

## Current Research Structure

The clean project order is tracked in [ROADMAP.md](ROADMAP.md).

Current phase status:

- **Phase 1 complete:** baseline thermodynamic-neuron reconstruction in `Scripts/baseline_paper_reconstruction/`.
- **Phase 2 started:** single-qubit structured-bath OQuPy benchmark in `Scripts/non_markovian_extension/`.
- **Phase 3 started:** small convergence scan for time step, memory time, and SVD tolerance.
- **Phase 4 planned:** full 3-qubit NOT/NOR gate under structured baths.
- **Floquet bridge started:** driven buffer prototype in `Scripts/floquet_buffer_extension/`.
- **Floquet screening added:** parameter sweep and decision report for deciding when the full gate attempt is justified.
- **Gate prototype added:** NOT-gate screening layer in `Scripts/thermodynamic_floquet_gate_prototype/`.
- **HEOM output stage validated:** converged structured-bath dynamics, Floquet phase sweeps, process tomography, information backflow, disorder sampling, and work accounting in `Scripts/heom_structured_bath_validation/`.
- **Architecture comparison added:** direct, passive-buffer, and driven-buffer HEOM results are separated quantitatively and compared with an idealized CMOS inverter transfer model.
- **Fallbacks planned:** single-qubit thermodynamic protocol, reaction-coordinate benchmark, or tensor-network geometry study.

Clean terminal commands:

```bash
scripts/run_baseline.sh
scripts/run_non_markovian_smoke.sh
scripts/run_convergence_scan.sh
scripts/run_convergence_scan_strong.sh
scripts/run_dt_convergence_scan.sh
scripts/run_memory_convergence_scan.sh
scripts/run_non_markovian_research.sh
scripts/run_floquet_buffer.sh
scripts/run_floquet_buffer_sweep.sh
scripts/run_floquet_buffer_ablation.sh
scripts/run_thermodynamic_audit.sh
scripts/run_project_decision.sh
scripts/run_thermodynamic_floquet_gate_prototype.sh
scripts/run_full_floquet_not_gate.sh
scripts/run_phase_v_structured_backend.sh
scripts/run_phase_v_structured_backend_research.sh
scripts/run_heom_validation.sh
scripts/run_heom_floquet_phase_sweep.sh
scripts/run_heom_mps_scaling.sh
scripts/run_advanced_heom_experiments.sh
scripts/run_architecture_cmos_comparison.sh
scripts/run_heom_output_checks.sh
```

Use `run_non_markovian_smoke.sh` for quick checks. Use `run_non_markovian_research.sh` only when you are ready for a heavier OQuPy run.
The HEOM comparison commands reuse committed sweep data where appropriate;
`run_heom_output_checks.sh` is the fast generated-result acceptance test.

## 📂 Code Files Overview

The repository consists of four distinct simulation components. Together, they demonstrate how to model open quantum systems from a basic single-qubit scenario up to advanced, multi-reservoir autonomous quantum logic networks.

---

### 1. `boson_spin_boson.py`
* **Concept:** Models a single open two-level system (qubit) strongly coupled to a highly structured, non-Markovian bosonic environment.
* **Physics:** Implements an Ohmic spectral density with an exponential high-frequency cutoff:
  $$J(\omega) = 2 \alpha \omega e^{-\omega/\omega_c}$$
* **Method:** Replaces memoryless Markov approximations by utilizing the **TEMPO** (Time-Evolving Matrix Product Operator) algorithm via the `oqupy.tempo_compute` engine to efficiently track environmental influence histories.
* **Validation & Metrics:** Tracks trace preservation ($\text{Tr}(\rho) = 1$), matrix Hermiticity ($\rho = \rho^\dagger$), and state purity degradation ($\text{Tr}(\rho^2)$) to ensure numerical validity throughout the time evolution.

---

### 2. `fermionic_quantum_dot.py`
* **Concept:** Shifts the environment from bosonic noise fields to a setting involving physical **particle exchange and charge transport**.
* **Physics:** Implements the **Single Impurity Anderson Model (SIAM)** by simulating an isolated quantum dot level connected to a macroscopic metallic electron lead. The bath statistics are governed strictly by the Pauli exclusion principle and a Fermi-Dirac distribution.
* **Method:** Utilizes OQuPy's specialized `CustomFermionicBath` framework, mapping the environment's Grassmann anti-commutation relations into contractible tensor networks.
* **Validation & Metrics:** Verifies that the particle occupation expectation value $\langle n \rangle$ stays physically bounded within $[0.0, 1.0]$ and asymptotically relaxes toward the analytical Fermi-Dirac thermal equilibrium level.

---

### 3. `non_markovian_nor_gate.py`
* **Concept:** Implements an autonomous thermodynamic **NOR Gate** operating under strong coupling and deep memory backflow regimes.
* **Physics:** Investigates the gate's logic relaxation trajectories across three distinct power-law environmental memory profiles governed by the exponent $\zeta$:
  * **Sub-Ohmic ($\zeta = 0.5$):** Heavy non-Markovian memory holding onto deep past states.
  * **Ohmic ($\zeta = 1.0$):** Linear dissipative decay profile.
  * **Super-Ohmic ($\zeta = 3.0$):** Fast phononic mode environment behaving closer to a Markovian system.
* **Validation & Metrics:** Confirms that only the logical input state $(0,0)$ shifts the energy landscape to allow the output qubit to relax into its logic `1` state.

---

### 4. `thermodynamic_neuron_replication.py`
* **Concept:** Exact, literal replication of the weak-coupling analytical framework established in *Lipka-Bartosik, Perarnau-Llobet, & Brunner (2025)* (`baseline_gate_paper.pdf`).
* **Physics:** Replaces the infinite baths with a **finite-size output reservoir** whose inverse temperature $\beta_z$ is adjusted dynamically over time by the quantum machine's local heat currents ($j_z$):
  $$\frac{d\beta_z(t)}{dt} = -\frac{\beta_z(t)^2}{C} j_z(t)$$
* **Replication Targets:** 1. Recreates **Figure 3B** by tracking the steady-state inverse temperature output ($\beta_z^\infty$) of an autonomous **NOT Gate** across a parameter sweep of collector energies $\epsilon_1$.
  2. Synthesizes a multi-input **NOR Truth Table**, providing a physical verification that the autonomous thermal machine functions mathematically as a linearly separable machine-learning **perceptron**.

# Thermodyanmic Floquet Engineered NOT Gate - Analogous to CMOS voltage inverter .
Also this repo contains an extension to numerical methods related to the extension of the non markovian thermodynamic gate. That is a thermodynamic gate with floquet buffers acting as gate isolation between the reservoir and the collector qubits and a feedforward floquet sublayer in the modulator part providing and preventing logical fidelity error generating a thermodynamic not gate analogous to real life CMOS voltage inverter.
This extension would be numerically quite heavy because engineering non markovian floquet enginneered heat inverter would be need huge computation so remember to run codes in parts when pulling from this repo instead of running scripts directly and remeber to use GPU rendering or else your system might die !!!

## 🛠️ Installation & Dependencies

To run these simulations, you will need `Python 3.8+` along with standard scientific computing libraries and the specialized **OQuPy** package for time-evolving matrix product operator simulations.

```bash
pip install numpy matplotlib oqupy qutip
