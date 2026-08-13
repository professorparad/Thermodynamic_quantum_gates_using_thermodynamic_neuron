# Thermodynamic Quantum Gates Research Roadmap

This roadmap follows the hierarchy from the project PDFs:

1. Reconstruct the Markovian baseline gate.
2. Build a single-qubit non-Markovian benchmark with OQuPy.
3. Study memory effects and convergence.
4. Try the full 3-qubit NOT/NOR gate under structured baths.
5. If the full multi-bath simulation is too expensive, pivot to a single-qubit thermodynamic protocol or reaction-coordinate benchmark.

Theory bridge note:

```text
papers_to_read_implemet/floquet_buffer_mps_theory_bridge.tex
papers_to_read_implemet/floquet_buffer_mps_theory_bridge.pdf
```

This note explains the chain from baseline thermodynamic neurons to Nakajima-Zwanzig memory, Floquet buffers, process tensors/MPS compression, and logical output distinguishability.

## Phase 1: Markovian Baseline Gate

**Status: complete as an analytical reconstruction.**

Location:

```text
Scripts/baseline_paper_reconstruction/
```

What is done:

- Fig. 2 virtual-temperature regimes.
- Fig. 3B NOT transfer curve.
- Fig. 3C Gaussian decoding error plus reset-model entropy production.
- Fig. 6 NOR response.
- Fig. 7 3-majority response slices.
- Fig. 8 XOR network response.

What remains inside Phase 1:

- Optionally compare local reset-model results against local/global GKSL numerics.

## Phase 2: Single-Qubit Non-Markovian Benchmark

**Status: next active phase.**

Location:

```text
Scripts/non_markovian_extension/
```

Goal:

Build the smallest serious non-Markovian benchmark before attempting a multi-qubit thermodynamic gate.

Physical system:

```text
single qubit + structured bosonic bath
```

Primary observable:

- Reduced-state dynamics of the qubit.
- Trace preservation and positivity.
- Relaxation toward thermal state.
- Trace distance between two input preparations.
- Non-Markovian revival/backflow indicators.

Structured bath:

```text
J(omega) = 2 alpha omega^s exp(-omega / omega_c)
```

Parameters to sweep:

- Coupling strength `alpha`.
- Cutoff frequency `omega_c`.
- Ohmic exponent `s`.
- Bath temperature.
- Process-tensor memory length.
- Time step.
- SVD tolerance.

Deliverable:

- A convergence table showing when dynamics is stable under smaller `dt`, larger memory length, and tighter SVD tolerance.
- A plot showing Markovian-like relaxation versus non-Markovian revival behavior.

## Phase 3: Memory And Convergence Study

**Status: started; first single-qubit convergence diagnostics pass at short time.**

Main question:

How expensive is environmental memory for the gate problem?

Quantities to record:

- Time step `dt`.
- Total simulation time.
- Memory length / bath correlation time.
- SVD tolerance.
- Maximum bond dimension.
- Runtime.
- Final observable drift.

Deliverable:

```text
outputs/convergence_table.csv
outputs/memory_scaling.png
```

Current diagnostic outputs:

```text
Scripts/non_markovian_extension/outputs/dt_scan_report.txt
Scripts/non_markovian_extension/outputs/memory_scan_report.txt
```

Current recommendation for the single-qubit Ohmic benchmark at `t_end = 1.0`:

```text
dt = 0.05
memory_time = 1.0
svd_tolerance = 1e-6
```

Both the isolated `dt` scan and isolated `memory_time` scan pass the current tolerance:

```text
|last step drift in final sigma_z| < 1e-3
```

Decision criterion:

Only move to the 3-qubit gate when the single-qubit benchmark is stable and the required bond dimension is computationally reasonable.

## Phase 4: Full 3-Qubit Gate Under Structured Baths

**Status: bridge model started; full gate still planned, high risk.**

Bridge location:

```text
Scripts/floquet_buffer_extension/
```

The bridge model tests the professor/student Floquet-buffer idea before embedding it in the full gate:

```text
logical qubit S <-> driven buffer F(t) <-> thermal bath
```

This is compared against:

```text
logical qubit S <-> thermal bath
```

Current bridge observables:

- final trace distance between two logical outputs,
- logical-qubit population,
- logical-qubit purity,
- integrated drive-work proxy.

Screening and decision commands:

```text
scripts/run_floquet_buffer_sweep.sh
scripts/run_floquet_buffer_ablation.sh
scripts/run_thermodynamic_audit.sh
scripts/run_project_decision.sh
scripts/run_thermodynamic_floquet_gate_prototype.sh
scripts/run_full_floquet_not_gate.sh
scripts/run_phase_v_structured_backend.sh
scripts/run_phase_v_structured_backend_research.sh
```

Current decision output:

```text
Scripts/project_decision/outputs/phase2_phase3_decision_report.txt
```

The current decision is `RISKY-GO`: attempt a minimal NOT-style prototype, but do
not claim the full thermodynamic non-Markovian gate yet.

Targets:

- Start with NOT gate because it is the baseline paper's simplest thermodynamic neuron.
- Then try NOR if NOT is stable.

Gate observables:

- Logic fidelity.
- Trace distance between output states.
- Steady-state output population.
- Relaxation/switching time.

Do **not** overclaim these until the thermodynamic audit is complete:

- Heat.
- Entropy production.
- Work cost at strong coupling.

Reason:

Beyond weak coupling and Markovianity, reduced-system heat currents can become ambiguous because interaction energy and bath memory matter.

## Phase 5: Pivot Paths If Full Gate Explodes

If the full 3-qubit multi-bath simulation becomes too large, use one of these publishable fallback directions.

### Pivot A: Single-Qubit Thermodynamic Protocol

Study:

```text
driven qubit + structured bath
```

Deliverables:

- Exact non-Markovian entropy-production audit.
- Speed/error/work trade-off.
- Markovian-limit recovery.

### Pivot B: Reaction-Coordinate Benchmark

Map the dominant bath mode into the system:

```text
qubit + reaction coordinate + residual Markovian bath
```

Purpose:

- Validate process-tensor results.
- Provide a lower-cost strong-coupling benchmark.

### Pivot C: Tensor-Network Geometry Study

If ordinary PT-MPO contraction fails:

- Analyze tree tensor network layouts for multiple baths.
- Estimate scaling improvements.
- Produce a thesis-quality feasibility result even without full gate simulation.

## Immediate Next Checklist

- [x] Baseline analytical reconstruction.
- [x] Add exact reset-model entropy production for Fig. 3C.
- [x] Check whether `oqupy` is installed in `.venv`.
- [x] Create a single-qubit OQuPy benchmark.
- [x] Define convergence CSV schema.
- [x] Generate first memory-scaling plot.
- [x] Add quick and strong convergence terminal commands.
- [x] Add separate dt and memory convergence scans.
- [x] Add first Floquet-buffer bridge model.
- [x] Add Floquet-buffer parameter sweep.
- [x] Add Floquet-buffer ablation study.
- [x] Add thermodynamic mathematics/interpretation audit.
- [x] Add Phase 2/3 decision report.
- [x] Add thermodynamic Floquet NOT prototype screening folder.
- [x] Add minimal NOT experiment specification from current evidence.
- [x] Implement runnable three-qubit GKSL NOT surrogate with direct and Floquet-buffered output branch.
- [x] Add Phase V OQuPy/TEMPO structured-bath backend with direct and Floquet-buffered truth-table interface.
- [ ] Tune Phase V PT-MPO/HEOM backend until the structured-bath truth table passes or map the failure boundary.
- [ ] Decide whether full 3-qubit gate is computationally realistic.
