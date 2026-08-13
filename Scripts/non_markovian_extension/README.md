# Non-Markovian Extension

This folder is for the next phase after the baseline-paper reconstruction.

The goal is **not** to jump directly to the full three-qubit thermodynamic neuron. The first target is a controlled single-qubit non-Markovian benchmark, following the staged plan in `ROADMAP.md`.

## Phase Goal

Build and validate:

```text
single qubit + structured bosonic bath
```

using process-tensor/TEMPO style numerics, preferably through OQuPy if available.

## Current Status

Implemented:

- OQuPy single-qubit structured-bath benchmark.
- Sub-Ohmic, Ohmic, and Super-Ohmic comparison.
- CSV export for full dynamics and summary diagnostics.
- Plot of `sigma_z` dynamics and purity.
- Utility tests for spectral density and trace distance.

## Why This Comes Next

The professor proposal warns that the full thermodynamic gate has a serious bond-dimension risk:

```text
3 qubits + 3 independent structured baths
```

can become too expensive before reaching the long-time steady state. The single-qubit benchmark is the calibration step that tells us what memory length, time step, and SVD tolerance are realistic on this machine.

## Planned Layout

```text
non_markovian_extension/
  main.py
  parameters.py
  src/
    spectral_density.py
    single_qubit_benchmark.py
    observables.py
    convergence.py
  visualization/
    plots.py
  tests/
    test_spectral_density.py
  outputs/
```

## Target Observables

- Qubit excited-state population.
- Trace distance between two initial states.
- Trace preservation.
- Positivity.
- Relaxation time.
- Non-Markovian revival/backflow indicator.
- If safe: energy-current diagnostics.

Currently generated:

```text
outputs/single_qubit_structured_bath_dynamics.csv
outputs/single_qubit_structured_bath_summary.csv
outputs/single_qubit_regime_comparison.png
```

## Parameters To Sweep

- Coupling strength `alpha`.
- Spectral exponent `s`.
- Cutoff frequency `omega_c`.
- Temperature / inverse temperature `beta`.
- Time step `dt`.
- Memory length.
- SVD tolerance.
- Maximum bond dimension.

## Implementation Notes

Primary route:

```text
OQuPy / process tensor / TEMPO
```

Fallback route if OQuPy is unavailable:

```text
QuTiP structured pseudomode or reaction-coordinate benchmark
```

The first deliverable should be a small plot comparing Markovian-like relaxation to non-Markovian revival dynamics.

Run:

```bash
.venv/bin/python Scripts/non_markovian_extension/main.py
```

Clean terminal commands from the repository root:

```bash
scripts/run_non_markovian_smoke.sh
scripts/run_non_markovian_research.sh
scripts/run_convergence_scan.sh
scripts/run_convergence_scan_strong.sh
scripts/run_dt_convergence_scan.sh
scripts/run_memory_convergence_scan.sh
```

Use `smoke` during normal development. Use `research` only when you are ready to wait for a heavier OQuPy calculation.
Use `run_convergence_scan_strong.sh` when you need a more serious Phase 3 decision.
Use the separate `dt` and `memory` scans to diagnose which numerical setting is limiting convergence.

Current short-time recommendation after the first Phase 3 diagnostics:

```text
dt = 0.05
memory_time = 1.0
svd_tolerance = 1e-6
```

Run tests:

```bash
.venv/bin/python -m unittest discover -s Scripts/non_markovian_extension/tests
```
