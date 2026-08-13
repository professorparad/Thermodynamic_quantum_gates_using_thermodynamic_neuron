# Floquet Buffer Extension

This folder implements the first concrete version of the Floquet-buffer idea from the project notes.

The goal is to avoid a theory/code mismatch. The first model is deliberately small:

```text
logical qubit S <-> driven buffer F(t) <-> Markovian thermal bath
```

This is not yet the full thermodynamic NOT/NOR gate. It is the bridge calculation that tests whether a periodically driven intermediate subsystem can change logical-state distinguishability before we place it inside the full thermodynamic neuron.

## Physics Model

The logical qubit has:

```text
H_S = 0.5 * epsilon_s * sigma_z
```

The Floquet buffer has:

```text
H_F(t) = 0.5 * epsilon_f * sigma_z + drive_amplitude * cos(drive_frequency * t) * sigma_x
```

The qubit-buffer coupling is:

```text
H_SF = coupling * sigma_x(S) sigma_x(F)
```

The bath acts only on the buffer through thermal Lindblad jumps. This realizes the architecture:

```text
S <-> F(t) <-> B
```

For comparison, the code also runs a direct baseline:

```text
S <-> B
```

## Observables

- Excited-state population of the logical qubit.
- Purity of the logical qubit.
- Trace distance between two logical outputs.
- Drive work proxy:

```text
W = integral Tr[rho(t) dH_F(t)/dt] dt
```

The trace distance is the clean operational metric: if two output states remain more distinguishable with the buffer, the buffer is helping logical readability.

## Run

From the repository root:

```bash
scripts/run_floquet_buffer.sh
```

Outputs:

```text
Scripts/floquet_buffer_extension/outputs/floquet_buffer_summary.csv
Scripts/floquet_buffer_extension/outputs/floquet_buffer_dynamics.csv
Scripts/floquet_buffer_extension/outputs/floquet_buffer_comparison.png
```

Run the small screening sweep:

```bash
scripts/run_floquet_buffer_sweep.sh
```

Sweep outputs:

```text
Scripts/floquet_buffer_extension/outputs/floquet_parameter_sweep.csv
Scripts/floquet_buffer_extension/outputs/floquet_parameter_sweep_report.txt
Scripts/floquet_buffer_extension/outputs/floquet_parameter_sweep.png
```

## Current Scope

This is a weak open-system/Floquet bridge model using QuTiP time-dependent GKSL evolution.

It is theory-consistent for:

- Markovian bath attached to the buffer.
- Classical periodic drive.
- Stroboscopic or time-resolved readout.

It is not yet:

- a non-Markovian process-tensor Floquet buffer,
- a full 3-qubit thermodynamic neuron,
- a strong-coupling thermodynamic heat/work proof.

## First Result

The first default run produced:

```text
direct final trace distance        ~= 0.135
Floquet-buffer final trace distance ~= 0.182
integrated drive work proxy         ~= 0.460
```

So the first parameter set shows the intended qualitative trade-off:

```text
better output distinguishability, but positive drive cost
```

This does not prove the full thesis claim yet. It is the first controlled bridge calculation showing that the Floquet-buffer idea can be tested with precise observables.

The sweep is a screening tool only. It chooses promising drive and coupling regions for the
next gate prototype; it is not a final thermodynamic heat/work proof.
