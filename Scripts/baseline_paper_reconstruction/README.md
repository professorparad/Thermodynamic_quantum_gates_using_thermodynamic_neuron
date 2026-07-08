# Baseline Paper Reconstruction

Clean reconstruction workspace for:

**Lipka-Bartosik, Perarnau-Llobet, and Brunner, "Thermodynamic Computing via Autonomous Quantum Thermal Machines" (2025).**

This folder is intentionally separate from the older GKSL/QuTiP experiments. The goal here is to reconstruct the **baseline paper model first**, using the equations in the paper directly, before adding heavier master-equation simulations.

## What This Reconstructs

The current implementation reconstructs the paper at the **equation/model level**. It does not yet simulate the full microscopic open quantum dynamics. Instead, it implements the reset-model thermodynamic-neuron formulas used by the paper to explain the figures.

| Paper item | Status | What is implemented |
| --- | --- | --- |
| Fig. 2 | Reconstructed | Virtual temperature regimes of the three-qubit thermal machine. |
| Fig. 3B | Reconstructed | Analytic NOT-gate transfer curve from the bounded thermodynamic-neuron response. |
| Fig. 3C | Partial/proxy | Gaussian decoding error is implemented. Dissipation is a monotone proxy, not full entropy-production integration yet. |
| Fig. 6 | Reconstructed | NOR response surface from the paper's virtual-temperature formula. |
| Fig. 7 | Reconstructed as slices | 3-majority response volume, visualized as 2D slices through `beta3`. |
| Fig. 8 | Reconstructed as network model | XOR built from NAND/OR feeding AND, following the paper's network idea. |
| Fig. 9 | Covered by NOT machinery | Same analytic response as Appendix A with adjustable parameters. |
| Schematic figures | Not reproduced | Fig. 1, Fig. 3A, Fig. 4, Fig. 5 are conceptual diagrams, not numerical reconstructions. |

## Physics Being Reconstructed

The paper models computation using autonomous quantum thermal machines. Logical inputs are encoded as bath inverse temperatures, heat currents drive a finite output reservoir, and the final reservoir inverse temperature is decoded as a logical output.

### 1. Thermal Qubit Occupation

The baseline model uses a two-level qubit thermal occupation:

```text
g(beta epsilon) = 1 / (1 + exp(beta epsilon))
```

This is implemented in:

```text
src/thermal_functions.py
```

This is important: this baseline folder does **not** use the bosonic `nbar = 1/(exp(beta omega)-1)` model from the older code. The paper's thermodynamic-neuron equations are based on qubit/Fermi-sigmoid occupations.

### 2. Virtual Temperature

For the three-qubit collector, the relevant two-dimensional virtual qubit has inverse temperature:

```text
beta_v = (epsilon0 / epsilon_z) beta0 - (epsilon1 / epsilon_z) beta1
epsilon_z = epsilon0 - epsilon1
```

This is implemented in:

```text
src/not_gate.py
src/virtual_qubit.py
```

This reconstructs the physics behind Fig. 2 and the collector part of Fig. 3B: changing the input temperature `beta1` changes `beta_v`, so the collector switches between cooling/heating regimes.

### 3. Collector and Modulator Currents

The paper's reset-model currents are:

```text
j_C = mu epsilon_z [g_z(beta_z) - g_z(beta_v)]
j_M = mu' epsilon_z [g_z(beta_z) - g_z(beta_r)]
```

The collector tries to thermalize the output reservoir toward `beta_v`. The modulator confines the output to a usable logical range.

Implemented in:

```text
src/not_gate.py
```

### 4. Bounded Output Response

The paper solves the steady-state condition `j_C + j_M = 0` and obtains a bounded output:

```text
beta_z^infinity = (1 / epsilon_z) log(Q(beta_v)^(-1) - 1)
```

where `Q(beta_v)` mixes the hot/cold logical temperatures through the qubit occupation function.

Implemented in:

```text
src/not_gate.py
src/logic_gates.py
```

This is the core reconstruction behind Fig. 3B, Fig. 6, Fig. 7, and Fig. 8.

### 5. Perceptron / Logic-Gate Mapping

For multi-input gates, the paper shows that the virtual temperature acts like a perceptron score:

```text
beta_v = alpha * linear_score(inputs)
```

Implemented examples:

```text
NOR:        beta_v = alpha * (1 - 2 beta1 - 2 beta2)
3-majority: beta_v = alpha * (4 - 3 beta1 - 3 beta2 - 3 beta3)
XOR:        network composition: NAND and OR -> AND
```

Implemented in:

```text
src/logic_gates.py
```

The larger `alpha` is, the sharper the thermal sigmoid response becomes. This matches the paper's statement that larger energy scales improve fidelity but increase thermodynamic cost.

## What Is Exact vs Approximate

### Exact / Direct Equation Reconstructions

These are direct implementations of formulas from the paper:

- Fermi-sigmoid qubit occupation.
- Virtual temperature formulas.
- Reset-model collector and modulator currents.
- Bounded steady-state response `beta_z^infinity`.
- NOR virtual temperature formula.
- 3-majority virtual temperature formula.

### Reconstructed But Visualized Differently

- Fig. 7 is a three-input response. The paper shows a 3D-style response; here it is plotted as several 2D slices through `beta3`.
- Fig. 8 is reconstructed as a deterministic NAND/OR/AND thermodynamic-neuron network. The paper says the weights were obtained by training; here the logical network is manually encoded with perceptron hyperplanes.

### Approximate / Not Yet Exact

- Fig. 3C dissipation is **not exact yet**.
- The average error follows the paper's Gaussian decoding idea.
- The dissipation axis is currently a monotone analytic proxy based on energy scale.
- To make Fig. 3C exact, we need to implement the full time-dependent heat currents and integrate entropy production:

```text
Sigma = integral Sigma_dot dt
Sigma_dot = -sum_k beta_k j_k
```

That upgrade should go into `src/tradeoff.py`.

## Layout

```text
baseline_paper_reconstruction/
  main.py
  parameters.py
  src/
    thermal_functions.py
    virtual_qubit.py
    not_gate.py
    logic_gates.py
    tradeoff.py
    io_utils.py
  visualization/
    not_gate_plots.py
    figure_plots.py
  tests/
    test_not_gate_equations.py
  outputs/
    generated CSVs and PNGs
```

## File-To-Figure Map

| Output | Source model | Plotter |
| --- | --- | --- |
| `fig2_virtual_temperature_regimes.png` | `src/virtual_qubit.py` | `visualization/figure_plots.py` |
| `fig3b_not_transfer_curve.png` | `src/not_gate.py` | `visualization/figure_plots.py` |
| `fig3c_error_dissipation_tradeoff.png` | `src/tradeoff.py` | `visualization/figure_plots.py` |
| `fig6_nor_response_surface.png` | `src/logic_gates.py` | `visualization/figure_plots.py` |
| `fig7_majority_response_slices.png` | `src/logic_gates.py` | `visualization/figure_plots.py` |
| `fig8_xor_network_response.png` | `src/logic_gates.py` | `visualization/figure_plots.py` |

## Run

From the repository root, use the repo virtual environment:

```bash
.venv/bin/python Scripts/baseline_paper_reconstruction/main.py
```

This regenerates all CSVs and plots in:

```text
Scripts/baseline_paper_reconstruction/outputs/
```

Run tests:

```bash
.venv/bin/python -m unittest discover -s Scripts/baseline_paper_reconstruction/tests
```

The tests check:

- Fermi occupation inverse consistency.
- Virtual temperature equations.
- Collector and modulator zero-current fixed points.
- NOR truth-table behavior.
- 3-majority truth-table behavior.
- XOR network truth-table behavior.

## Current Caveat

This is now a good **baseline analytical reconstruction**, not yet a complete microscopic simulation. The main missing physics upgrade is exact entropy production for Fig. 3C. After that, the next upgrade would be comparing this reset-model baseline against local/global GKSL simulations.

