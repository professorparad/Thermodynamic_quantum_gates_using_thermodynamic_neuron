# Thermodynamic Floquet Gate Prototype

This folder is the next step after the baseline reconstruction and the standalone
Floquet-buffer bridge.

The aim is not to pretend that the full non-Markovian three-qubit gate is finished.
Instead, this prototype connects the validated pieces:

```text
baseline thermodynamic NOT response
+ Floquet-buffer distinguishability gain
+ convergence/decision diagnostics
= first gate-attempt design table
```

## What This Prototype Answers

Before running a very expensive full gate, we ask:

```text
Which baseline NOT settings are sharp enough, and does the buffer currently
improve logical distinguishability enough to justify embedding it in the gate?
```

The output is a design CSV and report:

```text
outputs/not_floquet_design_table.csv
outputs/not_floquet_design_report.txt
```

## Programming Caveat

Do not use this folder as a final thermodynamic proof.  It is a design/decision
layer.  A final heat/work/entropy-production claim for a driven structured bath
requires a separate audit of interaction energy, memory, and drive work.

