#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python Scripts/heom_structured_bath_validation/heom_floquet_phase_sweep.py
.venv/bin/python Scripts/heom_structured_bath_validation/phase_sweep_figures.py
