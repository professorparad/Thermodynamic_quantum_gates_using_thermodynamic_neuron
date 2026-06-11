import sys
sys.path.insert(0, 'Scripts')
print("step 1: importing config.parametres")
from Thermodynamic_nueron.config.parametres import e0
print("step 2: importing operators")
from Thermodynamic_nueron.collector.operators import I, sm
print("step 3: importing hamiltonian")
from Thermodynamic_nueron.collector.hamiltonian import H
print("ALL DONE")
