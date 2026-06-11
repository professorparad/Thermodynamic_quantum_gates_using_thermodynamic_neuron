print("testing qutip import...")
from qutip import qeye, destroy, tensor
print("qutip imported OK")
I = qeye(2)
sm = destroy(2)
print("operators created OK")
