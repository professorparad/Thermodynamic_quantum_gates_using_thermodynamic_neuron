import sys
print("Python:", sys.version)

print("Importing qutip...", flush=True)
import qutip
print("qutip imported, version:", qutip.__version__, flush=True)

print("Testing qeye...", flush=True)
I = qutip.qeye(2)
print("qeye OK", flush=True)

print("Testing destroy...", flush=True)
sm = qutip.destroy(2)
print("destroy OK", flush=True)
