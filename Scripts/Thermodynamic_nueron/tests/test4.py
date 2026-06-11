import sys
import os
import traceback

print("Python:", sys.version)
print("CWD:", os.getcwd())
sys.stdout.flush()

# Test importing qutip piece by piece
print("\n--- Trying qutip import with diagnostic ---")
sys.stdout.flush()

try:
    import qutip
    print("qutip imported successfully")
    print("qutip version:", qutip.__version__)
    print("qutip file:", qutip.__file__)
    sys.stdout.flush()
    
    # Try operators
    from qutip import qeye, destroy, tensor, basis
    print("qeye/destroy/tensor/basis imported OK")
    sys.stdout.flush()
    
    # Try steadystate
    from qutip import steadystate
    print("steadystate imported OK")
    sys.stdout.flush()
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    sys.stdout.flush()
