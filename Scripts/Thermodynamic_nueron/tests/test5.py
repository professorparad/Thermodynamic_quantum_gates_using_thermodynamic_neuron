import sys
sys.stdout.flush()

print("Attempting qutip import with timeout...", flush=True)
import threading
import time

result = [None]
exception = [None]
done = threading.Event()

def import_qutip():
    try:
        import qutip
        result[0] = (qutip.__version__, qutip.__file__)
    except Exception as e:
        exception[0] = e
    done.set()

t = threading.Thread(target=import_qutip, daemon=True)
t.start()
t.join(timeout=20)

if done.is_set():
    if result[0]:
        print(f"qutip {result[0][0]} imported from {result[0][1]}", flush=True)
    if exception[0]:
        print(f"qutip import error: {exception[0]}", flush=True)
else:
    print("qutip import TIMED OUT after 20s", flush=True)
