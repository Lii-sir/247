"""Model-free blocking task used to test pending shared-memory leases."""
import time
from multiprocessing.shared_memory import SharedMemory


def hold(name, index):
    memory = SharedMemory(name=name)
    try:
        memory.buf[index] = 1
        deadline = time.monotonic()+15
        while not memory.buf[2]:
            if time.monotonic() >= deadline:
                raise TimeoutError("synthetic process gate was not released")
            time.sleep(.01)
    finally:
        memory.close()
