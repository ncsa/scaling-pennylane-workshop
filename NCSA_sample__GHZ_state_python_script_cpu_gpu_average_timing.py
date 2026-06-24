import pennylane as qp
import numpy as np
import time

## GHZ state exercise

def ghz_ops(n):
    """Apply the GHZ-preparation gates on wires 0..n-1."""
    qp.Hadamard(wires=0)
    for i in range(n - 1):
        qp.CNOT(wires=[i, i + 1])


def time_circuit(circuit_builder, device_name, n, repeat=3):
    """Average wall time (s) over `repeat` runs for a circuit on a device.
    Returns None if the device is unavailable.

    circuit_builder(dev, n) must return a callable QNode taking no arguments.
    """
    try:
        dev = qp.device(device_name, wires=n)
    except Exception:
        return None                            # e.g. lightning.gpu absent on a CPU box
    circuit = circuit_builder(dev, n)
    circuit()                                  # warm-up: exclude allocation / kernel init
    total = 0.0
    for _ in range(repeat):
        t0 = time.perf_counter()
        circuit()
        total += time.perf_counter() - t0
    return total / repeat

def build_ghz(dev, n):
    @qp.qnode(dev)
    def circuit():
        ghz_ops(n)
        return qp.expval(qp.PauliZ(0) @ qp.PauliZ(n - 1))
    return circuit


GHZ_SIZES = [5, 10, 15, 20, 25]
ghz_cpu, ghz_gpu = [], []

print(f"{'qubits':>7}{'CPU (s)':>12}{'GPU (s)':>12}{'GPU speedup':>14}")
print("-" * 45)
for n in GHZ_SIZES:
    cpu = time_circuit(build_ghz, "lightning.qubit", n)
    gpu = time_circuit(build_ghz, "lightning.gpu", n)
    ghz_cpu.append(cpu)
    ghz_gpu.append(gpu)
    gpu_str = f"{gpu:.5f}" if gpu is not None else "N/A"
    speed = f"{cpu/gpu:.2f}x" if (gpu is not None and gpu > 0) else "--"
    print(f"{n:>7}{cpu:>12.5f}{gpu_str:>12}{speed:>14}")

if all(g is None for g in ghz_gpu):
    print("\n(No lightning.gpu device found -> GPU column is N/A. "
          "Re-run on a GPU node to populate it.)")
