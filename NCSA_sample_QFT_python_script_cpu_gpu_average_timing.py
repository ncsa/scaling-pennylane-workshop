import pennylane as qp
import time

def time_qft(device_name, n, repeat=3):
    """Average wall time (s) over `repeat` runs for an n-qubit QFT.
    Returns None if the device is unavailable."""
    try:
        dev = qp.device(device_name, wires=n)
    except Exception as exc:
        # lightning.gpu is absent on non-GPU machines -> skip gracefully.
        return None

    @qp.qnode(dev)
    def circuit():
        for w in range(n):
            qp.Hadamard(wires=w)
        qp.QFT(wires=range(n))
        return qp.expval(qp.PauliZ(0))

    circuit()                              # warm-up: exclude allocation / kernel init
    total = 0.0
    for _ in range(repeat):
        t0 = time.perf_counter()
        circuit()
        total += time.perf_counter() - t0
    return total / repeat


#QFT_SIZES = [5, 10,15,20,25]
QFT_SIZES = [5, 10,15]


cpu_times, gpu_times = [], []

print(f"{'qubits':>7}{'CPU (s)':>12}{'GPU (s)':>12}{'GPU speedup':>14}")
print("-" * 45)
for n in QFT_SIZES:
    cpu = time_qft("lightning.qubit", n)
    gpu = time_qft("lightning.gpu", n)        # None if no GPU / plugin
    cpu_times.append(cpu)
    gpu_times.append(gpu)
    gpu_str = f"{gpu:.5f}" if gpu is not None else "N/A"
    speed = f"{cpu/gpu:.2f}x" if (gpu is not None and gpu > 0) else "--"
    print(f"{n:>7}{cpu:>12.5f}{gpu_str:>12}{speed:>14}")

if all(g is None for g in gpu_times):
    print("\n(No lightning.gpu device found -> GPU column is N/A. "
          "Re-run on a GPU node to populate it.)")
