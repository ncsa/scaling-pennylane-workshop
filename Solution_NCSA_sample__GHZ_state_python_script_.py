import pennylane as qp
import numpy as np

#set the number of qubits
N_QUBITS = 15
# Flip this to True on a machine with an MPI/cuQuantum-enabled lightning.gpu build.
USE_GPU = False

DEVICE_NAME = "lightning.gpu" if USE_GPU else "lightning.qubit"
print(f"Using device: {DEVICE_NAME}  with {N_QUBITS} qubits")



## GHZ state exercise

def ghz_ops(n):
    """Apply the GHZ-preparation gates on wires 0..n-1."""
    qp.Hadamard(wires=0)
    for i in range(n - 1):
        qp.CNOT(wires=[i, i + 1])

# Draw the small version so the structure is legible.
dev_small = qp.device(DEVICE_NAME, wires = 5)

@qp.qnode(dev_small)
def ghz_small(n):
    ghz_ops(n)
    return qp.state()

print(qp.draw(ghz_small)(5))


######

def make_device(n):
    """Single place that builds the device, so CPU/GPU is a one-line switch."""
    return qp.device(DEVICE_NAME, wires=n)


dev = make_device(N_QUBITS)

@qp.qnode(dev)
def ghz_sample():
    ghz_ops(N_QUBITS)
    return qp.sample()

# Set the total number of shots
N_SHOTS = 1000
ghz_sampled = qp.set_shots(ghz_sample, shots=N_SHOTS)

samples = ghz_sampled()                       # shape (N_SHOTS, N_QUBITS)
# Convert each bitstring row to an integer for tallying.
weights = 1 << np.arange(N_QUBITS)[::-1]
ints = samples.dot(weights)
values, counts = np.unique(ints, return_counts=True)

print(f"Distinct measurement outcomes: {len(values)} (a GHZ state must give exactly 2)")
for v, c in zip(values, counts):
    print(f"  |{format(v, f'0{N_QUBITS}b')}>  ->  {c} shots  ({100*c/N_SHOTS:.1f}%)")



######################

@qp.qnode(dev)
def ghz_correlation():
    ghz_ops(N_QUBITS)
    return qp.expval(qp.PauliZ(0) @ qp.PauliZ(N_QUBITS - 1))

corr = ghz_correlation()
print(f"<Z_0 Z_{N_QUBITS-1}> = {corr:.6f}   (ideal GHZ value = 1.0)")