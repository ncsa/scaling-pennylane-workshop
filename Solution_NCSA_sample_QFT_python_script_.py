import pennylane as qp
import numpy as np
import time



#set the number of qubits
N_QUBITS = 10
# Flip this to True on a machine with an MPI/cuQuantum-enabled lightning.gpu build.
USE_GPU = False

DEVICE_NAME = "lightning.gpu" if USE_GPU else "lightning.qubit"
print(f"Using device: {DEVICE_NAME}  with {N_QUBITS} qubits")



##############################################################################################
def make_device(n):
    """Single place that builds the device, so CPU/GPU is a one-line switch."""
    return qp.device(DEVICE_NAME, wires=n)

dev = make_device(N_QUBITS)

# Encode an integer x into the first few qubits as a basis state.
x_bits = np.zeros(N_QUBITS, dtype=int)
x_bits[[0, 2, 5]] = 1            # arbitrary nonzero pattern

@qp.qnode(dev)
def qft_probs_first3():
    qp.BasisState(x_bits, wires=range(N_QUBITS))
    qp.QFT(wires=range(N_QUBITS))
    return qp.probs(wires=[0, 1, 2])   # marginal over 3 wires -> 8 outcomes

probs3 = qft_probs_first3()
print("Marginal probabilities over the first 3 qubits after QFT:")
for i, p in enumerate(probs3):
    print(f"  {format(i, '03b')}: {p:.5f}")
print(f"\nUniform reference value 1/8 = {1/8:.5f}  -> QFT spreads amplitude evenly.")


##########################################################################################################

@qp.qnode(dev)
def qft_roundtrip():
    qp.BasisState(x_bits, wires=range(N_QUBITS))
    qp.QFT(wires=range(N_QUBITS))
    qp.adjoint(qp.QFT)(wires=range(N_QUBITS))
    return qp.probs(wires=range(8))     # look at first 8 wires of the recovered state

probs = qft_roundtrip()
recovered = int(np.argmax(probs))
expected = int("".join(map(str, x_bits[:8])), 2)
print(f"Recovered first-8-qubit pattern: {format(recovered, '08b')}")
print(f"Expected (input):                {format(expected, '08b')}")
print(f"Probability of recovered state:  {probs[recovered]:.6f}  (should be 1.0)")
assert recovered == expected and abs(probs[recovered] - 1.0) < 1e-6
print("PASS: QFT followed by inverse-QFT returns the input state.")