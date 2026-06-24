import pennylane as qp
import numpy as np
import time



#set the number of qubits
N_QUBITS = 10
# Flip this to True on a device with Nvidia GPU
USE_GPU = False

DEVICE_NAME = "lightning.gpu" if USE_GPU else "lightning.qubit"
print(f"Using device: {DEVICE_NAME}  with {N_QUBITS} qubits")



##############################################################################################
#Create a function to create a device
#INSERT CODE HERE
def make_device(n):
    """Single place that builds the device, so CPU/GPU is a one-line switch."""
    return None

#Create a device with N_QUBITS
dev = make_device(N_QUBITS)

# Encode an integer x into the first few qubits as a basis state.
x_bits = np.zeros(N_QUBITS, dtype=int)
x_bits[[0, 2, 5]] = 1            # arbitrary nonzero pattern

#Create a a quantum circuit that encodes the integer 'x' defined above (not as an input to the circuit) and takes its QFT, returning the marginal probability over 3 qubits/wires. 
#Link the circuit to the device above
#INSERT CODE HERE

def qft_probs_first3():
    #INSERT CODE HERE
    
    return None   # marginal over 3 wires -> 8 outcomes

probs3 = qft_probs_first3()
print("Marginal probabilities over the first 3 qubits after QFT:")
for i, p in enumerate(probs3):
    print(f"  {format(i, '03b')}: {p:.5f}")
print(f"\nUniform reference value 1/8 = {1/8:.5f}  -> QFT spreads amplitude evenly.")


##########################################################################################################

#Create a quantum circuit that encodes the integer 'x' takes it QFT and the inverse QFT and show that you recover the same state
#Link the circuit to a device using a QNODE
#INSERT CODE HERE

def qft_roundtrip():
    #INSERT CODE HERE

    return None    # look at first 8 wires of the recovered state

probs = qft_roundtrip()
recovered = int(np.argmax(probs))
expected = int("".join(map(str, x_bits[:8])), 2)
print(f"Recovered first-8-qubit pattern: {format(recovered, '08b')}")
print(f"Expected (input):                {format(expected, '08b')}")
print(f"Probability of recovered state:  {probs[recovered]:.6f}  (should be 1.0)")
assert recovered == expected and abs(probs[recovered] - 1.0) < 1e-6
print("PASS: QFT followed by inverse-QFT returns the input state.")