import pennylane as qp
import numpy as np

#set the number of qubits
N_QUBITS = 15
# Flip this to True on a device with Nvidia GPU
USE_GPU = False

DEVICE_NAME = "lightning.gpu" if USE_GPU else "lightning.qubit"
print(f"Using device: {DEVICE_NAME}  with {N_QUBITS} qubits")

##########################################################################################################################################################

## GHZ state exercise

def ghz_ops(n):
    """Apply the GHZ-preparation gates on wires 0..n-1."""
    #INSERT CODE HERE
    

# Draw the small version so the structure is legible.
#INSERT CODE HERE
dev_small = None

#Create a function that creates the GHZ state and link it to the device with a QNode
#INSERT CODE HERE

def ghz_small(n):
    #INSERT CODE HERE
    return None

#Draw the circuit
print(qp.draw(ghz_small)(5))


######################################################################################################################################################

#Create a function to create a device
#INSERT CODE HERE
def make_device(n):
    """Single place that builds the device, so CPU/GPU is a one-line switch."""
    return None

#call the function above creating a device for N_QUBITS
#INSERT CODE HERE
dev = None

#Create a quantum circuit that creates a GHZ state and draws samples of states
#INSERT CODE HERE
@qp.qnode(dev)
def ghz_sample():
    #INSERT CODE HERE
    
    return None

# Set the total number of shots
#INSERT CODE HERE
N_SHOTS = 1000
ghz_sampled = None

samples = ghz_sampled()                       # shape (N_SHOTS, N_QUBITS)
# Convert each bitstring row to an integer for tallying.
weights = 1 << np.arange(N_QUBITS)[::-1]
ints = samples.dot(weights)
values, counts = np.unique(ints, return_counts=True)

print(f"Distinct measurement outcomes: {len(values)} (a GHZ state must give exactly 2)")
for v, c in zip(values, counts):
    print(f"  |{format(v, f'0{N_QUBITS}b')}>  ->  {c} shots  ({100*c/N_SHOTS:.1f}%)")



####################################################################################################################################################

#INSERT CODE HERE
def ghz_correlation():
    #INSERT CODE HERE

    return None

corr = ghz_correlation()
print(f"<Z_0 Z_{N_QUBITS-1}> = {corr:.6f}   (ideal GHZ value = 1.0)")