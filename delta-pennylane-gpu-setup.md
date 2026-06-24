# Setting up a PennyLane + GPU virtual environment on NCSA Delta

This guide creates a Python virtual environment on the **NCSA Delta** cluster
with PennyLane and the NVIDIA GPU statevector simulator (`lightning.gpu`).

> **Package name note.**  The NVIDIA GPU device `lightning.gpu` is provided by **`pennylane-lightning-gpu`**, which
> also needs NVIDIA's **`custatevec-cu12`** (the cuQuantum statevector backend).
> `lightning.gpu` requires a CUDA GPU of compute capability >= 7.0 — on Delta
> that means the **A40 or A100** partitions (the `gpuMI100x8` partition is AMD
> and needs a different device).

Verified against PennyLane / Lightning **0.45.0** (current stable).

---

## 0. Key Delta facts this guide relies on

- The Python toolchain comes from the `miniforge3-python` module.
- After `module reset`, the environment variables `$WORK` and `$SCRATCH` are set.
  Build environments in `$WORK` (persistent project space), not your home dir.
- GPU allocations are charged to a "Project" account returned by the `accounts`
  command. You need a `-gpu` account to use GPU partitions.

If creating a fresh virtual environment

---

## 1. Load the Python module

```bash
module load miniforge3-python  # provides python / pip 
```

---

## 2. Create and activate the virtual environment

```bash
python -m venv /projects/bccu/dnino/ENV_NAME
source /projects/bccu/dnino/ENV_NAME/bin/activate
```

---

## 3. Install PennyLane + the GPU device

```bash
# Core PennyLane + CPU simulator (lightning.qubit) 
pip install pennylane pennylane-lightning  

# NVIDIA GPU device: the plugin AND the cuQuantum statevector backend
pip install pennylane-lightning-gpu
```

## 3.  Load the files

# Clone GitHub repo to your directory
```bash
git clone https://github.com/ncsa/scaling-pennylane-workshop
```
---

### Sources
- PennyLane install (current stable v0.45, CUDA multi-GPU simulator):
  https://pennylane.ai/install
- Lightning-GPU installation (cuQuantum / `custatevec-cu12`, `CUQUANTUM_SDK`):
  https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/installation.html
