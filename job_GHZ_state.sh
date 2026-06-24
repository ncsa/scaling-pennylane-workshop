#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # <- CPU cores for the job; also sets OMP_NUM_THREADS below
#SBATCH --partition=gpuA40x4      # <- GPU partition; nodes here also provide CPU cores
#SBATCH --account=bccu-delta-gpu    # <- GPU allocation; confirm via the "accounts" command
#SBATCH --job-name=myjobtest
#SBATCH --time=00:10:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch&work"
#SBATCH -e slurm-serve-ghz_state%j.err
#SBATCH -o slurm-serve-ghz_state%j.out
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load miniforge3-python  # ... or any appropriate modules
module list  # job documentation and metadata
source /projects/bccu/dnino/venv-test-shared/bin/activate
which python
cd /projects/bccu/dnino
srun python NCSA_sample__GHZ_state_python_script.py