#!/bin/bash

#SBATCH --account=lr_statmech
#SBATCH --job-name=TPS
#SBATCH --partition=lr7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --qos=lr_lowprio
#SBATCH --time=24:00:00
#SBATCH --output="/global/scratch/users/seokjinmoon/logs/%j.log"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

source ~/.bashrc
conda activate ase_env

for i in $(seq 0 29); do
    mkdir -p ../data/TPS/worker${i}

    srun --exclusive -N1 -n1 -c1 \
        bash -c "cd ../data/TPS/worker${i} && python ../../../examples/runTPS.py --worker_id ${i}" &
done
wait

