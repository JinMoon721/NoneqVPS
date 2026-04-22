#!/bin/bash
#SBATCH -A m1864_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH -J BAOAB
#SBATCH -o BAOAB.%j.out
#SBATCH -e BAOAB.%j.err

set -eo pipefail

BASE="."
echo "JobID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_JOB_NODELIST}"
echo "Start: $(date)"

# --- environment ---
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load conda

set +u
conda activate maceles_env
set -u

# --- sanity checks ---
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# --- run ---
# Use srun so Slurm properly wires up GPUs/CPUs for the job step.
srun -n 4 \
     --gpus-per-task=1 \
     --gpu-bind=single:1 \
     --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --cpu-bind=cores \
     --exact \
     bash -c '
       R=${SLURM_PROCID}
       echo "rank ${R} host $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} SLURM_STEP_GPUS=${SLURM_STEP_GPUS}"

       # show the one physical GPU this rank can access (UUID/bus)
       nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader | sed "s/^/rank ${R} /"

       OUTDIR=run_${R}
       python metaDynamics.py --seed $((100+R)) --outdir ${OUTDIR}
     '
echo "End: $(date)"
