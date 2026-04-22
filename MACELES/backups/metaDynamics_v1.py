from __future__ import annotations
import glob, re, random, sys
import os, time, socket, warnings, json, pickle, argparse
warnings.filterwarnings("ignore", category=FutureWarning)
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from mace.calculators import MACECalculator
from ase.calculators.plumed import Plumed

from VPS_pkg.config import MDConfig, plumedConfig
from VPS_pkg.physics.integrator import UnderdampedLangevinBAOAB
from VPS_pkg.plumed.io import initialize_from_checkpoint, save_checkpoint
from VPS_pkg.plumed.setup import plumedSetup

# -----------------------------
# Helper
# -----------------------------
from functools import partial
def force_scalar_energy(atoms):
    res = atoms.calc.results
    if "energy" in res:
        res["energy"] = float(np.asarray(res["energy"]).reshape(-1)[0])
    if "free_energy" in res:
        res["free_energy"] = float(np.asarray(res["free_energy"]).reshape(-1)[0])

def atomic_write_json(path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)   # atomic on POSIX


def main():
# -----------------------------
# CLI
# -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
# -----------------------------
# User parameters
# -----------------------------
    config = MDConfig (
        interval_log = 100,
        interval_dump = 100,
        interval_ckpt = 100,
        nsteps=1000,
        max_force_stop= 200.0, ## ev/A
        tempK = 300.0,
        dtASE = 1.0,
    ###    friction = 150.0, ## fs
        friction = 1.0 / (1000 * units.fs),
    )

    MODEL = "/pscratch/sd/s/seokjin/trainingMACELES/run/model_2/acetic_S3_compiled.model"

    plumed_config = plumedConfig (
        lam= 8.0,
        pace = 10,
        height = 0.25,
        sigma = 0.4,
        biasfactor = 10,
        stride = 100,
        tempK = config.tempK,
        lower = -2.0, ## lower harmonic bound loc, A
        upper = 8.5,  ## upper harmonic bound loc, A
        kappa = 100,  ## harmonic bound spring const kcal/molA
    )

# -----------------------------
# Logging
# -----------------------------
    print("=" * 70)
    print("ASE + MACE/LES Metadynamics Run 300K")
    print("=" * 70)

    start_datetime = datetime.now()
    t0 = time.perf_counter()
    print("Start time          :", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    print("SLURM_JOB_ID        :", os.environ.get("SLURM_JOB_ID"))
    print("SLURM_JOB_NAME      :", os.environ.get("SLURM_JOB_NAME"))
    print("SLURM_PROCID        :", os.environ.get("SLURM_PROCID"))
    print("SLURM_LOCALID       :", os.environ.get("SLURM_LOCALID"))
    print("Running on node     :", socket.gethostname())
    print("CPU cores allocated :", os.environ.get("SLURM_CPUS_PER_TASK"))

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Torch version       :", torch.__version__)
    print("Torch CUDA version  :", torch.version.cuda)
    print("CUDA available      :", torch.cuda.is_available())

    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        print("GPU device          :", gpu_name)
        print(f"GPU total memory    : {total_mem:.1f} GB")
    else:
        print("WARNING: CUDA not available!")

    print("=" * 70)


# -----------------------------
# Paths
# -----------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    master_hills = outdir / "HILLS_MASTER"
    master_colvar = outdir / "COLVAR_MASTER"
    state_json = outdir / "run_state.json"

    initial_file = "/global/homes/s/seokjin/MACELES/equilibrate/run_2/r2t200ps.traj"

# -----------------------------
# Main Loop
# -----------------------------
    ## get one snapshot first
    eq_atoms = read(initial_file, -1)
    if eq_atoms is None:
        raise ValueError(f"[stop] Could not read from {initial_file}")

    ## get latest snapshot and metadata
    atoms, segment_id, global_step, rng = initialize_from_checkpoint(ckpt_dir)    
    ## generate one if no restart file detected
    if atoms is None:
        atoms = eq_atoms
        rng = np.random.default_rng(args.seed)
        segment_id=0
        global_step=0

    ## get plumed setup, using index from initial_file(equilibrium config)

    plumed_lines, seg_hills, seg_colvar = plumedSetup(eq_atoms, outdir, segment_id, plumed_config)

    ## check velocity    
    if atoms.get_velocities() is None:
        print("Initial Velocity Not Found")
        MaxwellBoltzmannDistribution(atoms, temperature_K=config.tempK)
        Stationary(atoms)   # remove COM translation
        ZeroRotation(atoms) # remove overall rotation

    ## attach calculator
    atoms.calc = Plumed(
            calc = MACECalculator(model_paths = MODEL, device="cuda"),
            input=plumed_lines,
            atoms=atoms,
            timestep=config.dtASE * units.fs,
            kT=config.tempK * units.kB,
            log = str(outdir / f"plumed_seg{segment_id:04d}.log"),
            )

    ## attach dynamics
    dyn = Langevin(atoms, config.dtASE * units.fs, temperature_K=config.tempK, friction = config.friction, rng=rng)
    traj = Trajectory( outdir / f"run_seg{segment_id:04d}.traj", "w", atoms)
    logger = MDLogger(dyn, atoms, outdir / f"md_seg{segment_id:04d}.log", header=True, stress=False, peratom=False, mode="w")

    dyn.attach(traj.write, interval=config.interval_dump)
    dyn.attach(partial(force_scalar_energy, atoms), interval=config.interval_log)
    dyn.attach(logger, interval=config.interval_log)

    ## run dynamics
    local_step = 0
    initial_global_step = global_step

    save_checkpoint(atoms, rng, segment_id, global_step, local_step, "segment_start", ckpt_dir)

    meta = {
        "segment_id": segment_id,
        "initial_step": initial_global_step,
        "last_step": initial_global_step,
        "pace": plumed_config.pace,
        "dt_fs": config.dtASE,
        "status": "running"
    }
    meta_path = outdir / f"meta_seg{segment_id:04d}"
    atomic_write_json(meta_path, meta)

    while global_step < config.nsteps:
        try:
            dyn.run(1)
            local_step += 1
            global_step +=1
            forces = atoms.calc.results.get("forces", None)
            if forces is None:
                forces = atoms.get_forces()
            maxF = float(np.linalg.norm(forces, axis=1).max())
 
            if global_step % config.interval_ckpt == 0:
                save_checkpoint(atoms, rng, segment_id, global_step, local_step, "running", ckpt_dir)
            if maxF > config.max_force_stop:
                print(f"[STOP] segment={segment_id} global_step={global_step} maxF={maxF:.2f} eV/Å", flush=True)
                write(outdir / f"bad_seg{segment_id:04d}_step{global_step:09d}.extxyz", atoms)
                save_checkpoint(atoms, rng, segment_id, global_step, local_step, "stopped_maxforce", ckpt_dir)
            if global_step % plumed_config.pace == 0:
                meta["last_step"] = global_step
                atomic_write_json(meta_path, meta)
        except Exception as e:
            print(f"[CRASH] segment={segment_id} global_step={global_step} exception={repr(e)}", flush=True)
            try:
                write(outdir / f"crash_seg{segment_id:04d}_step{global_step:09d}.extxyz", atoms)
            except Exception as e2:
                print(f"[WARN] could not write crash snapshot: {repr(e2)}", flush=True)
            save_checkpoint(atoms, rng, segment_id, global_step, local_step, "crashed", ckpt_dir)
#    save_checkpoint(atoms, rng, segment_id, global_step, local_step, "segment_complete", ckpt_dir)

    processed_steps =  global_step - initial_global_step

# -----------------------------
# End program with timelogging
# -----------------------------
    t1 = time.perf_counter()
    end_datetime = datetime.now()

    print("=" * 70)
    print("End time            :", end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total walltime      : {t1 - t0:.2f} seconds")
    print("=" * 70)

    simulatedPs = processed_steps * config.dtASE/units.fs /1000.0
    seconds = t1-t0
    print(f"Simulated time      : {simulatedPs:.3f} ps")
    print(f"Performance         : {simulatedPs / seconds * 86400:.3f} ps/day")
    print(f"Performance         : {simulatedPs / seconds * 86400/ 24:.3f} ps/hours")


if __name__ == "__main__":
    main()
