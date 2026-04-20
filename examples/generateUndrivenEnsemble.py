from __future__ import annotations
import glob
import os
import re
import random
from pathlib import Path
import multiprocessing as mp
import torch
from ase import units
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VPS_pkg.config import SimConfig
from VPS_pkg.model.committor_net import CommittorNet
from VPS_pkg.train.checkpoint import load_latest_checkpoint, save_checkpoint
from VPS_pkg.train.trainer import train_on_one_npz
from VPS_pkg.data.initial import get_initial_snapshots
from VPS_pkg.model.biased import make_bias_force_fn
from VPS_pkg.physics.integrator import UnderdampedLangevinBAOAB
from VPS_pkg.data.sampling import attach_calc
from VPS_pkg.physics.trajectory import distance



def log_radial_force_info(atoms, dyn, i=0, j=1):
    # current distance
    r = distance(atoms, i, j)

    # MIC displacement and radial unit vector
    dr = atoms.positions[j] - atoms.positions[i]
    dr_mic, _ = find_mic(dr, atoms.cell, atoms.pbc)
    rhat = dr_mic / np.linalg.norm(dr_mic)

    # physical force
    F_phys = atoms.get_forces()

    # bias force
    tau = dyn.step_count / dyn.bias_total_steps
    F_bias = dyn.bias_force_fn(atoms, tau)

    # relative forces
    Frel_phys = F_phys[j] - F_phys[i]
    Frel_bias = F_bias[j] - F_bias[i]

    # radial projections
    Fr_phys = np.dot(Frel_phys, rhat)
    Fr_bias = np.dot(Frel_bias, rhat)

    def tendency(x):
        if x > 0:
            return "increase r"
        elif x < 0:
            return "decrease r"
        else:
            return "neutral"

    print(
        f"time {dyn.step_count/dyn.bias_total_steps * 0.5:4f} | "
        f"r = {r:8.3f} | "
        f"Fr_phys = {Fr_phys: .4e} ({tendency(Fr_phys)}) | "
        f"Fr_bias = {Fr_bias: .4e} ({tendency(Fr_bias)})"
        f"ratio = {Fr_bias/Fr_phys:.4f}"
    )

def main():
    config = SimConfig (
        L=200.0, # A
        sig=2.0, # A
        eps=0.01 * units.kcal / units.mol,
        seed=1,
        tempK=300.0, ## K
        dtASE=10.0, ## fs
        masses=np.array([35, 35], dtype=float),
        gammaASE=np.array([35, 35], dtype=float) / (150.0 * units.fs),
        rin=22.3,
        rout=70.0,
        dielec=25.0,
        )
    device = "cpu"

    init_atoms_list = get_initial_snapshots(
            n_samples = 100,
            config = config,
            eq_dir = "../data/equilibriumSamples",
            save_frequency=10000,
            production_fraction=0.8,
            rng_seed=1,
            )

    print(f"Total {len(init_atoms_list)} initial snapshots are prepared")

    files = glob.glob("../data/unbiasedTraj/unbiased*.npz")
    startidx=-1
    if files:
        indices=[]
        for f in files:
            name = os.path.basename(f)
            m = re.match(r"unbiased(\d+)\.npz", name)
            if m:
                indices.append(int(m.group(1)))

        startidx = max(indices) if indices else None
        print(f"Detected last generate file unbiased{startidx}.npz")
    else:
        ubpath = Path("../data/unbiasedTraj")
        ubpath.mkdir(parents=True, exist_ok = True)

    total_steps= 50000
    for i, atoms in enumerate(init_atoms_list, start=startidx+1):
        atoms = atoms.copy()
        attach_calc(atoms, config)

        rng = np.random.default_rng(100+i)

        dyn = UnderdampedLangevinBAOAB(
                atoms = atoms,
                timestep = config.dtASE * units.fs,
                temperature_K = config.tempK,
                gamma = config.gammaASE,
                rng = rng,
                )
        unbiased_file_name = Path(f"../data/unbiasedTraj/unbiased{i}")


        ## run dynamics, get pos_list and xi_list, save to npz
        pos_list = [atoms.get_positions().astype(np.float32).copy()]
        xi_list = []

        print(f"Start running trajectory {i+1} / {len(init_atoms_list)}")
        prevR = distance(atoms, 0, 1)
        active=False
        reactive=False
        for step in range(total_steps):
            dyn.run(1)
            xi_list.append(dyn.last_xi.astype(np.float32).copy())
            pos_list.append(atoms.get_positions().astype(np.float32).copy())
            currR = distance(atoms, 0, 1)
            if not active:
                if ( prevR > config.rout ) and ( currR <= config.rout):
                    active = True
            else:
                if (prevR > config.rin) and (currR <= config.rin):
                    reactive=True
                elif (prevR <= config.rout) and (currR > config.rout):
                    active=False
            prevR = currR

        positions = np.stack(pos_list, axis=0)
        noises = np.stack(xi_list, axis=0)
        out = {}
        out["positions"] = positions
        out["noises"] = noises
        np.savez(unbiased_file_name, **out)
        print(f"unbiased Trajectory {i} is reactive? {reactive}")
        print(f"unbiased Trajectory saved to {unbiased_file_name}")

       #print(f"start {i+1}'th trajectory")
       #freq=1000
       ##dyn.attach(lambda: log_force_norms(atoms, dyn), interval=freq)
       #dyn.attach(lambda: log_radial_force_info(atoms, dyn, i=0, j=1), interval=freq)
       #for step in range(0, total_steps, freq):
       #    dyn.run(freq)
       #    traj.write(atoms)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
