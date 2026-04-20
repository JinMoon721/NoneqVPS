from __future__ import annotations
import glob
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
from VPS_pkg.data.bruteForce import generate_original_TPS

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

    random_seed = 1111

    output_path = Path("../data/TPS/original.traj")
    dirpath = output_path.parent
    dirpath.mkdir(parents=True, exist_ok=True)


    max_steps = 1000000
    generate_original_TPS(
            config = config,
            max_steps = max_steps,
            target_steps = 50000,
            rin = config.rin,
            rout = config.rout,
            rng_seed = random_seed,
            output_traj = output_path,
            ) ## -> ../data/TPS/original.traj


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
