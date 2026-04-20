from __future__ import annotations
import random
from pathlib import Path
from ase import units
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.geometry import find_mic
import numpy as np

import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VPS_pkg.config import SimConfig
from VPS_pkg.data.tps import tps_shoot_once

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type = int, required= True)
    args=parser.parse_args()

    worker_id = args.worker_id

    naccept_target=50
    nattempt_max= 500
    sigma_v= 0.02
    max_branch_steps = 50000 ## 500 ps = linear regime
    shooting_margin = 5
    buffer_size = 30 ## save 30 initial positions and velocities for next TPS move
    target_num_steps = 50000 + 1
    accepted=0
    ## make directories for outputs
    Path("./meta").mkdir(parents=True, exist_ok=True)
    Path("./initialSamples").mkdir(parents=True, exist_ok=True)
    Path("./trainData").mkdir(parents=True, exist_ok=True)
    ## select {buffer_size} snapshots for initial TPS move, saved with steps forward and
    ##backward
    savepaths = Path("./initialSamples")
    if savepaths:
        accepted = max(
                (int(p.stem.replace("reactive", "")) for p in savepaths.glob("reactive*.traj")),
                default=0
                )
    else:
        print(f"Start initial TPS start sample preparation")
        original_file = f"../original.traj"
        path = read(original_file, index=":")
        total_steps = len(path)
        rng = np.random.default_rng(worker_id)
        indices = rng.choice(np.arange(shooting_margin, total_steps-shooting_margin), size=buffer_size, replace=False)
        indices.sort()
        samples=[]
        for k in indices:
            atoms = path[k]
            ## draw a random number to make the total steps = target total steps
            sb=random.randint(k+ shooting_margin, k+ target_num_steps - total_steps - shooting_margin) # [a, b]
            atoms.info["steps_forward"] = target_num_steps -1- sb
            atoms.info["steps_backward"] = sb
            samples.append(atoms)
        
        original_output = f"./initialSamples/reactive{accepted}.traj"
        with Trajectory(original_output, "w") as traj_out:
            for atoms in samples:
                traj_out.write(atoms)
        
        print(f"Initial TPS start sample preparation done at {original_output}")

    for attempt in range(nattempt_max):
        print(f"Starting attempt {attempt} TPS")
        current_file = f"./initialSamples/reactive{accepted}.traj"
        next_file    = f"./initialSamples/reactive{accepted+1}.traj"
        meta_dir=f"./meta/"
        train_file_name = f"./trainData/train{accepted+1}"

        ok = tps_shoot_once(
                current_traj_file = current_file,
                next_traj_file = next_file,
                meta_file_dir = meta_dir,
                train_file_name = train_file_name,
                config = config,
                accepted=accepted,
                sigma_v = sigma_v,
                shooting_margin = shooting_margin,
                buffer_size = buffer_size, 
                total_steps = target_num_steps,
                seed= (worker_id+20)*100 + attempt,
                )

        if ok:
            current_file = next_file
            accepted+=1
            if accepted >= naccept_target:
                break
    print(f"TPS done: accepted {accepted} trajectories.")

if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    main()
