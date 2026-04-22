import glob,pickle
from pathlib import Path
import numpy as np
from ase.io import read, write


def latest_checkpoint(ckpt_dir: Path):
    states = sorted(ckpt_dir.glob("state_seg*_step*.pkl"))
    if not states:
        return None
    return states[-1]

def parse_ckpt_name(path: Path):
    stem = path.stem
    parts = stem.split("_")
    seg = int(parts[1].replace("seg", ""))
    step = int(parts[2].replace("step", ""))
    return seg, step

def save_checkpoint(
        atoms,
        rng,
        segment_id: int,
        global_step: int,
        local_step: int,
        status: str,
        ckpt_dir: Path,
        ):
    tag = f"seg{segment_id:04d}_step{global_step:09d}"

    atoms_file = ckpt_dir / f"atoms_{tag}.traj"
    state_file = ckpt_dir / f"state_{tag}.pkl"

    write(atoms_file, atoms)

    state = {
            "segment_id": segment_id,
            "global_step": global_step,
            "local_step": local_step,
            "rng_state": rng.bit_generator.state,
            "status": status,
            }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)
    print(f"[CKPT] {tag}", flush=True)


def load_checkpoint(path_state: Path, ckpt_dir: Path):
    seg, step = parse_ckpt_name(path_state)
    atoms_path = ckpt_dir / path_state.name.replace("state_", "atoms_").replace(".pkl", ".traj")

    atoms = read(atoms_path, index=-1)

    with open(path_state, "rb") as f:
        state = pickle.load(f)

    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]

    return atoms, state, rng

def initialize_from_checkpoint(ckpt_dir: Path):
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is not None:
        atoms, state, rng = load_checkpoint(ckpt, ckpt_dir)
        print(f"[RESTART] loaded checkpoint {ckpt.name}", flush=True)
        return atoms, state["segment_id"]+1, state["global_step"], rng
    else:
        print("[INITIAL] restart files not found")
        return None, None, None, None
        
