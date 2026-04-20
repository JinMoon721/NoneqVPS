import numpy as np
import glob
import random
import os

def merge_npz_files(files, out_path):
    out = {}
    out["num_samples"] = len(files)

    for i, f in enumerate(files):
        data = np.load(f)

        # assume each file has single sample keys
        out[f"pos_fwd_{i}"] = data["pos_fwd"]
        out[f"xi_fwd_{i}"] = data["xi_fwd"]
        out[f"pos_bwd_{i}"] = data["pos_bwd"]
        out[f"xi_bwd_{i}"] = data["xi_bwd"]
    np.savez(out_path, **out)

all_files = sorted(glob.glob("../data/TPS/worker*/trainData/*.npz"))

exclude = {"train0.npz"}

files = [f for f in all_files if os.path.basename(f) not in exclude ]

random.shuffle(files)
print(f"Detected {len(files)} training reactive trajectories")
batch_size = 100
print(f"saving {batch_size} trajectires in batch")

for k in range(0, len(files), batch_size):
    chunk = files[k:k+batch_size]
    out_path = f"../data/validationSet/batch{k//batch_size}.npz"

    merge_npz_files(chunk, out_path)
