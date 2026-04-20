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
        out[f"positions_{i}"] = data["positions"]
        out[f"noises_{i}"] = data["noises"]
    np.savez(out_path, **out)

files = sorted(glob.glob("../data/biasedTraj/biased*.npz"))
random.shuffle(files)
print(f"Detected {len(files)} biased trajectories")
batch_size = 20
print(f"saving {batch_size} trajectories in batch")


#files = sorted(glob.glob("../data/biasedTraj/batch*.npz"))
#data = np.load(files[0])
#print(data.files)
#for k in data.files:
#    print(k, data[k].shape, data[k].dtype, data[k].nbytes)

for k in range(0, len(files), batch_size):
    chunk = files[k:k+batch_size]
    out_path = f"../data/biasedTraj/batch{k//batch_size}.npz"
    merge_npz_files(chunk, out_path)
    print(len(chunk))
