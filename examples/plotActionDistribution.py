from __future__ import annotations
import glob
import random
from pathlib import Path
import torch
from ase import units
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VPS_pkg.config import SimConfig
from VPS_pkg.model.committor_net import CommittorNet
from VPS_pkg.train.trainer import compute_on_one_npz


def sample_cumulants(x, max_order=6):
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    dx = x- mu

    centered ={}
    for n in range(2, max_order+1):
        centered[n] = np.mean(dx**n)

    kappa = {1: mu}
    if max_order >= 2:
        kappa[2] = centered[2]
    if max_order >= 3:
        kappa[3] = centered[3]
    if max_order >= 4:
        kappa[4] = centered[4] - 3 * centered[2]**2
    if max_order >= 5:
        kappa[5] = centered[5] - 10 * centered[3] * centered[2]
    if max_order >= 6:
        kappa[6] = (
                centered[6]
                - 15 * centered[4] * centered[2]
                - 10 * centered[3]**2
                + 30 * centered[2]**3
                )
    return kappa

def cumulant_expansion_exp_minus_x(x, order=4):
    """
    compute truncated cumulant expansion of - log<exp(-x)>
    """
    kappa = sample_cumulants(x, max_order=order)

    val = 0.0
    for n in range(1, order+1):
        val += ((-1)**(n+1)) * kappa[n] / math.factorial(n)
    return val, kappa

def full_exp_average(x):
    x = np.asarray(x, dtype=float)
    a = np.min(x)
    return a- np.log(np.mean(np.exp(-(x-a))))

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

    model = CommittorNet(tempK = config.tempK, in_dim=2, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-3,
            weight_decay=1e-4,
            )

    torch.set_num_threads(56)
    torch.set_num_interop_threads(1)
    print("torch threads:", torch.get_num_threads())
    print("interop threads:", torch.get_num_interop_threads())

    # -------------------------------
    # restart from latest checkpoint if exists
    # --------------------------------
    model_dir = Path("../data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    latest_ckpt_path = None
    latest_ckpt_epoch = -1

    ckpt_files = sorted(model_dir.glob("*.pt"))
    if ckpt_files:
        for ckpt_path in ckpt_files:
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                ckpt_epoch = int(ckpt.get("epoch", -1))
                if ckpt_epoch > latest_ckpt_epoch:
                    latest_ckpt_epoch = ckpt_epoch
                    latest_ckpt_path = ckpt_path
            except Exception as e:
                print(f"Skip unreadable checkpoint {ckpt_path}: {e}")
    if latest_ckpt_path is not None:
        print(f"Found checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location="cpu")

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = int(ckpt["epoch"]) 
        print(f"Restart training from epoch {start_epoch}")
        if "loss" in ckpt:
            print(f"Previous save loss: {ckpt['loss']}")
    else:
        print("No saved checkpoint found")


    ## search pre-computed action data
    datadir = Path("../data/rawActions/")
    datadir.mkdir(parents=True, exist_ok=True)

    batch_size = 8
    files = sorted(datadir.glob("model*"))
    if files:
        filepath = files[-1]
        print(f"Loading existing data : {filepath}")
        losses = np.load(filepath)
    else:
        print("No existing data found, compute...")
        train_files = sorted(glob.glob("../data/validationSet/*.npz"))
        print(f"found {len(train_files)} files")
        random.shuffle(train_files)

        print(f"\n==== Measure relative action distribution from latest model====")
        file_losses=[]
        for path in train_files:
            print(f"Load {path} and start measurement")
            batch_loss = compute_on_one_npz(
                path = path,
                model = model,
                optimizer = optimizer,
                config = config,
                device=device,
                batch_size = batch_size,
                )
            file_losses.append(batch_loss) ## ndarray, (nfiles, batch_size)

        losses = np.concatenate(file_losses)

        outpath = Path(f"../data/rawActions/model{start_epoch}.npy")
        np.save(outpath, losses)


    x = losses
    x = x[x <= 10]
    x = x[x >= -10]

    full_val = full_exp_average(x) 
    for n in range(1, 7):
        approx, _ = cumulant_expansion_exp_minus_x(x, order=n)
        print(f"order {n}: {approx}")
    print("full:", full_val)

    loss_min = x.min()
    loss_max = x.max()

    bins = "auto"
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=bins, edgecolor="black")
    plt.xlabel("Loss")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
