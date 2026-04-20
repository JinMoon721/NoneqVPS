from __future__ import annotations
import glob
import random
from pathlib import Path
import multiprocessing as mp
import torch
from ase import units
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from VPS_pkg.config import SimConfig
from VPS_pkg.model.committor_net import CommittorNet
from VPS_pkg.train.checkpoint import load_latest_checkpoint, save_checkpoint


K_COUL = 14.3996454784255
kB_eV_per_K = 8.617333262145e-5  # eV / K

def potential(r, config):
    sr6 = (config.sig / r)**6
    return 4.0 * config.eps * (sr6**2 - sr6) - K_COUL / config.dielec/r

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

    model = CommittorNet(tempK=config.tempK, in_dim=2, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)

    model_dir = Path("../data/models")
    start_epoch = load_latest_checkpoint(model, optimizer, model_dir, device=device)
    model.eval()

    n_t = 500
    n_r = 200

    t_vals = np.linspace(0.0, 1.0, n_t)
    r_vals = np.linspace(1.5, 200.0, n_r) / 80

    T, R = np.meshgrid(t_vals, r_vals)

    X = np.column_stack([R.ravel(), T.ravel()]) # (n_r * n_t , 2)

    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    kBT = kB_eV_per_K * config.tempK
    with torch.no_grad():
        _, logq= model.forward_from_features(x_tensor)
        logq = logq.squeeze(-1).cpu().numpy()
    dpot = -2. * kBT * logq.reshape(n_r, n_t) # doob potential eV
    dpot /= units.kcal / units.mol

    n_slices = 500
    t_indices = np.linspace(0, n_t-1, n_slices, dtype=int)

#   plt.figure(figsize=(7, 5))
#   cmap = plt.cm.viridis
#   colors = cmap(np.linspace(0, 1, n_slices))

    y_t = []
    for i, idx in enumerate(t_indices):
        t_val = t_vals[idx]
        t_ns = 0.5 * t_val
        x = r_vals*80
        loc = np.abs(x - 70).argmin()
        q_slice = dpot[:,idx] - dpot[loc, idx] 
        y = potential(x, config) / (units.kcal / units.mol)
        y_t.append(y + q_slice)
        #plt.plot(x, q_slice + y, color=colors[i], label=f"t={t_ns:.2f} ns")

    T = len(y_t)
    y_t = np.asarray(y_t)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    title = ax.set_title("")

    ax.set_xlim(2, 150)
    ax.set_ylim(-3, 2)

    def update(frame):
        line.set_data(x, y_t[frame])
        time = frame/500*0.5
        title.set_text(f"time = {time} ns")
        return line, title

    ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False)
    plt.plot(x, potential(x, config) / units.kcal * units.mol, 'k--')
    plt.show()

    #plt.plot(x, y, '--')

   #plt.xlabel("r")
   #plt.ylabel(r"$-2 k_B T \ln q(r, t) \ / \ kcal mol^{-1}$")
   #plt.tight_layout()
   #plt.legend()
   #plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
