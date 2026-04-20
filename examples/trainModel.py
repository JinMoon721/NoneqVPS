from __future__ import annotations
import glob
import random
from pathlib import Path
import torch
from ase import units
from ase.io.trajectory import Trajectory
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VPS_pkg.config import SimConfig
from VPS_pkg.model.committor_net import CommittorNet
from VPS_pkg.train.trainer import train_on_one_npz

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
    nepoch = 5000
    batch_size=4

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


    train_files = sorted(glob.glob("../data/trainSet/*.npz"))
    print(f"found {len(train_files)} files")
    random.shuffle(train_files)

    for epoch in range(start_epoch, nepoch):
        print(f"\n==== epoch {epoch+1}/{nepoch} ====")
        file_losses =[]
        for path in train_files:
            print(f"Load {path} and start training")
            avg_file_loss = train_on_one_npz(
                    path = path,
                    model = model,
                    optimizer = optimizer,
                    config = config,
                    device=device,
                    batch_size = batch_size,
                    )
            file_losses.append(avg_file_loss)
            print(f"Done: {path} loss: {avg_file_loss}")
        epoch_loss = sum(file_losses) / len(file_losses)
        print(f"epoch {epoch}: avg_loss = {epoch_loss:.6f}")
        q, _ = model.forward_from_features(test_x)
        print(q)

        if epoch % 5 == 0:
            print("Save pt file", flush=True)
            sys.stdout.flush()
            save_pt = f"../data/models/model{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": epoch_loss,
                }, save_pt)



if __name__ == "__main__":
    main()
