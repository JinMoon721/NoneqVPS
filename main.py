from __future__ import annotations
import glob
import random
from pathlib import Path
import multiprocessing as mp
import torch
from ase import units
import numpy as np

from VPS_pkg.config import SimConfig
from VPS_pkg.model.committor_net import CommittorNet
from VPS_pkg.train.checkpoint import load_latest_checkpoint, save_checkpoint
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

    torch.set_num_threads(56)
    torch.set_num_interop_threads(1)
    print("torch threads:", torch.get_num_threads())
    print("interop threads:", torch.get_num_interop_threads())

    model = CommittorNet(tempK=config.tempK, in_dim=2, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    nepoch = 5000
    batch_size = 4
    model_dir = Path("./models")
    start_epoch = load_latest_checkpoint(model, optimizer, model_dir, device=device)

    test_size = 10
    test_tau = torch.full((test_size, 1), 0.5, dtype=torch.float32, device=device)
    test_dr = torch.linspace(0.0, 100.0, test_size).unsqueeze(1) / 80.0
    test_x = torch.cat([test_dr, test_tau], dim=1)
    q, _ = model.forward_from_features(test_x)
    print(q)

    train_files = sorted(glob.glob("../trainTwoions/merged/*.npz"))
    print(f"found {len(train_files)} files")
    random.shuffle(train_files)

    for epoch in range(start_epoch, nepoch):
        print(f"\n==== epoch {epoch + 1}/{nepoch} ====")
        file_losses = []
        for path in train_files:
            print(f"Load {path} and start training")
            avg_file_loss = train_on_one_npz(
                    path = path,
                    model = model,
                    optimizer = optimizer,
                    config = config,
                    device=device,
                    batch_size=batch_size
                    )
            file_losses.append(avg_file_loss)
            print(f"Done: {path} loss: {avg_file_loss}")

        epoch_loss = sum(file_losses) / len(file_losses)
        print(f"epoch {epoch}: avg_loss = {epoch_loss:.6f}")
        q, _ = model.forward_from_features(test_x)
        print(q)

        if epoch % 5 == 0:
            save_path = model_dir / f"model{epoch}.pt"
            print(f"Save checkpoint: {save_path}", flush=True)
            save_checkpoint(model, optimizer, epoch, epoch_loss, save_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
