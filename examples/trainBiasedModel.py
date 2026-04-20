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
from VPS_pkg.train.trainer import train_on_two_npz
from VPS_pkg.data.io import find_latest_checkpoint

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
    # Search it1_models, find latest model / if not exist, use latest trained model
    # --------------------------------
    remodel_dir = Path("../data/it1_model")
    remodel_dir.mkdir(parents=True, exist_ok=True)
    remodel_path, remodel_epoch = find_latest_checkpoint(
            remodel_dir, prefix="model")
    if remodel_path is not None:
        print(f"Found checkpoint: {remodel_path} epoch {remodel_epoch}")
        ckpt = torch.load(remodel_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) 
        print(f"Restart training from epoch {start_epoch}")
        if "loss" in ckpt:
            print(f"Previous save loss: {ckpt['loss']}")
    else:
        print(f"Prev it1 model not found, search for pre-trained model")
        trained_model_dir = Path("../data/models")
        trained_model_path, trained_model_epoch = find_latest_checkpoint(
                trained_model_dir, prefix="model")
        if trained_model_path is not None:
            print(f"Found checkpoint: {trained_model_path} epoch {trained_model_epoch}")
            ckpt = torch.load(trained_model_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            #start_epoch = int(ckpt["epoch"]) 
            start_epoch = 0
            print(f"Start training from epoch {start_epoch}")
            if "loss" in ckpt:
                print(f"Previous save loss: {ckpt['loss']}")
        else:
            raise ValueError(f"Trained model is not detected in {trained_model_dir}")

    
    # -------------------------------
    # Load TPS files and biased files
    # --------------------------------

    tps_files = sorted(glob.glob("../data/trainSet/*.npz"))
    print(f"found {len(tps_files)} TPS files")
    random.shuffle(tps_files)

    biased_files = sorted(glob.glob("../data/biasedTraj/batch*.npz"))
    print(f"found {len(biased_files)} biased files")
    random.shuffle(biased_files)

    for epoch in range(start_epoch, nepoch):
        print(f"\n==== epoch {epoch+1}/{nepoch} ====")
        epoch_total_loss =[]
        epoch_tps_loss=[]
        epoch_bias_loss=[]
        for tps_path, bias_path in zip(tps_files, biased_files):
            print(f"Load {tps_path} and {bias_path}, Start training")
            file_tps_loss, file_bias_loss = train_on_two_npz(
                    tps_path = tps_path,
                    bias_path = bias_path,
                    model = model,
                    optimizer = optimizer,
                    config = config,
                    device=device,
                    batch_size = batch_size,
                    )
            epoch_tps_loss.append(file_tps_loss)
            epoch_bias_loss.append(file_bias_loss)
            epoch_total_loss.append(file_tps_loss + file_bias_loss)
            print(f"Done: {tps_path} loss {file_tps_loss}  / {bias_path} loss {file_bias_loss}")

     #  epoch_loss = sum(file_losses) / len(file_losses)
     #  print(f"epoch {epoch}: avg_loss = {epoch_loss:.6f}")
     #  q, _ = model.forward_from_features(test_x)
     #  print(q)

     #  if epoch % 5 == 0:
     #      print("Save pt file", flush=True)
     #      sys.stdout.flush()
     #      save_pt = f"../data/models/model{epoch}.pt"
     #      torch.save({
     #          "model_state_dict": model.state_dict(),
     #          "optimizer_state_dict": optimizer.state_dict(),
     #          "epoch": epoch,
     #          "loss": epoch_loss,
     #          }, save_pt)



if __name__ == "__main__":
    main()
