from __future__ import annotations
from pathlib import Path
import torch


def find_latest_checkpoint(model_dir: str | Path, device="cpu"):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = None
    latest_ckpt_epoch = -1
    for ckpt_path in sorted(model_dir.glob("*.pt")):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            ckpt_epoch = int(ckpt.get("epoch", -1))
            if ckpt_epoch > latest_ckpt_epoch:
                latest_ckpt_epoch = ckpt_epoch
                latest_ckpt_path = ckpt_path
        except Exception:
            continue
    return latest_ckpt_path, latest_ckpt_epoch


def load_latest_checkpoint(model, optimizer, model_dir: str | Path, device="cpu") -> int:
    ckpt_path, _ = find_latest_checkpoint(model_dir, device=device)
    if ckpt_path is None:
        print("No saved checkpoint found")
        return 0
    print(f"Found checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = int(ckpt["epoch"])
    print(f"Restart training from epoch {start_epoch}")
    if "loss" in ckpt:
        print(f"Previous save loss: {ckpt['loss']}")
    return start_epoch


def save_checkpoint(model, optimizer, epoch: int, loss: float, save_path: str | Path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        save_path,
    )
