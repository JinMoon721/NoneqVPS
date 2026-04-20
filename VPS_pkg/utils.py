from __future__ import annotations
import torch


def to_tensor(x, device="cpu", dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)
