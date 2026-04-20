from __future__ import annotations
import torch
from ..config import SimConfig, kB_eV_per_K
from ..utils import to_tensor


def build_baoab_coefficients(config: SimConfig, device="cpu", dtype=torch.float32):
    kBT = kB_eV_per_K * config.tempK
    m = to_tensor(config.masses.astype("float32"), device=device, dtype=dtype)
    gamma = to_tensor(config.gammaASE.astype("float32"), device=device, dtype=dtype)
    c = torch.exp(-gamma * config.dtASE / m)
    sigma = torch.sqrt((kBT / m) * (1.0 - c**2))
    pref = (config.dtASE / (2.0 * m * sigma)).view(1, 1, -1, 1)
    c = c.view(1, 1, -1, 1)
    return pref, c


def compute_om_action_single(xi: torch.Tensor, doob: torch.Tensor, config: SimConfig, device="cpu"):
    xi = to_tensor(xi, device=device)
    doob = to_tensor(doob, device=device)
    pref, c = build_baoab_coefficients(config, device=device, dtype=xi.dtype)
    doob_current = doob[:-1].unsqueeze(0)
    doob_next = doob[1:].unsqueeze(0)
    a = pref * (c * doob_current + doob_next)
    a = a.squeeze(0)
    dU = torch.sum(a * xi, dim=(1, 2)) - 0.5 * torch.sum(a * a, dim=(1, 2))
    return dU.sum()
