from __future__ import annotations
import torch
from ..config import SimConfig
from ..data.types import BatchSamples, ReactiveSample
from ..loss.om_action import build_baoab_coefficients, compute_om_action_single
from ..data.collate import collate_samples


def build_tau(steps_forward: torch.Tensor, steps_backward: torch.Tensor, dtype=torch.float32):
    device = steps_forward.device
    B = steps_forward.shape[0]
    Tp2 = steps_forward + steps_backward + 2
    Lmax = int(Tp2.max().item())
    tau = torch.zeros(B, Lmax, 1, device=device, dtype=dtype)
    for b in range(B):
        sf = int(steps_forward[b].item())
        sb = int(steps_backward[b].item())
        t_fwd = torch.arange(sb, sb + sf + 1, device=device)
        t_bwd = torch.arange(sb, -1, -1, device=device)
        t = torch.cat([t_fwd, t_bwd], dim=0)
        tau[b, : t.numel(), 0] = t.to(dtype) / float(sf + sb)
    return tau


def make_doob_masks(sf: torch.Tensor, T: int, N: int, dtype=torch.float32):
    device = sf.device
    B = sf.shape[0]
    k = torch.arange(T + 2, device=device).view(1, T + 2, 1, 1)
    sfv = sf.view(B, 1, 1, 1)
    return {
        "mask_doobC_fwd": (k < sfv).to(dtype).expand(B, T + 2, N, 3),
        "mask_doobC_bwd": ((k >= (sfv + 1)) & (k <= T)).to(dtype).expand(B, T + 2, N, 3),
        "mask_doobN_fwd": ((k >= 1) & (k <= sfv)).to(dtype).expand(B, T + 2, N, 3),
        "mask_doobN_bwd": (k >= (sfv + 2)).to(dtype).expand(B, T + 2, N, 3),
    }


def make_step_masks(sf: torch.Tensor, T: int, N: int, dtype=torch.float32):
    device = sf.device
    B = sf.shape[0]
    t = torch.arange(T, device=device).view(1, T, 1, 1)
    sfv = sf.view(B, 1, 1, 1)
    return (
        (t < sfv).to(dtype).expand(B, T, N, 3),
        (t >= sfv).to(dtype).expand(B, T, N, 3),
    )


def compute_batch_loss(batch: BatchSamples, model, config: SimConfig, device="cpu"):
    T = batch.total_steps
    steps_forward = batch.steps_forward
    steps_backward = batch.steps_backward
    positions = batch.positions
    noises = batch.noises
    cell = batch.cell
    N = positions.shape[2]

    tau = build_tau(steps_forward, steps_backward, dtype=positions.dtype)
    _, _, doob = model.doob(positions, tau, cell)
    if doob.shape != positions.shape:
        raise ValueError("model.doob must return the same shape as positions")

    masks = make_doob_masks(steps_forward, T, N, dtype=positions.dtype)
    doobC_fwd = (doob * masks["mask_doobC_fwd"])[:, :-2, :, :]
    doobN_fwd = (doob * masks["mask_doobN_fwd"])[:, 1:-1, :, :]
    doobC_bwd = (doob * masks["mask_doobC_bwd"])[:, 1:-1, :, :]
    doobN_bwd = (doob * masks["mask_doobN_bwd"])[:, 2:, :, :]

    pref, c = build_baoab_coefficients(config, device=device, dtype=positions.dtype)
    a_fwd = pref * (c * doobC_fwd + doobN_fwd)
    a_bwd = pref * (c * doobC_bwd + doobN_bwd)

    mask_fwd, mask_bwd = make_step_masks(steps_forward, T, N, dtype=positions.dtype)
    dU_fwd = torch.sum(a_fwd * noises * mask_fwd, dim=(1, 2, 3)) - 0.5 * torch.sum(a_fwd * a_fwd * mask_fwd, dim=(1, 2, 3))
    dU_bwd = torch.sum(a_bwd * noises * mask_bwd, dim=(1, 2, 3)) - 0.5 * torch.sum(a_bwd * a_bwd * mask_bwd, dim=(1, 2, 3))
    loss_vec = -(dU_fwd + dU_bwd)
    return loss_vec # (B,)

def compute_bias_batch_loss(batch:BatchBiasedSamples, model, config: SimConfig, device="cpu"):
    positions = batch.positions # (B, T+1, N, 3)
    noises = batch.noises # (B, T, N, 3)
    cell = batch.cell
    B = noises.shape[0]
    T = noises.shape[1]
    N = positions.shape[2]

    t = torch.arange(T+1, dtype = positions.dtype) #(T+1,)
    tau = (t / T).view(1, T+1, 1) # (1, T+1, 1)
    tau = tau.expand(B, T+1, 1)
    _, _, doob = model.doob(positions, tau, cell)

    if doob.shape != positions.shape:
        raise ValueError("model.doob must return the same shape as positions")

    doobC = doob[:, :-1, :, :] # (B, T, N, 3)
    doobN = doob[:, 1:, :, :] # (B, T, N, 3)

    pref, c = build_baoab_coefficients(config, device=device, dtype=positions.dtype)
    a = pref * (c * doobC + doobN)

    dU = torch.sum(a * noises, dim=(1, 2, 3)) - 0.5 * torch.sum(a * a, dim=(1, 2, 3))
    loss_vec = +dU 
    return loss_vec # (B,)


def compute_trajectory_loss(sample: ReactiveSample, model, config: SimConfig, device="cpu"):
    batch = collate_samples([sample], device=device)
    return compute_batch_loss(batch, model, config, device=device)
