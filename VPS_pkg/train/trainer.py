from __future__ import annotations
import gc
import random
import torch
from ..data.io import load_one_npz, load_biased_npz
from ..data.collate import collate_samples, collate_bias_samples
from ..loss.batch_loss import compute_batch_loss, compute_bias_batch_loss
from ..config import SimConfig


def batched(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def train_on_batch(batch_id: int, batch, model, optimizer, config: SimConfig, device="cpu"):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = compute_batch_loss(batch, model, config, device=device).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    print(f"Mini-batch {batch_id} Loss: {loss_value}")
    return loss_value


def train_on_one_npz(path, model, optimizer, config: SimConfig, device="cpu", batch_size=4, shuffle_inside=True):
    samples = load_one_npz(path, box_length=config.L, masses=config.masses)
    if shuffle_inside:
        random.shuffle(samples)
    batch_losses =[]
    for bi, mini_samples in enumerate(batched(samples, batch_size)):
        batch = collate_samples(mini_samples, device=device)
        loss_value = train_on_batch(bi, batch, model, optimizer, config, device=device)
        batch_losses.append(loss_value)
    del samples
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return sum(batch_losses) / len(batch_losses)

def compute_on_one_npz(path, model, optimizer, config: SimConfig, device="cpu", batch_size=4, shuffle_inside=True):
    samples = load_one_npz(path, box_length=config.L, masses=config.masses)
    if shuffle_inside:
        random.shuffle(samples)
    all_losses = []
    model.eval()
    for bi, mini_samples in enumerate(batched(samples, batch_size)):
        batch = collate_samples(mini_samples, device=device)
        loss_value = compute_batch_loss(batch, model, config, device=device)
        loss_value = loss_value.detach()
        print(f"batch {bi} loss : {loss_value}")
        all_losses.append(loss_value)

    all_tensors = torch.cat(all_losses, dim=0)
    all_array = all_tensors.detach().numpy()
    del samples
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return all_array

def train_on_two_batch(batch_id: int, tps_batch, bias_batch,  model, optimizer, config: SimConfig, device="cpu"):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tps_loss = compute_batch_loss(tps_batch, model, config, device=device)
    bias_loss = compute_bias_batch_loss(bias_batch, model, config, device=device)
    loss = tps_loss + bias_loss
    loss = loss.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    tps_loss_value = float(tps_loss.mean().detach().cpu())
    bias_loss_value = float(bias_loss.mean().detach().cpu())
    print(f"Mini-batch {batch_id} TPS Loss: {tps_loss_value} Bias Loss: {bias_loss_value}")
    return tps_loss_value, bias_loss_value


### using half unbiased tps trajectories + half biased trajectories
def train_on_two_npz(tps_path, bias_path, model, optimizer, config: SimConfig, device="cpu", batch_size=4, shuffle_inside=True):
    ### assume tps and bias samples have the same number of trajectories
    tps_samples = load_one_npz(tps_path, box_length=config.L, masses=config.masses)
    bias_samples = load_biased_npz(bias_path, box_length=config.L, masses=config.masses)
    if shuffle_inside:
        random.shuffle(tps_samples)
        random.shuffle(bias_samples)
    n = min(len(tps_samples), len(bias_samples))
    if len(tps_samples) != len(bias_samples):
        print(f"sample size mismatch: tps {len(tps_samples)} vs bias {len(bias_samples)}")
        print(f"rematch sample size")
        tps_samples = tps_samples[:n]
        bias_samples = bias_samples[:n]
    tps_batch_losses =[]
    bias_batch_losses =[]
    tps_batches = batched(tps_samples, batch_size) ## batched
    bias_batches = batched(bias_samples, batch_size)
    for bi, (tps_mini_batch, bias_mini_batch) in enumerate(zip(tps_batches, bias_batches)):
        tps_batch = collate_samples(tps_mini_batch, device=device)
        bias_batch = collate_bias_samples(bias_mini_batch, device=device)
        tps_loss_value, bias_loss_value = train_on_two_batch(bi, tps_batch, bias_batch,
                model, optimizer, config, device=device)
        tps_batch_losses.append(tps_loss_value)
        bias_batch_losses.append(bias_loss_value)
    del tps_samples
    del bias_samples
    del tps_batches
    del bias_batches
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return sum(tps_batch_losses) / len(tps_batch_losses), sum(bias_batch_losses) /len(bias_batch_losses)
