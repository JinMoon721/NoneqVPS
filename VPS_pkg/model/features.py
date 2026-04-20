from __future__ import annotations
import torch


def normalize_cell(cell: torch.Tensor) -> torch.Tensor:
    if cell.ndim not in (2, 3):
        raise ValueError(f"cell must have ndim 2 or 3, got shape {tuple(cell.shape)}")
    if cell.ndim == 3:
        if cell.shape[-2:] != (3, 3):
            raise ValueError(f"cell must have shape (B,3,3), got {tuple(cell.shape)}")
        return cell
    if cell.shape[-1] != 3:
        raise ValueError(f"cell must have shape (B,3), got {tuple(cell.shape)}")
    return torch.diag_embed(cell)


def make_committor_features(positions: torch.Tensor, tau: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    positions: (B, T, N, 3)
    tau:       (B, T) or (B, T, 1)
    cell:      (B, T, 3, 3)
    returns:   (B*T, 2)
    """
    B, T, N, _ = positions.shape
    dr = positions[:, :, 0, :] - positions[:, :, 1, :]
    box = torch.diagonal(cell, dim1=-2, dim2=-1)
    dr = dr.reshape(B * T, 3)
    box = box.reshape(B * T, 3)
    dr = dr - box * torch.round(dr / box)
    dr2 = (dr**2).sum(dim=1, keepdim=True)
    dr_mag = torch.sqrt(dr2 + 1e-12) / 80.0
    if tau.ndim == 2:
        tau = tau.unsqueeze(-1)
    tau = tau.reshape(B * T, 1)
    return torch.cat([dr_mag, tau], dim=-1)
