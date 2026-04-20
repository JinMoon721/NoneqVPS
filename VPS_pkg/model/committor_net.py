from __future__ import annotations
import torch
import torch.nn as nn
from ..config import kB_eV_per_K
from .features import make_committor_features


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x)


class CommittorNet(nn.Module):
    def __init__(self, tempK: float, in_dim: int = 2, hidden_dim: int = 128, q_eps: float = 1e-8):
        super().__init__()
        self.tempK = float(tempK)
        self.q_eps = q_eps
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, 1),
        )

    def make_features(self, positions: torch.Tensor, tau: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        return make_committor_features(positions, tau, cell)

    def forward(self, positions: torch.Tensor, tau: torch.Tensor, cell: torch.Tensor):
        B, T, _, _ = positions.shape
        feat = self.make_features(positions, tau, cell)
        logits = self.net(feat)
        q = torch.sigmoid(logits)
        q = torch.clamp(q, min=self.q_eps, max=1.0 - self.q_eps)
        logq = torch.log(q)
        return q.view(B, T, 1), logq.view(B, T, 1)

    def doob(self, positions: torch.Tensor, tau: torch.Tensor, cell: torch.Tensor, create_graph: bool = True):
        kBT = kB_eV_per_K * self.tempK
        positions = positions.clone().detach().requires_grad_(True)
        q, logq = self.forward(positions, tau, cell)
        grad = torch.autograd.grad(
            outputs=logq.sum(),
            inputs=positions,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        doob = 2.0 * kBT * grad
        return q, logq, doob

    def forward_from_features(self, feat: torch.Tensor):
        logits = self.net(feat)
        q = torch.sigmoid(logits)
        q = torch.clamp(q, min=self.q_eps, max=1.0 - self.q_eps)
        logq = torch.log(q)
        return q, logq
