import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.md import MolecularDynamics
from ase.io.trajectory import Trajectory
from ase.io import write, read
from pathlib import Path
from ase.md import MDLogger
from ase import units
from ase.geometry import find_mic


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import random
import sys

torch.set_num_threads(56)
torch.set_num_interop_threads(1)

print("torch threads:", torch.get_num_threads())
print("interop threads:", torch.get_num_interop_threads())

def to_tensor(x, device="cpu", dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)

import multiprocessing as mp

from dataclasses import dataclass
import matplotlib.pyplot as plt
import glob
@dataclass
class simConfig:
    L: float
    sig: float  ## LJ sigma
    eps: float  ## LJ epsilon
    seed: float ## seed for TPS snapshot selection
    tempK: float
    dtASE: float
    gammaASE: np.ndarray
    masses: np.ndarray
    rin: float
    rout: float
    dielec: float


config = simConfig(
        L = 200.0,
        sig = 2, ## A
        eps = 0.01 * units.kcal / units.mol,
        seed=1,
        tempK=300,
        dtASE=10, ## fs
        masses = np.array([35, 35]),
        gammaASE = np.array([35, 35]) / (150. * units.fs),
        rin = 22.3,
        rout= 70.0,
        dielec = 25,
        )
# ============================================================
# Basic Parameters
# ============================================================

# Coulomb constant in ASE-like units: eV * Angstrom / e^2
K_COUL = 14.3996454784255
kB_eV_per_K = 8.617333262145e-5  # eV / K


# ============================================================
# Calculator
# ============================================================
class CoulombLJCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        charges,
        sigma,
        epsilon,
        dielectric=1.0,
        lj_mixing="lorentz-berthelot",
        **kwargs
    ):
        """
        General N-particle Coulomb + LJ calculator.

        Parameters
        ----------
        charges : array-like, shape (N,)
            Charges in units of e.

        sigma : float or array-like
            LJ sigma parameter.
            Supported forms:
              - scalar: same sigma for all atoms
              - shape (N,): per-particle sigma, mixed into pair sigma_ij
              - shape (N,N): explicit pair sigma_ij

        epsilon : float or array-like
            LJ epsilon parameter.
            Supported forms:
              - scalar: same epsilon for all atoms
              - shape (N,): per-particle epsilon, mixed into pair epsilon_ij
              - shape (N,N): explicit pair epsilon_ij

        dielectric : float
            Relative dielectric constant.

        lj_mixing : str
            Currently only "lorentz-berthelot" supported for per-particle sigma/epsilon.
        """
        super().__init__(**kwargs)

        self.charges = np.asarray(charges, dtype=float)
        self.n = len(self.charges)
        self.dielectric = float(dielectric)
        self.lj_mixing = lj_mixing

        self.sigma = np.asarray(sigma, dtype=float)
        self.epsilon = np.asarray(epsilon, dtype=float)

    def _pair_lj_params(self, i, j):
        """
        Return sigma_ij, epsilon_ij.
        """
        # scalar case
        if self.sigma.ndim == 0:
            sigma_ij = float(self.sigma)
        elif self.sigma.ndim == 1:
            if self.lj_mixing != "lorentz-berthelot":
                raise ValueError("Only lorentz-berthelot mixing is supported.")
            sigma_ij = 0.5 * (self.sigma[i] + self.sigma[j])
        elif self.sigma.ndim == 2:
            sigma_ij = self.sigma[i, j]
        else:
            raise ValueError("sigma must be scalar, shape (N,), or shape (N,N).")

        if self.epsilon.ndim == 0:
            epsilon_ij = float(self.epsilon)
        elif self.epsilon.ndim == 1:
            if self.lj_mixing != "lorentz-berthelot":
                raise ValueError("Only lorentz-berthelot mixing is supported.")
            epsilon_ij = np.sqrt(self.epsilon[i] * self.epsilon[j])
        elif self.epsilon.ndim == 2:
            epsilon_ij = self.epsilon[i, j]
        else:
            raise ValueError("epsilon must be scalar, shape (N,), or shape (N,N).")

        return sigma_ij, epsilon_ij

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        positions = self.atoms.get_positions()
        n_atoms = len(positions)

        if n_atoms != self.n:
            raise ValueError(
                f"Number of atoms in Atoms object ({n_atoms}) does not match "
                f"length of charges ({self.n})."
            )

        cell = self.atoms.get_cell()
        cell_lengths = np.array([cell[0, 0], cell[1, 1], cell[2, 2]], dtype=float)

        energy = 0.0
        forces = np.zeros((n_atoms, 3), dtype=float)

        for i in range(n_atoms - 1):
            qi = self.charges[i]
            for j in range(i + 1, n_atoms):
                qj = self.charges[j]

                dr = positions[j] - positions[i]
                dr, _ = find_mic(dr, cell, self.atoms.pbc)
                r = np.linalg.norm(dr)

                if r < 1e-12:
                    raise ValueError(f"Atoms {i} and {j} are too close: r = {r}")

                rhat = dr / r

                # Coulomb
                e_coul = K_COUL * qi * qj / (self.dielectric * r)
                dUdr_coul = -K_COUL * qi * qj / (self.dielectric * r**2)

                # LJ
                sigma_ij, epsilon_ij = self._pair_lj_params(i, j)
                sr6 = (sigma_ij / r) ** 6
                sr12 = sr6 ** 2

                e_lj = 4.0 * epsilon_ij * (sr12 - sr6)
                dUdr_lj = 24.0 * epsilon_ij * (-2.0 * sr12 + sr6) / r

                # Total pair energy
                energy += e_coul + e_lj

                # Force on atom i:
                # F_i = dU/dr * rhat
                fij = (dUdr_coul + dUdr_lj) * rhat

                forces[i] += fij
                forces[j] -= fij

        self.results["energy"] = energy
        self.results["forces"] = forces


# ============================================================
# Integrator
# ============================================================

class UnderdampedLangevinBAOAB(MolecularDynamics):
    """
    Underdamped Langevin with explicit Euler-Maruyama update:

        r_{n+1} = r_n + dt * v_n
        v_{n+1} = v_n + dt * (F_n - gamma * v_n) / m
                        + sqrt(2 gamma kBT dt) / m * xi_n

    Everything is vectorized over atoms.

    Important:
    - positions are stored in Angstrom
    - velocities are in Angstrom / time_unit
    - forces are in eV / Angstrom
    - masses must be numerically consistent with your chosen time unit
    - gamma must also be consistent with that same convention

    For first debugging, you can use a reduced-unit interpretation.
    """

    def __init__(self, atoms, timestep, temperature_K, gamma, rng,
            trajectory=None, logfile=None, loginterval=1):
        super().__init__(
            atoms,
            timestep=timestep,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
        )

        self.dt = timestep
        self.T = float(temperature_K)
        self.kBT = kB_eV_per_K * self.T
        self.rng = np.random.default_rng() if rng is None else rng

        n = len(atoms)

        self.m = atoms.get_masses().reshape(n, 1)

        gamma = np.asarray(gamma, dtype=float)
        if gamma.ndim == 0:
            gamma = np.full((n, 1), float(gamma))
        elif gamma.shape == (n,):
            gamma = gamma.reshape(n, 1)
        elif gamma.shape != (n, 1):
            raise ValueError("gamma must be scalar, shape (N,), or shape (N,1)")
        self.gamma = gamma

        self.unwrapped_positions = atoms.get_positions().copy()

        if atoms.get_velocities() is None:
            atoms.set_velocities(np.zeros((n, 3), dtype=float))
            raise ValueError("Velocities not set")

        ## modification
        self.last_xi = None
        self.last_c = None
        self.last_sigma = None

    def step(self, forces=None):
        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces()

        #x = atoms.get_positions().copy()
        x = self.unwrapped_positions.copy()
        v = atoms.get_velocities().copy()

        m = self.m
        dt = self.dt
        gamma = self.gamma
        kBT = self.kBT

        ### BAOAB splitting of integrator


        # B: half kick
        v = v + 0.5 * dt * forces /m

        # A: half drift
        x = x + 0.5 * dt *v

        # O : exact OU velocity update
        c = np.exp(-gamma * dt/m)
        sigma = np.sqrt((kBT/m) * (1.0 -c**2))
        xi = self.rng.normal(size=v.shape)
        v = c * v + sigma * xi

        self.last_xi = xi.copy()
        self.last_c  = c.copy()
        self.last_sigma = sigma.copy()

        # A : half drift
        x = x+ 0.5 *dt * v

        # update ASE and compute new force
        self.unwrapped_positions = x.copy()
        atoms.set_positions(x)
        atoms.wrap()

        new_forces = atoms.get_forces()

        # B : half kick
        v = v + 0.5 * dt * new_forces/m

        atoms.set_velocities(v)

        return new_forces

# ============================================================
# helper: trajectory printer with cation in the center
# ============================================================
def cationCenteredAtoms(atoms, cation_index=0):
    a = atoms.copy()
    cell = a.get_cell()
    pos = a.get_positions()

    bcenter = 0.5 * (cell[0] + cell[1] + cell[2])
    shift = bcenter - pos[cation_index]
    pos_shifted = pos + shift

    a.set_positions(pos_shifted)
    a.wrap()

    return a

# ============================================================
# helper: rebuild calculator and dynamics for restarted atoms
# ============================================================
def attach_calc(atoms):
    calc = CoulombLJCalculator(
        charges=[+1.0, -1.0],
        sigma=[config.sig, config.sig],
        epsilon=[config.eps, config.eps],
        dielectric=config.dielec,
    )
    atoms.calc = calc

def make_dyn(atoms, rng):
    dyn = UnderdampedLangevinBAOAB(
        atoms=atoms,
        temperature_K=config.tempK,
        timestep=config.dtASE * units.fs,
        gamma=config.gammaASE,
        rng=rng,
    )
    return dyn

# ============================================================
# helper: event classification for one branch
# ============================================================
def classify_first_hit(prevR, currR, rin, rout):
    if (prevR > rin) and (currR <= rin):
        return "inner"

    if (prevR <= rout) and (currR > rout):
        return "outer"

    return None
# ============================================================
# helper: save one stitched TPS path
# ============================================================
def save_path(frames, outfile, centered=False):
    with Trajectory(outfile, "w") as traj:
        for a in frames:
            aa = a.copy()
            if centered:
                aa = cationCenteredAtoms(aa, cation_index=0)
            rij = distance(aa, 0, 1)
            aa.info["r"] = rij
            traj.write(aa)

# ============================================================
# helper: get pbc distance
# ============================================================
def distance(atoms, i=0, j=1):
    rij = atoms.positions[j] - atoms.positions[i]
    rij_mic, dist = find_mic(rij, atoms.cell, atoms.pbc)
    return dist

# --------------------------
# Utilities
# --------------------------

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x)


# --------------------------
#  Model
# --------------------------
def normalize_cell(cell : torch.Tensor) -> torch.Tensor:
    """
    convert cell to shape (B, 3, 3)
    """
    if cell.ndim != 3 and cell.ndim != 2:
        raise ValueError(f"3D cell must have shape (B,3,3), got {tuple(cell.shape)}")


    if cell.ndim == 3:
        if cell.shape[-2:] != (3, 3):
            raise ValueError(f"cell must have shape (B,3,3) or (B,3), got {tuple(cell.shape)}")
        return cell

    if cell.shape[-1] != 3:
        raise ValueError(f"2D cell must have shape (B,3), got {tuple(cell.shape)}")

    return torch.diag_embed(cell)

class CommittorNet(nn.Module):
    def __init__(self, in_dim: int=2, hidden_dim: int = 128, q_eps: float = 1e-8):
        super().__init__()
        self.q_eps = q_eps

        self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                Swish(),
                nn.Linear(hidden_dim, hidden_dim),
                Swish(),
                nn.Linear(hidden_dim, 1),
                )

    def make_features(self, positions, tau, cell):
        """
        positions : (B, T, N, 3)
        tau:        (B, T, ) or (B, T, 1)
        cell:   (B, T, 3, 3)
        return feat (B*T, 2)
        """
        B, T, N, _ = positions.shape

        dr = positions[:,:, 0, :] - positions[:, :,1, :] # (B, T, 3)
        box = torch.diagonal(cell, dim1=-2, dim2=-1) # (B,T, 3)

        dr = dr.reshape(B*T, 3)
        box = box.reshape(B*T, 3)

        dr = dr - box * torch.round(dr/box)
        dr2 = (dr ** 2).sum(dim=1, keepdim=True)
        dr = torch.sqrt(dr2+1e-12)/80 #(B*T, 1)

        if tau.ndim == 2:
            tau = tau.unsqueeze(-1) ## (B,T, 1)

        tau = tau.reshape(B*T, 1)

        feat = torch.cat([dr, tau], dim=-1) # (B*T, 2)
        return feat


    def forward(self, positions, tau, cell) -> tuple[torch.Tensor, torch.Tensor]:
        """
        get positions (B,T, N, 3) and tau (B,T, 1), cell (B,T, 3, 3)
        returns:
            q:    (B,T, 1)
            logq: (B,T, 1)
        """
        B, T, N, _ = positions.shape
        feat = self.make_features(positions, tau, cell) # (B*T, 2)
        logits = self.net(feat) # (B*T,  1)
        q = torch.sigmoid(logits) 
        q = torch.clamp(q, min=self.q_eps, max=1.0 - self.q_eps)
        logq = torch.log(q) #(B*T, 1)

        q = q.view(B, T, 1)
        logq = logq.view(B, T, 1)

        return q, logq

    def doob(self, positions, tau, cell, create_graph=True):
        """
        Compute doob force from spatial gradient of log q.
        Returns:
            q   :   (B, T, 1)
            logq:   (B, T, 1)
            doop:   (B, T,  N, 3)
        """
        kBT = kB_eV_per_K * config.tempK
        positions = positions.clone().detach().requires_grad_(True) # gen new graph 
        q, logq = self.forward(positions, tau, cell) # (B, T, 1)

        grad = torch.autograd.grad(
                outputs = logq.sum(), ## ok since each batch is independent
                inputs=positions,
                create_graph=True,
                retain_graph=True,
                )[0] # (B, T, N, 3) ## since we have one tensor "positions", result [0]
        doob = 2.0 * kBT * grad
        return q, logq, doob

    def forward_from_features(self, feat):
        """
        feat : (M,2)
        returns:
            q: (M, 1)
            logq: (M, 1)
        """
        logits = self.net(feat)
        q = torch.sigmoid(logits)
        q = torch.clamp(q, min=self.q_eps, max=1.0 - self.q_eps)
        logq = torch.log(q) #(B, 1)
        return q, logq


# ============================================================
# helper: get relative OM action using model
# ============================================================

def computeOMAction(
    xi,     # noise generated (T, N, 3) ### T = timesteps
    doob,   # doob force (T+1, N, 3)
    device="cpu",
):
    """
    Compute relative OM action with and without doob force
    Trajectory-level relative OM action calculator
    Streaming one-step BAOAB action increment:
        dU_n = sum_i [ doob_i · xi_i - 1/2 |doob_i|^2 ]
    with
        doob_i = dt/(2 m_i sigma_i) * (c_i doob_i^n + doob_i^{n+1})
        c_i = exp(-gamma dt / m_i)
        sigma_i = sqrt((kBT/m_i) * (1 - c_i^2))

    Returns:
    dU : torch scalar / cumulative increment with graph attached

    Note:
    doob force generated from positions (T+1)
    """

    kBT = kB_eV_per_K * config.tempK
    m = torch.tensor(config.masses.astype(np.float32), dtype=torch.float32, device=device) # (N,1) 
    gammaASE = torch.tensor(config.gammaASE.astype(np.float32), dtype=torch.float32, device=device) # (N,1) 

    if torch.is_tensor(xi):
        xi = xi.to(device=device, dtype=torch.float32)
    else:
        xi = torch.tensor(xi, dtype=torch.float32, device=device) # (T, N, 3)

    c = torch.exp(-gammaASE * config.dtASE / m)            # (N,)
    sigma = torch.sqrt((kBT / m) * (1.0 - c**2))    # (N,)

    pref = (config.dtASE / (2.0 * m * sigma)).view(1, -1, 1) # (1, N, 1)
    c = c.view(1, -1, 1)                              # (1, N, 1)

    doobC = doob[:-1] # (T, N, 3)
    doobN = doob[1:]  # (T, N, 3)

    a = pref * ( c * doobC + doobN)    # (T, N, 3)

    dU = torch.sum(a * xi, dim=(1,2)) - 0.5 * torch.sum(a * a, dim=(1, 2)) 
    loss = dU.sum()
    return loss


# ============================================================
# helper: run forward and backward dynamics, and save positions and noise history
# ============================================================

def generate_branch_data(atoms0, rng, n_steps):
    """
    Run branch (either forward or backward) and accumulate positions and noises necessary for loss computation

    For frame number k in this generated branch, stitched index is
    gsi : global_start_index / gss : global_step_sign
        idx_k = gsi + gss * k

    Returns:
        positions, noises
    """

    a = atoms0.copy()
    attach_calc(a)
    dyn = make_dyn(a, rng)

    pos_list = [a.get_positions().astype(np.float32).copy()]
    xi_list = []

    for i in range(n_steps):
        dyn.run(1) ## advance a
        xi_list.append(dyn.last_xi.astype(np.float32).copy())
        pos_list.append(a.get_positions().astype(np.float32).copy())

    positions = np.stack(pos_list, axis=0) # (T+1, N, 3)
    noises = np.stack(xi_list, axis=0) # (T, N, 3)
    return positions, noises

# ============================================================
# helper: run forward and backward dynamics, save trajectory data
# ============================================================
def generate_reactive_sample(
    atoms,
    seed_forward,
    seed_backward,
    steps_forward,
    steps_backward,
):
    """
    Read a initial snapshot, perform forward and backward run
    and generate trajectory info for  loss calculation

    Returns:
        dictionary
    """

    # --------------------------
    # forward branch
    # --------------------------
    print(f"Running Forward dynamics for {steps_forward} steps")
    rng_fwd = np.random.default_rng(seed_forward)
    pos_fwd, xi_fwd = generate_branch_data(
        atoms0=atoms, 
        rng=rng_fwd, 
        n_steps=steps_forward, 
    )

    # --------------------------
    # backward branch
    # use same forward BAOAB but flip velocities
    # --------------------------
    print(f"Running Backward dynamics for {steps_backward} steps")
    atoms_back = atoms.copy()
    atoms_back.set_velocities(-atoms_back.get_velocities())

    rng_bwd = np.random.default_rng(seed_backward)
    pos_bwd, xi_bwd = generate_branch_data(
        atoms0=atoms_back, 
        rng=rng_bwd, 
        n_steps=steps_backward, 
    )

    return {
            "pos_fwd": pos_fwd,
            "xi_fwd": xi_fwd,
            "cell": np.array(atoms.cell, dtype=np.float32),
            "pos_bwd": pos_bwd,
            "xi_bwd": xi_bwd,
            "steps_forward": steps_forward,
            "steps_backward": steps_backward,
            "masses": atoms.get_masses().astype(np.float32),
    }
            

# ============================================================
# helper: compute loss of batch trajectory data
# ============================================================

def compute_branch_loss(
        positions,  # (T+1, N, 3)
        noises,     # (T, N, 3 )
        cell,       # (T, 3, 3 )
        total_steps,
        gsi,
        gss,
        model,
        device="cpu",
        ):
    """
    compute loss for half run (either forward or backward)
    Returns:
        loss with graph
    """

    n_steps = positions.shape[0] # T+1
    step_idx = torch.arange(gsi, gsi + gss* (n_steps), gss, device=device, dtype=torch.float32)
    tau = step_idx / float(total_steps) # (T+1,)
    tau = tau.unsqueeze(1) # (T+1, 1)

    _, _, doob = model.doob(positions, tau, cell) # (T+1, N, 3)
    if doob.shape != positions.shape:
        raise ValueError(f"model.doob should return the same shape as position")
    
    dU = computeOMAction(xi = noises, doob = doob)

    return dU



def make_doob_masks(sf, T, N,  dtype=torch.float32):
    """
    sf: (B,) tensor of steps_fowrad
    T: int : sf+sb
    N : number of atoms
    returns 4 masks of shape (B, T+2, N, 3)
    """
    device = sf.device

    B = sf.shape[0]

    k = torch.arange(T+2, device=device).view(1, T+2, 1, 1)
    sfv = sf.view(B, 1, 1, 1)

    mask_doobC_fwd = (k < sfv)
    mask_doobC_bwd = (k >= (sfv+1)) & (k <= T)
    mask_doobN_fwd = (k >= 1) & (k <= sfv)
    mask_doobN_bwd = (k >= (sfv+2))


    mask_doobC_fwd = mask_doobC_fwd.to(dtype).expand(B, T+2, N, 3)
    mask_doobC_bwd = mask_doobC_bwd.to(dtype).expand(B, T+2, N, 3)
    mask_doobN_fwd = mask_doobN_fwd.to(dtype).expand(B, T+2, N, 3)
    mask_doobN_bwd = mask_doobN_bwd.to(dtype).expand(B, T+2, N, 3)

    return {
            "mask_doobC_fwd": mask_doobC_fwd,
            "mask_doobC_bwd": mask_doobC_bwd,
            "mask_doobN_fwd": mask_doobN_fwd,
            "mask_doobN_bwd": mask_doobN_bwd,
            }

def make_step_masks(sf, T, N, dtype=torch.float32):
    device = sf.device
    B = sf.shape[0]
    t = torch.arange(T, device=device).view(1, T, 1, 1) 
    sfv = sf.view(B, 1, 1, 1)

    mask_fwd = (t<sfv).to(dtype).expand(B, T, N, 3)
    mask_bwd = (t >= sfv).to(dtype).expand(B, T, N, 3)

    return mask_fwd, mask_bwd

    
def build_tau(steps_forward, steps_backward,  dtype=torch.float32):
    """
    steps_forward : (B,) int tensor
    Returns:
    tau : (B, T+2, 1)
    """
    device = steps_forward.device

    B = steps_forward.shape[0]
    Tp2 = steps_forward + steps_backward +2
    Lmax = int( Tp2.max().item())

    tau = torch.zeros(B, Lmax, 1, device=device, dtype=dtype)
    
    for b in range(B):
        sf = int(steps_forward[b].item())
        sb = int(steps_backward[b].item())

        t_fwd = torch.arange(sb, sb+sf+1, device=device)
        t_bwd = torch.arange(sb, -1, -1, device=device)
        t = torch.cat([t_fwd, t_bwd], dim=0) # (sf+1 + sb + 1,)
        denom = sf + sb

        tau_b = t.to(dtype) / float(denom)
        L = t.numel()
        tau[b, :L, 0] = tau_b
    return tau

def compute_batch_loss(
        samples,
        model,
        device="cpu",
        ):
    """
    samples["steps_forward"] : tensor (B,)
    samples["steps_backward"] : tensor (B,)
    samples["total_steps"] : int
    samples["positions"] : tensor (B, T+2, N, 3)
    samples["noises"] : tensor (B, T, N, 3)
    samples["cell"] : tensor (B, T+2, 3, 3)
    """
    T = samples["total_steps"]
    steps_forward = samples["steps_forward"]
    steps_backward = samples["steps_backward"]
    B = steps_forward.shape[0]
    positions = samples["positions"]
    noises = samples["noises"]
    N = positions.shape[2]
    cell = samples["cell"]

    tau = build_tau(steps_forward, steps_backward)

    _, _, doob = model.doob( positions, tau, cell) # (B, T+2, N, 3)
    if doob.shape != positions.shape:
        raise ValueError(f"model.doob must return the same size as positions")

    masks = make_doob_masks(sf = steps_forward, T= T, N=N, 
            dtype = positions.dtype)

    doobC_fwd = (doob * masks["mask_doobC_fwd"])[:, :-2, :, :] ## (B, T, N, 3)
    doobN_fwd = (doob * masks["mask_doobN_fwd"])[:, 1:-1, :, :] ## (B, T, N, 3)
    doobC_bwd = (doob * masks["mask_doobC_bwd"])[:, 1:-1, :, :]
    doobN_bwd = (doob * masks["mask_doobN_bwd"])[:, 2:, :, :]## (B, T, N, 3)

    kBT = kB_eV_per_K * config.tempK
    m = to_tensor(config.masses.astype(np.float32)) # (N, )
    gammaASE = to_tensor(config.gammaASE.astype(np.float32)) # (N, )

    c = torch.exp(-gammaASE * config.dtASE / m)            # (N,)
    sigma = torch.sqrt((kBT / m) * (1.0 - c**2))    # (N,)

    pref = (config.dtASE / (2.0 * m * sigma)).view(1, 1, -1, 1) # (1, 1, N, 1)
    c = c.view(1, 1, -1, 1)                              # (1, 1, N, 1)

    a_fwd = pref * ( c * doobC_fwd + doobN_fwd)    # (B, T, N, 3)
    a_bwd = pref * ( c * doobC_bwd + doobN_bwd)    # (B, T, N, 3)

    mask_fwd, mask_bwd = make_step_masks(steps_forward, T, N, 
            dtype= positions.dtype) # (B, T, N, 3)

    dU_fwd = (torch.sum(a_fwd * noises * mask_fwd, dim=(1,2,3)) 
            - 0.5 * torch.sum(a_fwd * a_fwd * mask_fwd, dim=(1, 2,3)) 
            )

    dU_bwd = (torch.sum(a_bwd * noises * mask_bwd, dim=(1,2,3)) 
            - 0.5 * torch.sum(a_bwd * a_bwd * mask_bwd, dim=(1, 2,3))
            )

    loss_vec = -(dU_fwd + dU_bwd) # (B, )
    loss = loss_vec.mean()

    return loss


# ============================================================
# helper: compute loss of one full trajectory
# ============================================================

def computeTrajectoryLoss(sample, model, device="cpu"):
    steps_forward = int(sample["steps_forward"])
    steps_backward= int(sample["steps_backward"])
    total_steps = steps_forward + steps_backward

    if not torch.is_tensor(sample["pos_fwd"]):
        pos_fwd = torch.tensor(sample["pos_fwd"], dtype=torch.float32, device=device)
    if not torch.is_tensor(sample["xi_fwd"]):
        xi_fwd = torch.tensor(sample["xi_fwd"], dtype=torch.float32, device=device)

    if not torch.is_tensor(sample["pos_bwd"]):
        pos_bwd = torch.tensor(sample["pos_bwd"], dtype=torch.float32, device=device)
    if not torch.is_tensor(sample["xi_bwd"]):
        xi_bwd = torch.tensor(sample["xi_bwd"], dtype=torch.float32, device=device)

    cell = torch.tensor(sample["cell"], dtype=torch.float32, device=device)
    cell_fwd = cell.unsqueeze(0).expand(steps_forward+1, 3, 3)
    cell_bwd = cell.unsqueeze(0).expand(steps_backward+1, 3, 3)
        
    obj_fwd = compute_branch_loss(
            positions = pos_fwd,
            noises    = xi_fwd,
            cell      = cell_fwd,
            total_steps = total_steps,
            gsi = steps_backward,
            gss = +1,
            model = model,
            device=device,
            )
    obj_bwd = compute_branch_loss(
            positions = pos_bwd,
            noises    = xi_bwd,
            cell      = cell_bwd,
            total_steps = total_steps,
            gsi = steps_backward,
            gss = -1,
            model = model,
            device=device,
            )

    ## loss = -dU
    return -(obj_fwd + obj_bwd)

def collate_samples(samples, device="cpu"):
    """
    Inputs:
        samples : list of sample dicts
            each sample contains
                steps_forward
                steps_backward
                pos_fwd : (sf+1, N, 3)
                pos_bwd : (sb+1, N, 3)
                xi_fwd  : (sf, N, 3)
                xi_bwd  : (sb, N, 3)
                cell    : (3, 3)
    Returns:
        dict with
            steps_forward   : (B,)
            steps_backward  : (B,)
            total_steps     : int
            positions       : (B, total_steps+2, N, 3)
            noises          : (B, total_steps, N, 3)
            cell            : (B, total_steps+2, 3, 3)
    """
    B = len(samples)
    if B == 0:
        raise ValueError("samples is empty")
    steps_forward = torch.tensor(
            [int(s["steps_forward"]) for s in samples],
            dtype=torch.long,
            device=device,
            ) # (B,)
    steps_backward = torch.tensor(
            [int(s["steps_backward"]) for s in samples],
            dtype=torch.long,
            device=device,
            ) # (B,)
    total_steps_all = steps_forward + steps_backward
    if not torch.all(total_steps_all == total_steps_all[0]):
        raise ValueError("All samples must have the same total_steps")

    total_steps = int(total_steps_all[0].item())

    positions_list=[]
    noises_list = []
    cell_list = []

    for s in samples:
        sf = int(s["steps_forward"])
        sb = int(s["steps_backward"])

        pos_fwd = to_tensor(s["pos_fwd"], device=device)    # (sf+1, N, 3)
        xi_fwd = to_tensor(s["xi_fwd"], device=device)      # (sf,   N, 3)
        pos_bwd = to_tensor(s["pos_bwd"], device=device)    # (sb+1, N, 3)
        xi_bwd = to_tensor(s["xi_bwd"], device=device)      # (sb,   N, 3)
        cell = to_tensor(s["cell"], device=device)          # (3, 3)

        pos = torch.cat([pos_fwd, pos_bwd], dim=0) # (total_steps+2, N, 3)
        noise = torch.cat([xi_fwd, xi_bwd], dim=0) # (total_steps, N, 3)

        if pos.shape[0] != total_steps +2:
            raise ValueError(f"positions length {pos.shape[0]} != tot+2 {total_steps+2}")

        if noise.shape[0] != total_steps:
            raise ValueError(f"noises length {noise.shape[0]} != tot+2 {total_steps+2}")

        cell_full = cell.unsqueeze(0).expand(total_steps +2, 3, 3) 

        positions_list.append(pos)
        noises_list.append(noise)
        cell_list.append(cell_full)
    
    positions = torch.stack(positions_list, dim=0) # (B, total_steps+2, N, 3)
    noises = torch.stack(noises_list, dim=0) # (B, total_steps, N, 3)
    cell = torch.stack(cell_list, dim=0) # (B, total_steps+2, N, 3)

    return {
            "steps_forward": steps_forward,
            "steps_backward": steps_backward,
            "total_steps": total_steps,
            "positions": positions,
            "noises": noises,
            "cell": cell
            }


            

# ============================================================
# helper: parallel generation of sample
# ============================================================

def worker_generate(job):
    idx, meta_file, snap_file = job
    atom, sef, seb, stf, stb = read_metadata(meta_file, snap_file)
    sample = generate_reactive_sample(
            atoms = atom,
            seed_forward = sef,
            seed_backward = seb,
            steps_forward = stf,
            steps_backward = stb,
            )
    return idx, sample

def generate_chunk(jobs, nproc):
    with mp.Pool(processes=nproc) as pool:
        results = pool.map(worker_generate, jobs)
    results.sort(key=lambda x: x[0])
    return [x[1] for x in results]


# ============================================================
# helper: reverse path and flip velocities back
# ============================================================
def reverse_path_with_velocity_flip(frames):
    out = []
    for a in reversed(frames):
        aa = a.copy()
        v = aa.get_velocities()
        if v is None:
            raise ValueError("Saved trajectory frame has no velocities.")
        aa.set_velocities(-v)
        out.append(aa)
    return out

# ============================================================
# helper: save metadata for accepted tps 
# ============================================================
from typing import Any, Optional, List, Dict

def _format_value_for_txt(value: Any) -> str:
    """Convert Python object to a readable text representation."""
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating,)):
        return repr(float(value))
    if isinstance(value, (np.bool_,)):
        return "True" if bool(value) else "False"
    if isinstance(value, (list, tuple, dict, bool, int, float, str)) or value is None:
        return repr(value)
    return repr(str(value))

def save_metadata(
        atomsIn: Atoms,
        trial_id: int,
        output_dir: str | Path,
        seed_forward: int,
        seed_backward: int,
        steps_forward: int,
        steps_backward: int,
        ) -> tuple[Path, Path]:
    """
    Save accepted TPS metadata
    Returns
    meta_path
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"trial_{trial_id}"
    atom_path= output_dir / f"{tag}.traj"
    meta_path= output_dir / f"{tag}_meta"
    
    atoms = atomsIn.copy()

    v=atoms.get_velocities()
    if v is None:
        raise ValueError("Somethings wrong")

    snapshot_initial = Trajectory(atom_path, "w")
    snapshot_initial.write(atoms)
    snapshot_initial.close()
    metadata= {
            "trial_id": trial_id,
            "structure_file":atom_path.name,
            "n_atoms":len(atoms),
            "seed_forward":seed_forward,
            "seed_backward":seed_backward,
            "steps_forward":steps_forward,
            "steps_backward":steps_backward,
            }
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("# Accepted TPS trial metadata\n")
        f.write("# Format: key = Python-literal-readable value\n")
        for key, value in metadata.items():
            f.write(f"{key} = {_format_value_for_txt(value)}\n")
    return atom_path, meta_path


# ============================================================
# helper: read metadata and snapshots for accepted tps 
# ============================================================
import ast

def read_metadata(
        meta_path: str | Path,
        structure_path: str | Path,
        ) -> tuple[Atoms, dict[str, Any]]:
    """
    Read metadata and snapshot to reproduce reactive path
    """
    meta_path = Path(meta_path)
    metadata: dict[str, Any] = {}

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            metadata[key] = ast.literal_eval(value)

    structure_path = Path(structure_path)
    atoms = read(structure_path)

    seed_forward = metadata.get("seed_forward", None)
    seed_backward = metadata.get("seed_backward", None)

    steps_forward = metadata.get("steps_forward", None)
    steps_backward = metadata.get("steps_backward", None)

    return atoms, seed_forward, seed_backward, steps_forward, steps_backward


# ============================================================
# helper: build job list from read metadata 
# ============================================================
def build_job_list(meta_paths, snap_paths):
    if len(meta_paths) != len(snap_paths):
        raise ValueError("path length mismatch")
    jobs =[]
    for i, (m, s) in enumerate(zip(meta_paths, snap_paths)):
        jobs.append((i, str(m), str(s)))
    
    return jobs

import gc
def train_from_generated_chunks(
        model,
        optimizer,
        all_jobs,
        chunk_size=10,
        nproc=10,
        nepoch=1,
        device="cpu",
        ):
    for epoch in range(nepoch):
        model.train()
        epoch_loss=0.0
        n_seen = 0

        for start in range(0, len(all_jobs), chunk_size):
            jobs = all_jobs[start:start + chunk_size]
            samples = generate_chunk(jobs, nproc=nproc)

            optimizer.zero_grad(set_to_none=True)

            total_loss = torch.zeros((), dtype=torch.float32, device=device)

            for sample in samples:
                loss_i = computeTrajectoryLoss(
                        sample = sample,
                        model=model,
                        device=device,
                        )
                total_loss = total_loss + loss_i

            total_loss = total_loss / len(samples)
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * len(samples)
            n_seed += len(samples)

            del samples, total_loss
            gc.collect()
        print(f"epoch {epoch}: loss = {epoch_loss / n_seen:.6f}")


def train_on_chunk(samples, model, optimizer, device="cpu"):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_size = len(samples)

    total_loss = torch.zeros((), dtype=torch.float32, device=device)

    for sample in samples:
        print("loss computation starts")
        loss_i = computeTrajectoryLoss(
                sample = sample,
                model=model,
                device=device,
                )
        total_loss = total_loss + loss_i
        print("loss computation is done")
    print("setup optimizer")
    total_loss = total_loss / len(samples)
    total_loss.backward()
    optimizer.step()
    print("setup optimizer done")

    loss_value = float(total_loss.detach().cpu())
    return loss_value

def train_on_batch(batchid, samples, model, optimizer, device="cpu"):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = compute_batch_loss(samples, model, device)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    print(f"Mini-batch {batchid} Loss : {loss_value}")
    return loss_value

def save_chunk(samples, path):
    out = {}
    out["num_samples"] = len(samples)
    for i, s in enumerate(samples):
        out[f"pos_fwd_{i}"] = s["pos_fwd"]
        out[f"xi_fwd_{i}"] = s["xi_fwd"]
        out[f"cell_{i}"] = s["cell"]
        out[f"pos_bwd_{i}"] = s["pos_bwd"]
        out[f"xi_bwd_{i}"] = s["xi_bwd"]
        out[f"steps_forward_{i}"] = np.array([s["steps_forward"]], dtype=np.int32)
        out[f"steps_backward_{i}"] = np.array([s["steps_backward"]], dtype=np.int32)
        out[f"masses_{i}"] = s["masses"]
    np.savez(path, **out)

def load_chunk(path):
    path = Path(path)

    with np.load(path) as data:
        num_samples = int(data["num_samples"])

        samples = []
        cell = np.array([[ config.L, 0., 0.], [0., config.L, 0.], [0., 0., config.L]])
        for i in range(num_samples):
            s = {
                "pos_fwd": data[f"pos_fwd_{i}"],
                "xi_fwd": data[f"xi_fwd_{i}"],
                "pos_bwd": data[f"pos_bwd_{i}"],
                "xi_bwd": data[f"xi_bwd_{i}"],
                "cell": cell,
                "steps_forward": len(data[f"pos_fwd_{i}"])-1,
                "steps_backward": len(data[f"pos_bwd_{i}"])-1,
                "masses": config.masses,
                }
            samples.append(s)
    return samples

def load_one_npz(path: str) -> List[Dict[str, Any]]:
    """
    Read one merged npz file that contains up to 20 trajectories.
    Returns a list of sample dicts
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    num_samples = int(data["num_samples"])
    samples = []
    cell = np.array([[ config.L, 0., 0.], [0., config.L, 0.], [0., 0., config.L]])
    for i in range(num_samples):
        s = {
            "pos_fwd": data[f"pos_fwd_{i}"],
            "xi_fwd": data[f"xi_fwd_{i}"],
            "pos_bwd": data[f"pos_bwd_{i}"],
            "xi_bwd": data[f"xi_bwd_{i}"],
            "cell": cell,
            "steps_forward": len(data[f"pos_fwd_{i}"])-1,
            "steps_backward": len(data[f"pos_bwd_{i}"])-1,
            }
        samples.append(s)
    return samples

def batched(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def train_on_one_npz(path, model, optimizer, device="cpu", batch_size=4,
        shuffle_inside=True):
    """
    load one npz file, split into mini-batch, train, then free memory
    """

    samples = load_one_npz(path)
    if shuffle_inside:
        random.shuffle(samples)

    batch_losses =[]

    for bi, mini_samples in enumerate(batched(samples, batch_size)):
        merged_samples = collate_samples(mini_samples, device=device)
        loss_value = train_on_batch(
                batchid = bi,
                samples = merged_samples,
                model = model,
                optimizer=optimizer,
                device=device,
                )
        batch_losses.append(loss_value)

    del samples
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return sum(batch_losses) / len(batch_losses)

def main():

    device="cpu"

    model = CommittorNet(in_dim=2, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
            )
    nepoch = 5000
    batch_size=4

    # -------------------------------
    # restart from latest checkpoint if exists
    # --------------------------------
    model_dir = Path("./models")
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

    dr_min = 0.0
    dr_max = 100

    test_size=10
    test_tau = torch.full((test_size, 1), 0.5, dtype=torch.float32, device=device)
    test_dr = torch.linspace(dr_min, dr_max, test_size).unsqueeze(1)/80
    test_x=torch.cat([test_dr, test_tau], dim=1)
 
    q, _ = model.forward_from_features(test_x)
    print(q)

    train_files = sorted(glob.glob("./merged/*.npz"))
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
            save_pt = f"./models/model{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": epoch_loss,
                }, save_pt)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

