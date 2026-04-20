from __future__ import annotations
import numpy as np
from ase.md.md import MolecularDynamics
from ..config import kB_eV_per_K


class UnderdampedLangevinBAOAB(MolecularDynamics):
    def __init__(self, atoms, timestep, temperature_K, gamma, rng, bias_force_fn=None,
            bias_total_steps=None, trajectory=None, logfile=None, loginterval=1):
        super().__init__(atoms, timestep=timestep, trajectory=trajectory, logfile=logfile, loginterval=loginterval)
        self.dt = timestep
        self.T = float(temperature_K)
        self.kBT = kB_eV_per_K * self.T
        self.rng = np.random.default_rng() if rng is None else rng
        self.bias_force_fn = bias_force_fn
        if bias_force_fn is not None:
            self.bias_total_steps = bias_total_steps
        self.step_count=0

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

        self.last_xi = None
        self.last_c = None
        self.last_sigma = None

    def step(self, forces=None):
        atoms = self.atoms
        amp=1
        if self.bias_force_fn is not None:
            tau = self.step_count / self.bias_total_steps
            bias_forces = self.bias_force_fn(atoms, tau)
            if forces is None:
                forces = atoms.get_forces() + bias_forces*amp
            else:
                forces = forces + bias_forces*amp

        if forces is None:
            forces = atoms.get_forces()

        x = self.unwrapped_positions.copy()
        v = atoms.get_velocities().copy()
        m = self.m
        dt = self.dt
        gamma = self.gamma
        kBT = self.kBT

        v = v + 0.5 * dt * forces / m
        x = x + 0.5 * dt * v
        c = np.exp(-gamma * dt / m)
        sigma = np.sqrt((kBT / m) * (1.0 - c**2))
        xi = self.rng.normal(size=v.shape)
        v = c * v + sigma * xi

        self.last_xi = xi.copy()
        self.last_c = c.copy()
        self.last_sigma = sigma.copy()

        x = x + 0.5 * dt * v
        self.unwrapped_positions = x.copy()
        atoms.set_positions(x)
        atoms.wrap()
        new_forces = atoms.get_forces()
        if self.bias_force_fn is not None:
            tau = (self.step_count+0.5) / self.bias_total_steps
            bias_forces = self.bias_force_fn(atoms, tau) 
            new_forces = new_forces + bias_forces*amp
        v = v + 0.5 * dt * new_forces / m
        atoms.set_velocities(v)

        self.step_count += 1
        return new_forces
