from __future__ import annotations
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from ..config import K_COUL


class CoulombLJCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, charges, sigma, epsilon, dielectric=1.0, lj_mixing="lorentz-berthelot", **kwargs):
        super().__init__(**kwargs)
        self.charges = np.asarray(charges, dtype=float)
        self.n = len(self.charges)
        self.dielectric = float(dielectric)
        self.lj_mixing = lj_mixing
        self.sigma = np.asarray(sigma, dtype=float)
        self.epsilon = np.asarray(epsilon, dtype=float)

    def _pair_lj_params(self, i: int, j: int):
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
            raise ValueError(f"Number of atoms in Atoms object ({n_atoms}) does not match charges ({self.n}).")

        cell = self.atoms.get_cell()
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

                e_coul = K_COUL * qi * qj / (self.dielectric * r)
                dUdr_coul = -K_COUL * qi * qj / (self.dielectric * r**2)

                sigma_ij, epsilon_ij = self._pair_lj_params(i, j)
                sr6 = (sigma_ij / r) ** 6
                sr12 = sr6**2
                e_lj = 4.0 * epsilon_ij * (sr12 - sr6)
                dUdr_lj = 24.0 * epsilon_ij * (-2.0 * sr12 + sr6) / r

                energy += e_coul + e_lj
                fij = (dUdr_coul + dUdr_lj) * rhat
                forces[i] += fij
                forces[j] -= fij

        self.results["energy"] = energy
        self.results["forces"] = forces
