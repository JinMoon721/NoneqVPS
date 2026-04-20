from __future__ import annotations
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic


def cation_centered_atoms(atoms, cation_index=0):
    a = atoms.copy()
    cell = a.get_cell()
    pos = a.get_positions()
    box_center = 0.5 * (cell[0] + cell[1] + cell[2])
    shift = box_center - pos[cation_index]
    a.set_positions(pos + shift)
    a.wrap()
    return a


def distance(atoms, i=0, j=1):
    rij = atoms.positions[j] - atoms.positions[i]
    _, dist = find_mic(rij, atoms.cell, atoms.pbc)
    return dist


def classify_first_hit(prev_r, curr_r, rin, rout):
    if (prev_r > rin) and (curr_r <= rin):
        return "inner"
    if (prev_r <= rout) and (curr_r > rout):
        return "outer"
    return None


def save_path(frames, outfile, centered=False):
    with Trajectory(outfile, "w") as traj:
        for atoms in frames:
            aa = atoms.copy()
            if centered:
                aa = cation_centered_atoms(aa, cation_index=0)
            aa.info["r"] = distance(aa, 0, 1)
            traj.write(aa)


def reverse_path_with_velocity_flip(frames):
    out = []
    for atoms in reversed(frames):
        aa = atoms.copy()
        v = aa.get_velocities()
        if v is None:
            raise ValueError("Saved trajectory frame has no velocities.")
        aa.set_velocities(-v)
        out.append(aa)
    return out
