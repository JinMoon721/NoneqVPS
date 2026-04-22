from ase import Atoms
from VPS_pkg.config import MDConfig, plumedConfig
import numpy as np
from pathlib import Path


def initialIndexing(atoms, cutCO=1.8, cutCH=1.5):
    pos = atoms.get_positions()
    cell = atoms.cell
    
    # element indices
    H0 = [i for i, a in enumerate(atoms) if a.symbol == "H"]
    C0 = [i for i, a in enumerate(atoms) if a.symbol == "C"]
    O0 = [i for i, a in enumerate(atoms) if a.symbol == "O"]
    
    # minimum-image distance
    def micDistance(r1, r2, cell):
        dr = r1 - r2
        A = np.asarray(cell)
        invA = np.linalg.inv(A)
        ds = dr @ invA
        ds = ds.copy()
        for k in range(3):
            ds[k] -= np.round(ds[k])
        dr_mic = ds @ A
        return np.linalg.norm(dr_mic)
    
    soluteO, waterO = [], []
    protonH = []
    
    # classify O atoms
    for oi in O0:
        is_solute = any(
            micDistance(pos[oi], pos[ci], cell) < cutCO
            for ci in C0
        )
        (soluteO if is_solute else waterO).append(oi)
    
    # classify H atoms
    for hi in H0:
        is_proton = any(
            micDistance(pos[hi], pos[ci], cell) < cutCH
            for ci in C0
        )
        if not is_proton:
            protonH.append(hi)
    
    return soluteO, waterO, protonH

def toPlumed(idxs0):
    return ",".join(str(i+1) for i in idxs0)


def plumedSetup(atoms, outdir: Path, segment_id: int, config: plumedConfig):
    soluteO, waterO, protonH = initialIndexing(atoms)
    print("WaterO:", len(waterO), "SoluteO:", len(soluteO), "protonH:", len(protonH))

    sites0 = waterO + soluteO
    groups = [0]*len(waterO) + [1]*len(soluteO)
    w0 = [2.0]*len(waterO) + [0.5]*len(soluteO)

    SITES = toPlumed(sites0)
    HYDROGENS = toPlumed(protonH)
    GROUP = ",".join(map(str, groups))
    W0 = ",".join(f"{x:.6g}" for x in w0)

    seg_hills=str(outdir / f"HILLS_seg{segment_id:04d}")
    seg_colvar=str(outdir / f"COLVAR_seg{segment_id:04d}")

    plumed_lines = [
        "UNITS LENGTH=A TIME=fs ENERGY=kcal/mol",
        "LOAD FILE=/pscratch/sd/s/seokjin/NoneqVPS/VPS_pkg/plumed/sd.so",
        ## define CV
        f"sd: SD LAMBDA={config.lam} "
        f"SITES={SITES} "
        f"HYDROGENS={HYDROGENS} "
        f"GROUP={GROUP} "
        f"W0={W0} ",
        
        ## well-tempered metadynamics
        # PACE: steps between hill depositions
        # HEIGHT: initial hill height (in PLUMED ENERGY units)
        # SIGMA: gaussian width in CV units (Å here)
        # BIASFACTOR: gamma = (T+ΔT)/T
        # TEMP: thermostat temperature in K
        f"metad: METAD ARG=sd PACE={config.pace} HEIGHT={config.height} SIGMA={config.sigma} BIASFACTOR={config.biasfactor} TEMP={config.tempK} FILE={seg_hills}",
        f"LOWER_WALLS ARG=sd AT={config.lower} KAPPA={config.kappa} EXP=2 OFFSET=0",
        f"UPPER_WALLS ARG=sd AT={config.upper} KAPPA={config.kappa} EXP=2 OFFSET=0",
        
        ## output
        f"PRINT STRIDE={config.stride} ARG=sd,metad.bias FILE={seg_colvar}",
        ##f"DUMPDERIVATIVES STRIDE=10 ARG=sd FILE=DERIV_{tag}",
        ]

    plumed_str = "\n".join(plumed_lines) + "\n"
    open(outdir / f"plumed_seg{segment_id:04d}.dat", "w").write(plumed_str)

    return plumed_lines, seg_hills, seg_colvar
