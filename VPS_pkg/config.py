from dataclasses import dataclass
import numpy as np
from ase import units

K_COUL = 14.3996454784255
kB_eV_per_K = 8.617333262145e-5

@dataclass(frozen=True)
class SimConfig:
    L: float
    sig: float
    eps: float
    seed: int
    tempK: float
    dtASE: float
    gammaASE: np.ndarray
    masses: np.ndarray
    rin: float
    rout: float
    dielec: float
    rin: float
    rout: float

@dataclass(frozen=True)
class MDConfig:
    interval_log: int
    interval_dump: int
    interval_ckpt: int
    nsteps: int
    max_force_stop: float ## ev/A
    tempK: float
    dtASE: float
    friction: float ## fs

@dataclass(frozen=True)
class plumedConfig:
    lam: float
    pace: int
    height: float
    sigma: float
    biasfactor: float
    stride: int
    tempK: float
    lower: float
    upper: float
    kappa: float ## Harmonic potential 

  
def default_config() -> SimConfig:
    return SimConfig(
        L=200.0,
        sig=2.0,
        eps=0.01 * units.kcal / units.mol,
        seed=1,
        tempK=300.0,
        dtASE=10.0,
        masses=np.array([35, 35], dtype=float),
        gammaASE=np.array([35, 35], dtype=float) / (150.0 * units.fs),
        rin=22.3,
        rout=70.0,
        dielec=25.0,
    )
