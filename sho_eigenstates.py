import numpy as np
from functools import partial
from vqmc_core import metropolis, local_energy, laplacian, local_energy_nd
from systems import psi_sho, V_sho, V_H, psi_H, H
from pathlib import Path
import matplotlib.pyplot as plt

from scipy.special import k1

def main():
    Path("plots").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    sho_energies(rng)



def sho_energies(rng):
    for n in range(5):
        samples, acc = metropolis(partial(psi_sho, n), rng.normal(0, 2, 1000),
                                  n_steps=2000, step=1.5, burn_in=200, rng=rng)
        print(f"n={n}: shape {samples.shape}, acceptance {acc:.2f}, "
              rf"<x^2> = {np.mean(samples**2):.3f} (exact {n + 0.5})")
        E = local_energy(partial(psi_sho, n), V_sho, samples, h=2e-3)
        print(rf"n={n}: <H> = {E.mean():.10f}, std(E_l) = {E.std():.1e}  (exact {n + 0.5})")

if __name__ == "__main__":
    main()


