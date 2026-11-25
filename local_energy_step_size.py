import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial

from systems import H, psi_sho, V_sho
from vqmc_core import hot2, hot4, hot6, local_energy

x_grid = np.linspace(-8, 8, 40000) + 1e-7

def weighted_error(deriv, n, h, x=x_grid):
    #rho-weighted mean of |E_l - E_n| over the grid x.
    f = partial(psi_sho, n)
    rho = f(x)**2
    rho /= np.trapezoid(rho, x) # normalise the PDF on the grid
    h = np.asarray(h, dtype=float)[..., None]  # (len(h), 1) so it broadcasts against x -> (len(h), len(x))
    return np.trapezoid(rho * np.abs(E_l(deriv, n, x, h) - (n + 0.5)), x, axis=-1)

if __name__ == "__main__":
    Path("plots").mkdir(exist_ok=True)
    hs = np.logspace(-6, 0, 61)
    eps = np.finfo(float).eps
    for n in range(5):
        plt.figure()
        for deriv, label in [(hot2, "HOT2"), (hot4, "HOT4"), (hot6, "HOT6")]:
            plt.loglog(hs, weighted_error(deriv, n, hs), label=label)
        plt.loglog(hs, eps / hs**2, "k--", lw=1, label=r"$\epsilon/h^2$")
        plt.title(rf"Weighted error for $\psi_{n}(x)$")
        plt.xlabel("$h$ (oscillator units)")
        plt.ylabel(r"$\rho$-weighted mean of $|E_l - E_n|$")
        plt.legend()
        plt.savefig(f"plots/weighted_error_{n}.png", dpi=200)
        plt.close()