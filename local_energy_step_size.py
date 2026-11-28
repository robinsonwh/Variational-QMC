import numpy as np
import matplotlib.pyplot as plt
import timeit

from pathlib import Path
from functools import partial

from systems import H, psi_sho, V_sho, timed
from vqmc_core import hot2, hot4, hot6, local_energy

x_grid = np.linspace(-8, 8, 40000) + 1e-7

def weighted_error(deriv, n, h, x=x_grid):
    #rho-weighted mean of |E_l - E_n| over the grid x.
    f = partial(psi_sho, n)
    rho = f(x)**2
    rho /= np.trapezoid(rho, x) # normalise the PDF on the grid
    h = np.asarray(h, dtype=float)[..., None]  # (len(h), 1) so it broadcasts against x -> (len(h), len(x))
    return np.trapezoid(rho * np.abs(local_energy(f, V_sho, x, h, deriv) - (n + 0.5)), x, axis=-1)

def make_error_plots():
    hs = np.logspace(-6, 0, 61)
    eps = np.finfo(float).eps
    fig, axes = plt.subplots(3, 2, figsize=(9, 10), sharex=False, sharey=False,
                             layout="constrained")
    axes = axes.ravel()
    for n in range(5):
        ax = axes[n]
        for deriv, label in [(hot2, "HOT2"), (hot4, "HOT4"), (hot6, "HOT6")]:
            ax.loglog(hs, weighted_error(deriv, n, hs), label=label)
        ax.loglog(hs, eps / hs**2, "k--", lw=1, label=r"$\epsilon/h^2$")
        ax.set_title(rf"$\psi_{n}(x)$")
    for ax in axes[4:5]:
        ax.set_xlabel("$h$ (oscillator units)")
    axes[3].set_xlabel("$h$ (oscillator units)")
    for ax in axes[::2]:
        ax.set_ylabel(r"$\rho$-weighted $|E_l - E_n|$")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].axis("off")
    axes[5].legend(handles, labels, loc="center", fontsize=12)
    fig.savefig("plots/weighted_error.png", dpi=200)
    plt.close(fig)

def time_stencils(n=0):
    """Cost of one local-energy evaluation over x_grid, for each stencil."""
    f = partial(psi_sho, n)
    base = None
    for deriv in (hot2, hot4, hot6):
        t = min(timeit.repeat(lambda: local_energy(f, V_sho, x_grid, 2e-3, deriv),
                              number=1, repeat=20))
        base = base or t
        print(f"{deriv.__name__}: {t*1e3:.2f} ms ({t/base:.2f}x HOT2)")


if __name__ == "__main__":
    Path("plots").mkdir(exist_ok=True)
    make_error_plots()
    time_stencils()
    
