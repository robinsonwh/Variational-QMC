import numpy as np
from functools import partial
from vqmc_core import metropolis, local_energy, laplacian, local_energy_nd
from systems import V_H, psi_H
from pathlib import Path
import matplotlib.pyplot as plt

from scipy.special import k1

def main():
    Path("plots").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    hydrogen_theta_scan(rng)



def hydrogen_theta_scan(rng):
    thetas = np.linspace(0.5, 1.5, 11)
    H_mean = np.empty(len(thetas))
    H_err = np.empty(len(thetas))
    E_std = np.empty(len(thetas))

    for i, theta in enumerate(thetas):
        H, E, samples = calculate_energy(theta=theta, x0=rng.normal(0, 1/theta, (1000, 3)), 
                                         rng=rng, n_steps=2000, burn_in=200, step=1.0/theta)
        H_mean[i] = H
        walker_means = E.mean(axis=0)
        H_err[i] = walker_means.std(ddof=1) / np.sqrt(len(walker_means))
        E_std[i] = E.std()
        #print(f"<H> = {H_mean[i]} +/- {H_err[i]}, std(E_l) = {E_std[i]}")

    th = np.linspace(0.45, 1.55, 200)
    exact = thetas**2 / 2 - thetas
    residual = H_mean - exact
    mask = H_err > 1e-8
    chi2_dof = np.sum((residual[mask] / H_err[mask])**2) / mask.sum()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 8),
                                    height_ratios=[2, 1, 1], layout="constrained")
    ax1.errorbar(thetas, H_mean, fmt="x", yerr=H_err, label="VMC")
    ax1.plot(th, th**2 / 2 - th, label=r"Exact: $\theta^2/2 - \theta$")
    ax1.set_title(r"$\langle H \rangle$ and std($E_l$) vs $\theta$")
    #ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\langle H \rangle$")
    ax1.legend()

    ax2.errorbar(thetas, residual, yerr=H_err, fmt="x")
    ax2.axhline(0, color="k", lw=1)
    ax2.set_ylabel(r"Residuals in $\langle H \rangle$")

    ax3.plot(thetas, E_std, "x", label=r"std($E_l$)")
    ax3.plot(th, th*np.abs(th-1), label=r"$\theta|\theta - 1|$")
    ax3.set_xlabel(r"$\theta$")
    ax3.set_ylabel(r"std($E_l$)")
    #ax3.set_title(r"std($E_l$) vs $\theta$")
    ax3.legend()
    fig.savefig("plots/hmean_vs_theta.png", dpi=200)
    plt.show()
    plt.close(fig)


def calculate_energy(theta, x0, rng, n_steps, burn_in, step, h=2.5e-4):
    f = partial(psi_H, theta)
    samples, acc = metropolis(f, x0, n_steps, step, burn_in, rng)
    E = local_energy_nd(f, V_H, samples, h)
    return E.mean(), E, samples

def P_r(theta, r):
    return 4 * theta**3 * r**2 * np.exp(-2*theta*r)

def rho_x(theta, x):
    return 0.5 * theta * (1 + 2 * theta * np.abs(x)) * np.exp(-2 * theta * np.abs(x))

def sampler_check(rng):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    r_fine = np.linspace(0, 8, 400)
    for theta in [0.5, 1.0, 1.5]:
        f = partial(psi_H, theta)
        samples, acc = metropolis(f, rng.normal(0, 1/theta, (1000, 3)),
                                  n_steps=2000, step=1/theta, burn_in=200, rng=rng)
        r = np.linalg.norm(samples, axis=-1).ravel()
        ax1.hist(r, bins=200, range=(0, 8), density=True, histtype="step",
                 label=rf"$\theta = {theta}$")
        ax1.plot(r_fine, P_r(theta, r_fine), "k--", lw=1)
        print(f"theta={theta}: acc={acc:.2f}, <r>={r.mean():.4f} (exact {1.5/theta:.4f}), "
              f"<r^2>={np.mean(r**2):.4f} (exact {3/theta**2:.4f})")
        if theta == 1.0:
            x_fine = np.linspace(-6, 6, 400)
            for d, name in enumerate("xyz"):
                ax2.hist(samples[..., d].ravel(), bins=200, range=(-6, 6),
                         density=True, histtype="step", label=name)
            ax2.plot(x_fine, rho_x(1.0, x_fine), "k--", lw=1, label="exact")
    ax1.set_xlabel("$r$"); ax1.set_ylabel("$P(r)$"); ax1.legend()
    ax1.set_title("Radial distribution (dashed: exact)")
    ax2.set_xlabel("coordinate value"); ax2.set_ylabel("density"); ax2.legend()
    ax2.set_title(r"Marginals at $\theta = 1$")
    fig.savefig("plots/sampler_check_3d.png", dpi=200)
    plt.show()
    plt.close(fig)

def n_xy(theta, x, y):
    s = np.sqrt(x**2 + y**2)
    return 2 * theta ** 3 . np.pi * s * k1(2*theta*s)

def plot_projected_density(samples, theta, L=4, bins=150):
    x = samples[]


if __name__ == "__main__":
    main()