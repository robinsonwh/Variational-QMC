import numpy as np
from functools import partial
from vqmc_core import metropolis, local_energy, local_energy_nd, optimise, final_energy
from systems import V_H, psi_H, n_xy
from pathlib import Path
import matplotlib.pyplot as plt
from density_plotting import projected_density, plot_density
#import time



def main():
    Path("plots").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    rngs = np.random.default_rng(42).spawn(6)

    #hydrogen_theta_scan(rngs[0])
    #sampler_check(rngs[1])
    laplacian_check(rngs[2])

    thetas_opt, history = optimise(lambda th, R: psi_H(th[0], R), V_H,
         lambda th, R: -np.linalg.norm(R, axis=-1)[..., None],
         theta0=[0.5], x0=rngs[3].normal(0, 2, (1000, 3)), rng=rngs[3],
         step=lambda th: 1/th[0], alpha=1.0)
    theta_opt = thetas_opt[0]
    plot_optimisation(rngs[4])

    H, H_err, E_std, samples = final_energy(
    partial(psi_H, theta_opt), V_H,
    x0=rngs[5].normal(0, 1/theta_opt, (1000, 3)), rng=rngs[5], step=1/theta_opt)
    print(f"theta* = {theta_opt:.10f}")
    print(f"<H> = {H:.10f} +/- {H_err:.1e}, std(E_l) = {E_std:.1e}  (exact -0.5)")
    np.savetxt("results/hydrogen.csv", [[H, H_err]], delimiter=",",
           header="E_single,E_single_err", comments="")
    plot_projected_density(samples, theta_opt)


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


def laplacian_check(rng, h=2.5e-4):
    R = rng.normal(0, 1, (10000, 3))
    r = np.linalg.norm(R, axis=-1)
    print(r"theta | max(err) | median(err)")
    for theta in np.linspace(0.5, 1.5, 11):
        f = partial(psi_H, theta)
        numerical = local_energy_nd(f, V_H, R, h)
        analytic = -theta**2 / 2 + (theta - 1)/r
        err = np.abs(numerical - analytic)
        print(f"{theta:5.2f}   {err.max():.1e}     {np.median(err):.1e}")


def plot_optimisation(rng, alphas=(0.1, 0.5, 1.0, 1.9), theta0=0.5):
    fig, ax = plt.subplots(layout="constrained")
    for alpha in alphas:
        theta_opt, history = optimise(lambda th, R: psi_H(th[0], R), V_H,
         lambda th, R: -np.linalg.norm(R, axis=-1)[..., None],
         theta0=[0.5], x0=rng.normal(0, 2, (1000, 3)), rng=rng,
         step=lambda th: 1/th[0], alpha=1.0)
        thetas = np.append(history[:, 0], theta_opt)
        ax.semilogy(np.arange(len(thetas)), np.abs(thetas - 1), "o-", ms=3,
                    label=rf"$\alpha = {alpha}$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$|\theta - 1|$")
    ax.legend()
    fig.savefig("plots/optimisation.png", dpi=200)



def plot_projected_density(samples, theta, L=4, bins=150):
    x = samples[..., 0].ravel()
    y = samples[..., 1].ravel()
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-L, L], [-L, L]], density=True)
    
    xc = 0.5 * (xedges[1:] + xedges[:-1])
    yc = 0.5 * (yedges[1:] + yedges[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    exact = n_xy(theta, X, Y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    H, xe, ye = projected_density(samples[..., 0].ravel(), samples[..., 1].ravel())
    # exact n_xy on the bin centres as before, then:
    im = plot_density(ax1, H, xe, ye, "VMC samples", vmax=exact.max())
    plot_density(ax2, exact, xe, ye, "Exact", vmax=exact.max())
    fig.colorbar(im, ax=[ax1, ax2], label=r"$n_{xy}(x, y)$")
    fig.savefig("plots/projected_density.png", dpi=200)


if __name__ == "__main__":
    main()