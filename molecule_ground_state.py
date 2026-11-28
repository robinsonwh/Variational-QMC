import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.optimize import curve_fit

from systems import psi_H2, V_H2, V_morse, dlogpsi_H2, timed
from vqmc_core import optimise, final_energy
from density_plotting import plot_density, projected_density


def main():
    Path("plots").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    E_single, E_single_err = np.loadtxt("results/hydrogen.csv", delimiter=",", skiprows=1)  # from Section 3
    separations, energies, energy_errors, theta_stars = molecule_scan(rng)

    (D, a, r0), (D_err, a_err, r0_err) = fit_morse(E_single, separations, energies, energy_errors)
    print(f"r0 = {r0:.3f} +/- {r0_err:.3f}  (experiment ~1.4)")
    print(f"D  = {D:.4f} +/- {D_err:.4f}  (experiment ~0.17)")
    plot_morse(separations, energies, energy_errors, E_single, D, a, r0)

@timed
def ground_state(d, theta0, rng, max_iter=40, n_avg=10):
    """Optimise theta at separation d, then evaluate <H> at fixed theta* with fresh walkers."""
    q1, q2 = np.array([-d/2, 0, 0]), np.array([d/2, 0, 0])
    psi = lambda th, R: psi_H2(th, q1, q2, R)
    V = lambda R: V_H2(q1, q2, R)
    x0 = lambda: rng.normal(0, 1, (1000, 6)) + np.r_[q1, q2]   # electron 1 near q1, 2 near q2

    _, history = optimise(psi, V, lambda th, R: dlogpsi_H2(th, q1, q2, R),
                          theta0=theta0, x0=x0(), rng=rng, step=lambda th: 0.6,
                          alpha=[0.5, 4.0, 4.0], max_iter=max_iter)
    theta_star = history[-n_avg:, :3].mean(axis=0)      # average out the drift in theta3

    H, H_err, E_std, samples = final_energy(lambda R: psi(theta_star, R), V, x0=x0(),
                                            rng=rng, step=0.6, n_steps=2500, burn_in=500)
    return theta_star, H, H_err, samples

def fit_morse(E_single, separations, energies, energy_errors):
    popt, pcov = curve_fit(lambda r, D, a, r0: V_morse(r, D, a, r0, E_single),
                           separations, energies, sigma=energy_errors,
                           absolute_sigma=True, p0=[0.17, 1.0, 1.4])

    return popt, np.sqrt(np.diag(pcov))

def molecule_scan(rng):
    separations = np.concatenate([np.linspace(0.6, 2.2, 9),    # denser near the minimum
                              np.linspace(2.6, 4.6, 5)])   # tail towards dissociation
    energies = np.empty(len(separations))
    energy_errors = np.empty(len(separations))
    theta_opts = np.empty((len(separations), 3))

    theta0 = np.array([1.0, 0.4, 0.6])
    for i, d in enumerate(separations):
        q1, q2 = np.array([-d/2, 0, 0]), np.array([d/2, 0, 0])
        theta_opt, history = optimise(lambda th, R: psi_H2(th, q1, q2, R), lambda R: V_H2(q1, q2, R),
         lambda th, R: dlogpsi_H2(th, q1, q2, R),
         theta0=[1.0, 0.4, 0.6], x0=rng.normal(0, 1, (1000, 6)) + np.r_[q1, q2],
         rng=rng, step=lambda th: 0.6, alpha=[0.5, 4.0, 4.0])
        H, H_err, E_std, samples = final_energy(
            lambda R: psi_H2(theta_opt, q1, q2, R), lambda R: V_H2(q1, q2, R),
            x0=rng.normal(0, 1, (1000, 6)) + np.r_[q1, q2], rng=rng, step=0.6)
        print(f"d = {d:.2f}: theta* = {np.array2string(theta_opt, precision=4)}, "
            f"<H> = {H:.5f} +/- {H_err:.1e}")
        energies[i], energy_errors[i], theta_opts[i] = H, H_err, theta_opt
        plot_projected_density_H2(samples, d)       # one per separation
        theta0 = theta_opt                          # warm start for the next separation

        np.savetxt("results/h2_scan.csv", separations=separations[:i+1], energies=energies[:i+1],
             energy_errors=energy_errors[:i+1], theta_opts=theta_opts[:i+1])

def plot_projected_density_H2(samples, d, bins=150):
    x = np.concatenate([samples[..., 0].ravel(), samples[..., 3].ravel()])
    y = np.concatenate([samples[..., 1].ravel(), samples[..., 4].ravel()])
    L = max(4.0, d/2 + 3.0)                              # keep both atoms in frame
    H, xedges, yedges = projected_density(x, y, L=L, bins=bins, n_electrons=2)

    fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
    im = plot_density(ax, H, xedges, yedges, rf"H$_2$, $d = {d:.2f}$")
    ax.plot([-d/2, d/2], [0, 0], "c+", ms=10, mew=1.5, label="nuclei")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label=r"$n_{xy}(x, y)$")
    fig.savefig(f"plots/density_H2_d{d:.2f}.png", dpi=200)
    plt.close(fig)

def plot_morse(separations, energies, energy_errors, E_single, D, a, r0):
    r = np.linspace(separations.min(), separations.max(), 300)
    fig, ax = plt.subplots(layout="constrained")
    ax.errorbar(separations, energies, yerr=energy_errors, fmt="o", ms=4, capsize=3, label="VMC")
    ax.plot(r, V_morse(r, D, a, r0, E_single), "k-", lw=1, label="Morse fit")
    ax.axhline(2 * E_single, color="grey", ls="--", lw=1, label=r"$2E_\mathrm{single}$")
    ax.set_xlabel("nuclear separation $d$")
    ax.set_ylabel(r"$\langle H \rangle$")
    ax.legend()
    fig.savefig("plots/morse_fit.png", dpi=200)
    plt.close(fig)  

if __name__ == "__main__":
    main()