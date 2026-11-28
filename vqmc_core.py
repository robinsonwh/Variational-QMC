import numpy as np
from functools import partial

def hot2(f, x, h):
    return (f(x+h) - 2*f(x) + f(x-h))/h**2

def hot4(f, x, h):
    return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h))/(12*h**2)

def hot6(f, x, h):
    return (2*f(x + 3*h) - 27*f(x + 2*h) + 270*f(x + h) - 490*f(x) + 270*f(x - h) - 27*f(x - 2*h) + 2*f(x - 3*h))/(180*h**2)

def local_energy(f, V, x, h, deriv=hot4):
    return -0.5 * deriv(f, x, h) / f(x) + V(x)

def metropolis(f, x0, n_steps, step, burn_in=0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.array(x0, dtype=float)
    fx = f(x)
    samples = np.empty((n_steps - burn_in, *x.shape))
    n_accepted = 0
    for t in range(n_steps):
        x_new = x + step * rng.normal(size=x.shape)          # symmetric proposal
        f_new = f(x_new)
        accept = rng.random(len(x)) < (f_new / fx)**2       # p_acc = min(1, ratio)
        x[accept] = x_new[accept]                           # rejected walkers stay put
        fx[accept] = f_new[accept]
        if t >= burn_in:
            samples[t - burn_in] = x                        # record every step, accepted or not
            n_accepted += accept.sum()
    return samples, n_accepted / ((n_steps - burn_in) * len(x))

def laplacian(f, R, h, deriv=hot4):
    return sum(deriv(lambda s: f(R + s * e), 0.0, h) for e in np.eye(R.shape[-1]))

def local_energy_nd(f, V, R, h, deriv=hot4):
    return -0.5 * laplacian(f, R, h, deriv) / f(R) + V(R)

def optimise(psi, V, dlogpsi, theta0, x0, rng, step, alpha=0.5, max_step=0.2,
             max_iter=200, n_steps=300, burn_in=50, h=2.5e-4,
             tol=1e-8, k_sigma=2.0, patience=3):
    #Notes: 12.8
    #Brief: 16
    theta = np.array(theta0, dtype=float)
    x = np.array(x0, dtype=float)
    history, n_ok = [], 0
    for it in range(max_iter):
        f = partial(psi, theta)
        samples, acc = metropolis(f, x, n_steps, step(theta), burn_in, rng)
        x = samples[-1].copy()
        E = local_energy_nd(f, V, samples, h)
        H = E.mean()
        g_w = 2 * ((E - H)[..., None] * dlogpsi(theta, samples)).mean(axis=0)  # per walker
        grad = g_w.mean(axis=0)
        grad_err = g_w.std(axis=0, ddof=1) / np.sqrt(len(g_w))
        history.append(np.concatenate([theta, [H], grad, grad_err]))
        delta = -np.asarray(alpha) * grad
        delta *= min(1.0, max_step / np.max(np.abs(delta)))     # cap the step size
        theta = theta + delta
        n_ok = n_ok + 1 if np.all(np.abs(grad) < np.maximum(tol, k_sigma * grad_err)) else 0
        if n_ok >= patience:
            break
    return theta, np.array(history)

def final_energy(f, V, x0, rng, step, n_steps=5500, burn_in=500, h=2.5e-4):
    samples, acc = metropolis(f, x0, n_steps, step, burn_in, rng)
    E = local_energy_nd(f, V, samples, h)
    walker_means = E.mean(axis=0)
    H = E.mean()
    H_err = walker_means.std(ddof=1) / np.sqrt(len(walker_means))
    return H, H_err, E.std(), samples