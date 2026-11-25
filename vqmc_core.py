import numpy as np
import systems

def hot2(f, x, h):
    return (f(x+h) - 2*f(x) + f(x-h))/h**2

def hot4(f, x, h):
    return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h))/(12*h**2)

def hot6(f, x, h):
    return (2*f(x + 3*h) - 27*f(x + 2*h) + 270*f(x + h) - 490*f(x) + 270*f(x - h) - 27*psi(x - 2*h) + 2*psi(x - 3*h))/(180*h**2)

def local_energy(f, V, x, h, deriv=hot4):
    """E_l(x) = -(1/2) f''(x)/f(x) + V(x) for a 1D wavefunction f (eq. 13)."""
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
