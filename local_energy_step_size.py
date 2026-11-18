import numpy as np
import matplotlib.pyplot as plt

def H(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    elif n == 2:
        return 4*x**2 - 2
    elif n == 3:
        return 8*x**3 - 12*x
    elif n == 4:
        return 16*x**4 - 48*x**2 + 12
    elif type(n) != int:
        raise TypeError("n is not an integer.")
    else:
        raise ValueError("n must be an integer between 0 and 4 inclusive.")

def psi(n, x):
    return H(n, x) * np.exp(-x**2 / 2)

def hot2(n, x, h):
    return (psi(n, x+h) - 2*psi(n, x) + psi(n, x-h))/h**2

def hot4(n, x, h):
    return (-psi(n, x + 2*h) + 16*psi(n, x + h) - 30*psi(n, x) + 16*psi(n, x - h) - psi(n, x - 2*h))/(12*h**2)

def hot6(n, x, h):
    return (2*psi(n, x + 3*h) - 27*psi(n, x + 2*h) + 270*psi(n, x + h) - 490*psi(n, x) + 270*psi(n, x - h) - 27*psi(n, x - 2*h) + 2*psi(n, x - 3*h))/(180*h**2)

def E_l(deriv, n, x, h):
    return ((-1/(2*psi(n, x))) * deriv(n, x, h) + 0.5*x**2)

def weighted_error(deriv, n, h):
    #rho-weighted mean of |E_l - E_n| over the grid x.
    x = np.linspace(-8, 8, 100) + 1e-7
    rho = psi(n, x)**2
    rho /= np.trapezoid(rho, x) # normalise the PDF on the grid
    h = np.asarray(h, dtype=float)[..., None]  # (len(h), 1) so it broadcasts against x -> (len(h), len(x))
    return np.trapezoid(rho * np.abs(E_l(deriv, n, x, h) - (n + 0.5)), x, axis=-1)


hs = np.logspace(-6, 0, 61)
for n in range(5):
    plt.figure()
    plt.loglog(hs, weighted_error(hot2, n, hs), label="HOT2")
    plt.loglog(hs, weighted_error(hot4, n, hs), label="HOT4")
    plt.loglog(hs, weighted_error(hot6, n, hs), label="HOT6")
    plt.title(rf"Weighted error for $\psi_{n}(x)$")
    plt.xlabel("h")
    plt.ylabel(r"$\rho$-weighted mean of $|E_L - E_n|$")
    plt.legend()
    plt.savefig(f'plots/weighted_error_{n}')
fig, axs = plt.subplots(3, 2)

plt.show()