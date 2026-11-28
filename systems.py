"""
Evaluates wavefunctions and potentials for all systems. Also times function calls.
"""
import numpy as np
import functools
import time
from scipy.special import k1

def timed(func):
    """Print how long each call to func takes."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - t0:.2f} s")
        return result
    return wrapper

def H(n, x):
    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer.")
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
    else:
        raise ValueError("n must be an integer between 0 and 4 inclusive.")

def psi_sho(n, x):
    return H(n, x) * np.exp(-x**2 / 2)

def V_sho(x): return 0.5 * x**2

def psi_H(theta, R):
    return np.exp(-theta * np.linalg.norm(R, axis=-1))

def V_H(R):
    return -1 / np.linalg.norm(R, axis=-1)

def dist(a, b):
    return np.linalg.norm(a - b, axis=-1)

def psi_H2(thetas, q1, q2, R):
    t1, t2, t3 = thetas
    r1, r2 = R[..., :3], R[..., 3:]
    orbital = (np.exp(-t1 * (dist(r1, q1) + dist(r2, q2)))
               + np.exp(-t1 * (dist(r1, q2) + dist(r2, q1))))
    jastrow = np.exp(-t2 / (1 + t3 * dist(r1, r2)))
    return orbital * jastrow

def V_H2(q1, q2, R):
    r1, r2 = R[..., :3], R[..., 3:]
    return (-1/dist(r1, q1) - 1/dist(r1, q2) - 1/dist(r2, q1) - 1/dist(r2, q2)
            + 1/dist(r1, r2) + 1/dist(q1, q2))

def V_morse(r, D, a, r0, E_single):
    return D*(1 - np.exp(-a*(r-r0)))**2 - D + 2*E_single

def dlogpsi_H2(theta, q1, q2, R):
    t1, t2, t3 = theta
    r1, r2 = R[..., :3], R[..., 3:]
    sA = dist(r1, q1) + dist(r2, q2)          # electron 1 at nucleus 1, electron 2 at nucleus 2
    sB = dist(r1, q2) + dist(r2, q1)          # electrons swapped
    A, B = np.exp(-t1 * sA), np.exp(-t1 * sB)
    r12 = dist(r1, r2)
    return np.stack([-(A*sA + B*sB) / (A + B),          # ∂/∂θ1
                     -1 / (1 + t3*r12),                  # ∂/∂θ2
                     t2 * r12 / (1 + t3*r12)**2],        # ∂/∂θ3
                    axis=-1)

def n_xy(theta, x, y):
    s = np.sqrt(x**2 + y**2)
    return 2 * theta**3 / np.pi * s * k1(2*theta*s)