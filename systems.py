import numpy as np

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