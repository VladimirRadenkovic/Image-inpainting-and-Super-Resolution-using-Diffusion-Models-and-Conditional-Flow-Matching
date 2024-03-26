import numpy as np
from functools import partial

a = 0.3
b = 1.0
xi = 0.9

def R_center(xi, N):
    return np.eye(N) - (xi / N) * np.ones((N, N))

def R_sum(b, N):
    # Matrix with ones in diagonal, twos in off-diagonal 1, three in off-diagonal 2, etc.
    R = np.eye(N)
    for i in range(1, N):
        R += np.diag(b ** (np.ones(N - i)* i), k=-i)
    return R

def R_init(b, N):
    R = np.diag(np.ones(N))
    R[0,0] = 1. / np.sqrt(1.-b**2 + 1e-4)
    return R

def R(N: int, *, a: float = 0.3, b: float = 1.0, xi: float = 0.9):
    return a * R_center(xi, N) @ R_sum(b, N) #@ R_init(b, N)  # NOTE: exluded R_init bc of numerical issues

Rfunc = partial(R, a=0.3, b=1.0, xi=0.9)