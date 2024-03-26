# cython: warning=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow

def kabsch_alignment(np.ndarray[float, ndim=2, mode="c"] P, np.ndarray[float, ndim=2, mode="c"] Q):
    cdef np.ndarray[float, ndim=1, mode="c"] centroid_P, centroid_Q
    cdef np.ndarray[float, ndim=2, mode="c"] P_centered, Q_centered, C, V, W, U, P_aligned
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    C = np.dot(np.transpose(P_centered), Q_centered)
    V, S, W = np.linalg.svd(C)
    if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    P_aligned = np.dot(P_centered, U) 
    P_aligned += centroid_Q
    return P_aligned

def rmsd(np.ndarray[float, ndim=2, mode="c"] P, np.ndarray[float, ndim=2, mode="c"] Q):
    return sqrt(np.mean((P - Q)**2))