import numpy as np
from numba import prange, njit

'''
New version.
'''

@njit()
def tot_length(ring1):
    L = 0
    n = len(ring1)
    for i in range(0, n):
        L+= np.linalg.norm(ring1[(i+1)%n] - ring1[(i-1)%n])
    
    return L


@njit(parallel=True)
def compute_trivalent_feynman_diagram(ring1):
    n1 = ring1.shape[0]

    matrix = np.zeros((n1, n1, n1))
    val = 0.0
    L = tot_length(ring1)

    for s in range(n1):
        for t in range(s):
            for u in prange(t):
                matrix[s, t, u] += I(ring1, s, t, u, L)
    
    val = np.sum(matrix)
               
    return matrix, val

@njit()
def I(ring1, s, t, u, L):
    n = len(ring1)
    
    eps = 1e-10

    sm1 = (s - 1) % n
    tm1 = (t - 1) % n
    um1 = (u - 1) % n
    su1 = (s + 1) % n
    tu1 = (t + 1) % n
    uu1 = (u + 1) % n

    dxs = (ring1[su1] - ring1[sm1])
    dxt = (ring1[tu1] - ring1[tm1])
    dxu = (ring1[uu1] - ring1[um1])

    ds = np.linalg.norm(dxs)/L
    dt = np.linalg.norm(dxt)/L
    du = np.linalg.norm(dxu)/L

    dxs *= (0.5/ds)
    dxt *= (0.5/dt)
    dxu *= (0.5/du)


    st = ring1[s] - ring1[t]
    us = ring1[u] - ring1[s]
    tu = ring1[t] - ring1[u]

    ts = ring1[t] - ring1[s]
    su = ring1[s] - ring1[u]
    ut = ring1[u] - ring1[t]

    nst = np.linalg.norm(st)
    nus = np.linalg.norm(us)
    ntu = np.linalg.norm(tu)
    nut = np.linalg.norm(ut)
    nsu = np.linalg.norm(su)

    if nst < eps or nus < eps or ntu < eps:
        return 0.0

    num_1 = nst + nus - ntu
    denom_1 = nst * nut * nsu

    val_1 = num_1/denom_1
    val_21 = 0

    denom_2 = (nst * nus) + np.dot(ts, us)
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                prod = dxs[i]*dxt[j]*dxu[k]
                delik = delts = delut = 0
                if i == j:
                    delts = ts[k]
                if i == k:
                    delik = su[j]
                if j == k:
                    delut = ut[i]
                num_21 = prod * (delts + delik + delut)
                val_21 += num_21 

    val_21 /= denom_2

    if np.abs(denom_2) < eps:
        return 0.0

    return val_1*val_21*(ds*dt*du)