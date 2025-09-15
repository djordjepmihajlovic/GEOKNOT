import numpy as np
from numba import prange, njit
import math

'''
New version.
'''

@njit
def log_B_sym(alpha):
    return 2.0*math.lgamma(alpha) - math.lgamma(2.0*alpha)

@njit
def log_beta_pdf_sym(u, alpha, logB):
    # pdf(u) = u^(α-1) (1-u)^(α-1) / B(α,α)
    eps = 1e-15
    u = min(max(u, eps), 1.0 - eps)  # guard logs
    return (alpha-1.0)*math.log(u) + (alpha-1.0)*math.log1p(-u) - logB

@njit
def inv_beta_pdf_weight(u_s, u_t, u_u, alpha, logB):
    # returns 1 / (p(u_s)p(u_t)p(u_u)) in a stable way
    lp = (log_beta_pdf_sym(u_s, alpha, logB) +
          log_beta_pdf_sym(u_t, alpha, logB) +
          log_beta_pdf_sym(u_u, alpha, logB))
    return math.exp(-lp)

@njit
def log_mix_pdf(u, a_in, a_end, lam, logB_in, logB_end):
    # log[(1-lam)*Beta(a_in) + lam*Beta(a_end)] via log-sum-exp
    li = math.log1p(-lam) + log_beta_pdf_sym(u, a_in, logB_in)
    le = math.log(lam)     + log_beta_pdf_sym(u, a_end, logB_end)
    m = li if li > le else le
    return m + math.log(math.exp(li-m) + math.exp(le-m))

@njit
def sample_u_mixture(a_in, a_end, lam):
    if np.random.random() < lam:
        return np.random.beta(a_end, a_end)
    else:
        return np.random.beta(a_in, a_in)

@njit()
def tot_length(ring1):
    L = 0
    n = len(ring1)
    for i in range(0, n):
        L+= np.linalg.norm(ring1[(i+1)%n] - ring1[i])
    
    return L


@njit(parallel=True)
def compute_trivalent_feynman_diagram(ring1, discretization=5, a_in=100, a_end=0.7, lam=0.3):
    n1 = ring1.shape[0]

    matrix = np.zeros((n1, n1, n1))
    val = 0.0
    L = tot_length(ring1)

    for s in range(n1):
        for t in range(s):
            for u in prange(t):
                matrix[s, t, u] = I(ring1, s, t, u, L, discretization, a_in, a_end, lam)
    
    val = np.sum(matrix)
               
    return matrix, val

@njit()
def I(ring1, s, t, u, L, samples, a_in, a_end, lam):

    tot = 0
    n = len(ring1)
    
    eps = 1e-10

    su1 = (s + 1) % n
    tu1 = (t + 1) % n
    uu1 = (u + 1) % n

    dxs = (ring1[su1] - ring1[s])
    dxt = (ring1[tu1] - ring1[t])
    dxu = (ring1[uu1] - ring1[u])

    ds = np.linalg.norm(dxs)/L
    dt = np.linalg.norm(dxt)/L
    du = np.linalg.norm(dxu)/L

    dxs *= (1/ds)
    dxt *= (1/dt)
    dxu *= (1/du)

    logB_in  = log_B_sym(a_in)
    logB_end = log_B_sym(a_end)

    for m in range(samples):
        ss = sample_u_mixture(a_in, a_end, lam)
        tt = sample_u_mixture(a_in, a_end, lam)
        uu = sample_u_mixture(a_in, a_end, lam)

        lps = log_mix_pdf(ss, a_in, a_end, lam, logB_in, logB_end)
        lpt = log_mix_pdf(tt, a_in, a_end, lam, logB_in, logB_end)
        lpu = log_mix_pdf(uu, a_in, a_end, lam, logB_in, logB_end)
        w = math.exp(-(lps + lpt + lpu))

        rs = ring1[s]+(ring1[su1] - ring1[s])*ss
        rt = ring1[t]+(ring1[tu1] - ring1[t])*tt
        ru = ring1[u]+(ring1[uu1] - ring1[u])*uu

        st = rs - rt
        us = ru - rs
        tu = rt - ru

        ts = -st
        su = -us
        ut = -tu

        nst = np.linalg.norm(st)
        nus = np.linalg.norm(us)
        ntu = np.linalg.norm(tu)
        nut = np.linalg.norm(ut)
        nsu = np.linalg.norm(su)

        if nst < eps or nus < eps or ntu < eps:
            continue

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

        if np.abs(denom_2) < eps:
            continue

        else:
            val_21 /= denom_2
            tot += val_1*val_21*w

    return (tot/samples)*(ds*dt*du)

#########################################################################################

# import numpy as np
# from numba import prange, njit

# '''
# New version.
# '''

# @njit()
# def tot_length(ring1):
#     L = 0
#     n = len(ring1)
#     for i in range(0, n):
#         L+= np.linalg.norm(ring1[(i+1)%n] - ring1[i])
    
#     return L


# @njit(parallel=True)
# def compute_trivalent_feynman_diagram(ring1, discretization=5):
#     n1 = ring1.shape[0]

#     matrix = np.zeros((n1, n1, n1))
#     val = 0.0
#     L = tot_length(ring1)

#     for s in range(n1):
#         for t in range(s):
#             for u in prange(t):
#                 matrix[s, t, u] = I(ring1, s, t, u, L, discretization)
    
#     val = np.sum(matrix)
               
#     return matrix, val

# @njit()
# def I(ring1, s, t, u, L, samples):

#     tot = 0
#     n = len(ring1)
    
#     eps = 1e-10

#     sm1 = (s - 1) % n
#     tm1 = (t - 1) % n
#     um1 = (u - 1) % n
#     su1 = (s + 1) % n
#     tu1 = (t + 1) % n
#     uu1 = (u + 1) % n

#     dxs = (ring1[su1] - ring1[s])
#     dxt = (ring1[tu1] - ring1[t])
#     dxu = (ring1[uu1] - ring1[u])

#     ds = np.linalg.norm(dxs)/L
#     dt = np.linalg.norm(dxt)/L
#     du = np.linalg.norm(dxu)/L

#     dxs *= (1/ds)
#     dxt *= (1/dt)
#     dxu *= (1/du)

#     for m in range(samples):
#         alpha = 200
#         ss = np.random.beta(alpha, alpha)
#         tt = np.random.beta(alpha, alpha)
#         uu = np.random.beta(alpha, alpha)
#         rs = ring1[s]+(ring1[su1] - ring1[s])*ss
#         rt = ring1[t]+(ring1[tu1] - ring1[t])*tt
#         ru = ring1[u]+(ring1[uu1] - ring1[u])*uu

#         st = rs - rt
#         us = ru - rs
#         tu = rt - ru

#         ts = -st
#         su = -us
#         ut = -tu

#         nst = np.linalg.norm(st)
#         nus = np.linalg.norm(us)
#         ntu = np.linalg.norm(tu)
#         nut = np.linalg.norm(ut)
#         nsu = np.linalg.norm(su)

#         if nst < eps or nus < eps or ntu < eps:
#             continue

#         num_1 = nst + nus - ntu
#         denom_1 = nst * nut * nsu

#         val_1 = num_1/denom_1
#         val_21 = 0

#         denom_2 = (nst * nus) + np.dot(ts, us)
#         for i in range(0, 3):
#             for j in range(0, 3):
#                 for k in range(0, 3):
#                     prod = dxs[i]*dxt[j]*dxu[k]
#                     delik = delts = delut = 0
#                     if i == j:
#                         delts = ts[k]
#                     if i == k:
#                         delik = su[j]
#                     if j == k:
#                         delut = ut[i]
#                     num_21 = prod * (delts + delik + delut)
#                     val_21 += num_21 

#         val_21 /= denom_2


#         if np.abs(denom_2) < eps:
#             tot += 0

#         else:
#             tot += val_1*val_21

#     return (tot/samples)*(ds*dt*du)

