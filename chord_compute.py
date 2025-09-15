import numpy as np
from numba import prange, njit

'''
Chord compute code to compute the chord components of a knot with the relevant FM compactification...
_____________________________________________________________________________________________________________
Writhe computations a.la. Klenin et. al: Computation of writhe in modeling of supercoiled DNA
With additional Fulton-MacPherson compactification for the configurational space integrals
The above is essential to let us compute higher order interactions with much higher accuracy. (Vassiliev invs)
'''

@njit()
def compute_chord(ring1, ring2):
    '''
    General computation over all segments.
    '''

    matrix = np.zeros((ring1.shape[0],ring2.shape[0]))
    # Loop on the first ring
    for i in prange(ring1.shape[0]):
        # Loop on the second ring
        for j in prange(ring2.shape[0]):
            matrix[i,j] = compute_kernel_chord(ring1, ring2, i, j)
    return matrix

@njit()
def FM_compactification(one, two, three, four, rho_factor=1e-3):
    '''
    Fulton MacPherson compactification. Should motivate the theory a bit I think.
    '''

    length_one = np.linalg.norm(two-one)
    length_two = np.linalg.norm(four-three)
    # Either a==d or b==c (can't have the edge points of a meet c for example - according to the formulation here).
    rho = rho_factor * max(1e-12, min(length_one, length_two))
    tau_1 = (two-one)/length_one # this is whats determining the signage below btw
    tau_2 = (four-three)/length_two

    if np.allclose(two, three):
        two_compact = two - rho * tau_1
        three_compact = three + rho * tau_2
        return one, two_compact, three_compact, four
    
    if np.allclose(one, four):
        one_compact = one + rho * tau_1
        four_compact = four - rho * tau_2
        return one_compact, two, three, four_compact
    
    else:
        return one, two, three, four
    
@njit()
def vec_cross(a, b):
    n = np.cross(a, b)
    norm_n = np.linalg.norm(n)
    if norm_n > 1e-12:
        return n / norm_n
    return np.array((0.0, 0.0, 0.0), dtype=np.float64)

@njit()
def clip1n1(x):
    return np.minimum(1.0, np.maximum(-1.0, x))

@njit()
def compute_kernel_chord(ring1, ring2, i, j):
    '''
    Klenin computation of writhe
    '''

    P = ring1.shape[0]
    one = ring1[np.mod(i-1,ring1.shape[0]),:]
    three = ring2[np.mod(j-1,ring2.shape[0]),:]
    two = ring1[np.mod(i,ring1.shape[0]),:]
    four = ring2[np.mod(j,ring2.shape[0]),:]

    if i == j: 
        return 0.0
    
    if (j-i)%P in (1, P-1):
        # Build Fulton-Macpherson compactification on edges which share vertices.
        # https://www.jstor.org/stable/pdf/2946631.pdf
        one, two, three, four = FM_compactification(one, two, three, four)

    # Standard Klenin techniques
    r12=two-one
    r34=four-three
    r23=three-two
    r13=three-one
    r14=four-one
    r24=four-two

    n1 = vec_cross(r13, r14)
    n2 = vec_cross(r14, r24)
    n3 = vec_cross(r24, r23)
    n4 = vec_cross(r23, r13)

    n1n2=clip1n1(np.dot(n1,n2))
    n2n3=clip1n1(np.dot(n2,n3))
    n3n4=clip1n1(np.dot(n3,n4))
    n4n1=clip1n1(np.dot(n4,n1))

    triple = float(np.dot(np.cross(r34,r12),r13))
    sign = 0.0 if abs(triple) < 1e-18 else np.sign(triple)

    omega = (np.arcsin(n1n2) + np.arcsin(n2n3) + np.arcsin(n3n4) + np.arcsin(n4n1)) * sign
    return omega

