import os
import time
import numpy as np
from numba import njit, prange, set_num_threads

start_time = time.time()

def load_propagator(knot_type, Nbeads):
    '''
    Load the data from the knots database
    '''

    fname_sts = f"3DSignedWrithe_{knot_type}_small.dat.lp10.dat"
    my_knot_dir = "PyKnotData/data/"
    ab_propagator = np.loadtxt(os.path.join(my_knot_dir, fname_sts))
    ab_propagator = ab_propagator.reshape(-1, Nbeads, Nbeads)

    fname = f"3DSignedTri_small{knot_type}.npz"
    tri_propagator = np.load(fname)["tri_array"] # has kept structure

    return ab_propagator, tri_propagator

@njit(parallel = True)
def compute_trivalent_feynman_diagram(tri_propagator, ab_propagator, samples):
    '''
    Calculate mixed Feynman diagrams (3rd order)
    '''

    feynman_data = np.zeros((samples, 2)) # [[0, 1, 2, 3], [0, 2, 1, 3]]
    N = 152

    for idy in prange(0, samples): # samples
        for i in range(0, N):
            for j in range(0, N):
                if i>j:
                    for k in range(0, N):
                        if j>k:
                            for l in range(0, N):
                                if k>l:
                                    for m in range(0, N):
                                        if l>m:
                                            feynman_data[idy][0] += ab_propagator[idy][i, j]*tri_propagator[idy][k, l, m]
                                            feynman_data[idy][1] += ab_propagator[idy][i, k]*tri_propagator[idy][j, l, m]

    return feynman_data

def main():
    knots = ["smallntk", "smalltk"]
    set_num_threads(40)
    for x in knots:
        ab_propagator, tri_propagator = load_propagator(x, 152) # this is quite slow
        print("Gauge propagator loaded")
        print("Calculating possible feynman diagrams...")
        start_time = time.time()
        tri2 = compute_trivalent_feynman_diagram(ab_propagator, tri_propagator, len(ab_propagator))
        print(time.time()-start_time)
        np.savetxt(f'samples/feynman_diagram_{x}_t11.csv', tri2[:,0])
        np.savetxt(f'samples/feynman_diagram_{x}_t12.csv', tri2[:,1])
    
main()