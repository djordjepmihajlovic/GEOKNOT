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
    my_knot_dir = "data/"
    ab_propagator = np.loadtxt(os.path.join(my_knot_dir, fname_sts))
    ab_propagator = ab_propagator.reshape(-1, Nbeads, Nbeads)
    return ab_propagator

@njit(parallel = True)
def compute_feynman_diagram(ab_propagator, samples):
    '''
    Calculate the non-trivalent 'abelian propagator' Feynman diagrams
    '''

    feynman_data_1 = np.zeros((samples, 1)) # [[0, 1]]
    feynman_data_2 = np.zeros((samples, 2)) # [[0, 1, 2, 3], [0, 2, 1, 3]]
    feynman_data_3 = np.zeros((samples, 5)) # [[0, 1, 2, 3, 4, 5],[0, 1, 2, 4, 3, 5],[0, 1, 2, 5, 3, 4],[0, 2, 1, 4, 3, 5],[0, 3, 1, 4, 2, 5]]
    N = 152

    for idy in prange(0, samples): # samples
        for i in range(0, N):
            for j in range(0, N):
                if i>j:
                    feynman_data_1[idy][0] += ab_propagator[idy][i,j]

                    for k in range(0, N):
                        if j>k:
                            for l in range(0, N):
                                if k>l:
                                    feynman_data_2[idy][0] += ab_propagator[idy][i, j] * ab_propagator[idy][k, l]
                                    feynman_data_2[idy][1] += ab_propagator[idy][i, k] * ab_propagator[idy][j, l]

                                    for m in range(0, N):
                                        if l>m:
                                            for n in range(0, N):
                                                if n>m:
                                                    feynman_data_3[idy][0] += ab_propagator[idy][i, j]*ab_propagator[idy][k, l]*ab_propagator[idy][m, n]
                                                    feynman_data_3[idy][1] += ab_propagator[idy][i, j]*ab_propagator[idy][k, m]*ab_propagator[idy][l, n]
                                                    feynman_data_3[idy][2] += ab_propagator[idy][i, j]*ab_propagator[idy][k, n]*ab_propagator[idy][l, m]
                                                    feynman_data_3[idy][3] += ab_propagator[idy][i, k]*ab_propagator[idy][j, m]*ab_propagator[idy][l, n]
                                                    feynman_data_3[idy][4] += ab_propagator[idy][i, l]*ab_propagator[idy][j, m]*ab_propagator[idy][k, n]

    return feynman_data_1, feynman_data_2, feynman_data_3



def main():
    knots = ["smallntk", "smalltk"]
    set_num_threads(1)
    for x in knots:
        propagators = load_propagator(x, 152) # this is quite slow
        print("Gauge propagator loaded")
        print("Calculating possible feynman diagrams...")
        start_time = time.time()
        v1, v2, v3 = compute_feynman_diagram(propagators, len(propagators))
        print(time.time()-start_time)
        np.savetxt(f'samples/feynman_diagram_{x}_1.csv', v1[:,0])
        np.savetxt(f'samples/feynman_diagram_{x}_21.csv', v2[:,0])
        np.savetxt(f'samples/feynman_diagram_{x}_22.csv', v2[:,1])
        np.savetxt(f'samples/feynman_diagram_{x}_31.csv', v3[:,0])
        np.savetxt(f'samples/feynman_diagram_{x}_32.csv', v3[:,1])
        np.savetxt(f'samples/feynman_diagram_{x}_33.csv', v3[:,2])
        np.savetxt(f'samples/feynman_diagram_{x}_34.csv', v3[:,3])
        np.savetxt(f'samples/feynman_diagram_{x}_35.csv', v3[:,4])
    
main()