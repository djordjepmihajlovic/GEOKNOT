import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

'''
This module calculates quantum invariants of a knot to determine knot type (not preserved in pivot algorithm).
Takes in state space array and returns a Q_invariant object.
'''

class Q_invariant:
    def __init__(self, array, q_group):
        '''
        Knot array of state space, and specified quantum group.
        '''
        self.array = array
        self.q_group = q_group

    def flat(self):
        '''
        Flattens the state space array on 2D plane (yz).
        '''
        indicies = np.argwhere(self.array>0)

        for i in indicies:
            projected_vectors_yz = self.array[:, i[1], i[2]] # list of yz plane projections to create planar diagram.
            '''
            There will be an issue! What about n>2 n-tangle crossings? Need to think...
            '''
            elements = np.argwhere(projected_vectors_yz > 0)
            if len(elements) > 1:
                print(f'{len(elements)}-tangle crossing detected.')
                for j in range(len(elements)):
                    print(f'{projected_vectors_yz[elements[j]]}.')