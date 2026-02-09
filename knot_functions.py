import numpy as np
from numba import njit

# Example feature: ROG

def radius_of_gyration(array_dict):
    '''
    You can then set fn_1 = radius_of_gyration as an argument when running code:

        python knot_sampler.py -distr [(0, 3), (500, 1000)] -fns [radius_of_gyration, False]

    This will now sample a knot with radius of gyration values between (0, 3) 
    and default=long_range_entanglement values between (500, 1000)
    '''

    # build dense array from hash map
    max_x = max(p[0] for p in array_dict) + 1
    max_y = max(p[1] for p in array_dict) + 1
    max_z = max(p[2] for p in array_dict) + 1
    dense = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (x, y, z), val in array_dict.items():
        dense[x, y, z] = val

    indicies = np.argwhere(dense > 0)
    center_of_mass = np.mean(indicies, axis=0)
    return np.sqrt(np.mean(np.sum((indicies - center_of_mass)**2, axis=1)))