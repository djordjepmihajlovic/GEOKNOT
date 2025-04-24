import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as mcolors  
from knot_init import *
from defunct.knot_evolution import lattice_writhe_Klenin

# knot = np.loadtxt('/Users/s1910360/Desktop/max_entang.csv', delimiter=',', dtype=int)

def read(knot):
    state = np.zeros((100, 100, 100), dtype=np.int64)
    for i in knot:
        state[i[1]][i[2]][i[3]] = i[0]
    return state

# im = lattice_writhe_Klenin(read(knot))
# print(np.sum(im))
# plot_3d_line(read(knot))
# plt.imshow(im)
# plt.show()