import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as mcolors  
from knot_init import *
from defunct.knot_evolution import lattice_writhe_Klenin

# knot = np.loadtxt('/Users/s1910360/Desktop/min_wrentang_0_1.csv', delimiter=',', dtype=np.float64)
# knot = np.loadtxt('examples/config_0_1.csv', delimiter=',', dtype=np.float64)

def read_array(knot):
    state = np.zeros((100, 100, 100), dtype=np.int64)
    for i in knot:
        state[round(i[1])][round(i[2])][round(i[3])] = i[0] # snap to integer values 
    return state

def read_coord(knot):
    coord_list = [(float(i[0]), float( i[1]), float(i[2]), float(i[3])) for i in knot]
    coord_list = sorted(coord_list, key=lambda x: x[0])

    # for i in range(len(coord_list)):
    #     # fix pdbs
    #     if float(coord_list[i][1]) > 0:
    #         coord_list[i] = (coord_list[i][0], float(coord_list[i][1]) - 100, coord_list[i][2], coord_list[i][3])
    #     if float(coord_list[i][2]) > 0:
    #         coord_list[i] = (coord_list[i][0], coord_list[i][1], float(coord_list[i][2]) - 100, coord_list[i][3])
    #     if float(coord_list[i][3]) > 0:
    #         coord_list[i] = (coord_list[i][0], coord_list[i][1], coord_list[i][2], float(coord_list[i][3]) - 100)

    return coord_list

# im = lattice_writhe_Klenin(read_coord(knot))
# print(np.sum(im))
# plt.imshow(im)
# plt.colorbar()
# plot_3d_line(read_coord(knot))