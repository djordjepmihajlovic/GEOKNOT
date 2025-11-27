import numpy as np
import matplotlib.pyplot as plt
from knot_init import *
from knot_evolution_hash import lattice_writhe_Klenin
# from quantum_knot_invs import *

# knot = np.loadtxt('/Users/s1910360/Desktop/ntk/K_9_7_2821.csv', delimiter=',', dtype=np.float64)
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

# coords = read_array(knot)
# # Check topology
# proposed_state = {tuple(coord[1:]): coord[0] for coord in read_coord(knot)}
# topo = Q_invariant(proposed_state, 'Uq(sl2)').alexander_polynomial_hash('0_1', joggle=False)
# coords = read_coord(knot)
# print(knot)
# cross_mat, len_cross = compute_crossing_matrix(knot=knot[:,1:], projection=np.array([0.3, 0.4, 0.7], dtype=np.float32), dim=len(knot))
# plt.imshow(cross_mat)
# plt.show()
# plot_3d_line(coords)