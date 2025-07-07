import numpy as np
from knot_init import *
from defunct.knot_evolution import lattice_writhe_Klenin
from quantum_knot_invs import *
import csv
import matplotlib.pyplot as plt


# make range (as in sampler)

def load_func(partitions, writhe_bins, entang_bins):

    writhe_range = (0, 35)
    entang_range = (500, 3000)

    writhe_edges = np.linspace(*writhe_range, writhe_bins + 1)
    entang_edges = np.linspace(*entang_range, entang_bins + 1)

    writhe_ranges = [(writhe_edges[i], writhe_edges[i + 1]) for i in range(len(writhe_edges) - 1)]
    entang_ranges = [(entang_edges[i], entang_edges[i + 1]) for i in range(len(entang_edges) - 1)]
    states = []

    number = 0

    for i in partitions:
        for j in range(len(writhe_ranges)):
            for k in range(len(entang_ranges)):
                knot = np.loadtxt(f'/Users/s1910360/Desktop/ntk/K_{i}_{int((writhe_ranges[j][0]+writhe_ranges[j][1])/2)}_{int((entang_ranges[k][0]+entang_ranges[k][1])/2)}.csv', delimiter=',', dtype=np.float64)
                proposed_state = {tuple(coord[1:]): coord[0] for coord in read_coord(knot)}
                topo = Q_invariant(proposed_state, 'Uq(sl2)').alexander_polynomial_hash('0_1', joggle=False)
                if not topo:
                    print('Knot type consistent...')
                    # im = lattice_writhe_Klenin(read_coord(knot))
                else:
                    print(f'Inconsistent knot found! @ {i}, {int((writhe_ranges[j][0]+writhe_ranges[j][1])/2)}, {int((entang_ranges[k][0]+entang_ranges[k][1])/2)}')
                    states.append([i, int((writhe_ranges[j][0]+writhe_ranges[j][1])/2), int((entang_ranges[k][0]+entang_ranges[k][1])/2)])
                    number +=1
    
    print(number)

    return states


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


writhe_bins = entang_bins = 7
partitions = np.arange(0, 10)

samples = load_func(partitions, writhe_bins, entang_bins)

with open('samples/wrong_labels_ntk.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Writhe', 'Partition', 'Wr_bin', 'Ent_bin'])  # Header row
    for entry in samples:
        writer.writerow(entry)

# data = []
# colors = []
# with open('samples/wrong_labels_tk.csv', mode='r') as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip the header row
#     for row in reader:
#         writhe = float(row[0])  # First value per row
#         partition = int(row[1])  # Second value per row
#         data.append(writhe)
#         colors.append(partition)

# plt.hist(data)
# plt.show()