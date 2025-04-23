import numpy as np
from knot_init import *
from knot_evolution_hash import *
import os
import time
import networkx as nx
from multiprocessing import Pool


def process_independent_set(args):
    array_dict, points, old_energy = args
    updated_dict = dict(array_dict)  # shallow copy

    for coord in points:
        if coord not in updated_dict:
            continue

        new_edge = find_new(updated_dict, coord)

        if new_edge == (-1, -1, -1, -1):
            continue

        old_val = updated_dict[coord]
        del updated_dict[coord]
        updated_dict[new_edge[:3]] = updated_dict.get(new_edge[:3], 0) + old_val

        if check_verticies(updated_dict) < -2:
            continue

        # Convert dict to dense array for energy computation
        max_x = max(p[0] for p in updated_dict) + 1
        max_y = max(p[1] for p in updated_dict) + 1
        max_z = max(p[2] for p in updated_dict) + 1
        update_array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
        for (i, j, k), val in updated_dict.items():
            update_array[i, j, k] = val

        p1 = points_on_axis(update_array, np.array([np.pi,  np.e/2,  np.sqrt(2)/2]))
        p2 = points_on_axis(update_array, np.array([np.pi, -np.e/2,  np.sqrt(2)/2]))
        p3 = points_on_axis(update_array, np.array([np.pi,  np.e/2, -np.sqrt(2)/2]))
        p4 = points_on_axis(update_array, np.array([np.pi, -np.e/2, -np.sqrt(2)/2]))

        new_writhe_energy = lattice_writhe_Cimasoni(update_array,
                                             projections_111=p1,
                                             projections_1m11=p2,
                                             projections_11m1=p3,
                                             projections_1m1m1=p4)
        
        new_energy = crumple(updated_dict)

        if metropolis_acceptance(old_energy, new_energy, 0.001):
            array_dict = dict(updated_dict)
            old_energy = new_energy

    return array_dict, old_energy


def build_correlation_graph(array_dict):
    G = nx.Graph()
    indices = list(array_dict.keys())
    for idx, p in enumerate(indices):
        G.add_node(idx, coords=p)
    for i, p1 in enumerate(indices):
        for j in range(i + 1, len(indices)):
            if sum(abs(a - b) for a, b in zip(p1, indices[j])) <= 2:
                G.add_edge(i, j)
    return G


def find_independent_sets(G):
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    sets = {}
    for node, col in coloring.items():
        sets.setdefault(col, []).append(G.nodes[node]['coords'])
    return list(sets.values())


def dense_to_dict(arr):
    return {(x, y, z): arr[x, y, z] for x, y, z in zip(*np.nonzero(arr))}


def main():

    # Initialize lattice knot
    state_space = np.zeros((100, 100, 100))
    knot = Knot('3_1', state_space).initialize()
    array_dict = dense_to_dict(knot)
    array_dict = orient(array_dict)

    unknot_init = dict(array_dict)
    max_x = max(p[0] for p in unknot_init) + 1
    max_y = max(p[1] for p in unknot_init) + 1
    max_z = max(p[2] for p in unknot_init) + 1
    unknot_init = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (i, j, k), val in array_dict.items():
        unknot_init[i, j, k] = val

    # Compute initial energy
    p1 = points_on_axis(unknot_init, np.array([np.pi,  np.e/2,  np.sqrt(2)/2]))
    p2 = points_on_axis(unknot_init, np.array([np.pi, -np.e/2,  np.sqrt(2)/2]))
    p3 = points_on_axis(unknot_init, np.array([np.pi,  np.e/2, -np.sqrt(2)/2]))
    p4 = points_on_axis(unknot_init, np.array([np.pi, -np.e/2, -np.sqrt(2)/2]))
    old_writhe_energy = lattice_writhe_Cimasoni(unknot_init, p1, p2, p3, p4)

    old_energy = crumple(array_dict)

    # Start multiprocessing pool
    n_procs = os.cpu_count() 
    pool = Pool(processes=n_procs)

    steps = 0
    max_steps = 10000

    start_time = time.time()
    try:
        while steps < max_steps:
            print(f"Progress: {steps}/{max_steps}")

            G = build_correlation_graph(array_dict)
            indep_sets = find_independent_sets(G)
            largest_set = max(indep_sets, key=len)

            # Send jobs to worker pool
            tasks = [(array_dict, largest_set, old_energy)]
            results = pool.map(process_independent_set, tasks)

            for new_dict, new_energy in results:
                array_dict = new_dict  
                old_energy = new_energy

            steps += len(largest_set)

    finally:
        pool.close()
        pool.join()

    print(f"Total time: {time.time() - start_time:.2f} seconds")

    max_x = max(p[0] for p in array_dict) + 1
    max_y = max(p[1] for p in array_dict) + 1
    max_z = max(p[2] for p in array_dict) + 1
    f_knot = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (i, j, k), val in array_dict.items():
        f_knot[i, j, k] = val

    plot_3d(f_knot)

if __name__ == '__main__':
    main()