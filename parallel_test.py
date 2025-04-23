import numpy as np
from knot_init import *
from knot_evolution import *
from quantum_knot_invs import *
import ctypes
import os
import time
import networkx as nx
from multiprocessing import Pool, Array
import random

# Global shared array view for workers
global_arr = None

def worker_init(shared_base, shape, dtype):
    """
    Worker initializer: map the shared Array buffer into a NumPy array once.
    """
    global global_arr
    # Create a NumPy view over the shared buffer
    global_arr = np.frombuffer(shared_base, dtype=dtype).reshape(shape)


def process_independent_set(points):
    """
    Perform BFACF moves on all points in one independent set.
    Operates in-place on global_arr.
    """
    array = global_arr
    for coord in points:
        x, y, z = coord
        # Compute current energy
        p1 = points_on_axis(array, np.array([np.pi,  np.e/2,  np.sqrt(2)/2]))
        p2 = points_on_axis(array, np.array([np.pi, -np.e/2,  np.sqrt(2)/2]))
        p3 = points_on_axis(array, np.array([np.pi,  np.e/2, -np.sqrt(2)/2]))
        p4 = points_on_axis(array, np.array([np.pi, -np.e/2, -np.sqrt(2)/2]))
        old_energy = lattice_writhe_Cimasoni(array,
                                            projections_111=p1,
                                            projections_1m11=p2,
                                            projections_11m1=p3,
                                            projections_1m1m1=p4)
        # Propose move
        new_edge = find_new(array, (x, y, z))
        # new_edge has 4 entries: (x', y', z', ...)
        if np.array_equal(new_edge, np.array([-1, -1, -1, -1], dtype=new_edge.dtype)):
            continue
        # Test move on a copy
        temp = array.copy()
        temp[x, y, z] = 0
        i0, i1, i2 = new_edge[:3].astype(int)
        temp[i0, i1, i2] += array[x, y, z]
        # Validate geometry
        if check_verticies(temp) < -2:
            continue
        # Compute new energy
        q1 = points_on_axis(temp, np.array([np.pi,  np.e/2,  np.sqrt(2)/2]))
        q2 = points_on_axis(temp, np.array([np.pi, -np.e/2,  np.sqrt(2)/2]))
        q3 = points_on_axis(temp, np.array([np.pi,  np.e/2, -np.sqrt(2)/2]))
        q4 = points_on_axis(temp, np.array([np.pi, -np.e/2, -np.sqrt(2)/2]))
        new_energy = lattice_writhe_Cimasoni(temp,
                                            projections_111=q1,
                                            projections_1m11=q2,
                                            projections_11m1=q3,
                                            projections_1m1m1=q4)
        # Metropolis acceptance (T=1)
        if metropolis_acceptance(old_energy, new_energy, 1):
            # Commit move to shared array
            array[x, y, z] = 0
            array[i0, i1, i2] += temp[i0, i1, i2]
    return None


def build_correlation_graph(array):
    indices = np.argwhere(array > 0)
    G = nx.Graph()
    for idx, p in enumerate(indices):
        G.add_node(idx, coords=tuple(p))
    for i, p1 in enumerate(indices):
        for j in range(i+1, len(indices)):
            if np.linalg.norm(p1 - indices[j], ord=1) <= 2:
                G.add_edge(i, j)
    return G


def find_independent_sets(G):
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    sets = {}
    for node, col in coloring.items():
        sets.setdefault(col, []).append(G.nodes[node]['coords'])
    return list(sets.values())


def main():
    # Initialize lattice knot

    start_time = time.time()
    state_space = np.zeros((100, 100, 100))
    knot = Knot('3_1', state_space)
    unknot = orient(knot.initialize())

    # Shared Array: ctypes array for buffer sharing
    shape = unknot.shape
    dtype = unknot.dtype
    size = unknot.size
    c_type = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    shared_base = Array(c_type, size, lock=False)
    shared_main = np.frombuffer(shared_base, dtype=dtype).reshape(shape)
    np.copyto(shared_main, unknot)

    # Persistent Pool
    n_procs = os.cpu_count() or 1
    pool = Pool(processes=n_procs,
                initializer=worker_init,
                initargs=(shared_base, shape, dtype))

    try:
        steps = 0
        max_steps = 1000
        points_per_round = 10

        while steps < max_steps:
            print(f"Progress: {steps}/{max_steps}")
            G = build_correlation_graph(shared_main)
            indep_sets = find_independent_sets(G)

            # Flatten all independent sets into a list of coords
            all_points = [pt for s in indep_sets for pt in s]
            if len(all_points) > points_per_round:
                sampled_points = random.sample(all_points, points_per_round)
            else:
                sampled_points = all_points

            # Wrap into single "chunk" for map
            # pool.map expects a list of iterables, so we wrap into one chunk
            chunksize = 1
            pool.map(process_independent_set, [sampled_points], chunksize=chunksize)

            steps += len(sampled_points)

    finally:
        pool.close()
        pool.join()

    end_time = time.time() - start_time
    print(end_time)

if __name__ == '__main__':
    main()

# 38.7s (no pooling)
# 