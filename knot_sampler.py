import matplotlib.pyplot as plt
from argparse import ArgumentParser
from knot_evolution_hash import *
from knot_reader import *
from quantum_knot_invs import *
import time
from multiprocessing import Pool
import os
import math
import numpy as np

'''
Its always useful to write down ideas:)
So I want to sample across the space of knots - wang landau just simulates without bias and picks
to fulfill criteria, that is ok however i dont think it'll reach distant configs that often.
Whereas explicitly biasing toward high entanglement will flatten highly entangled configurations.
Could do this for a range?
Also; if an 0_1 or 3_1 change; save them as respective knot.
'''

def wang_landau_sampling(oriented, knot_type, writhe_bins, entang_bins, sub_samples, f_init=math.e, flatness_crit=0.9):
    '''
    Need a way to randomly implement levels of energy checking to get samples that are highly writhed
    Set no. bins = no. sub_samples, then we want each bin to be filled w exactly one sample.
    '''

    g = np.zeros((writhe_bins, entang_bins)) # 2d matrix of log probs
    H = np.zeros((writhe_bins, entang_bins)) # 2d matrix of histogram counts
    f = f_init
    pivot_lag = 5000
    BFACF_lag = 10000

    writhe_range = (0, 1000)
    entang_range = (0, 1000)

    writhe_edges = np.linspace(*writhe_range, writhe_bins + 1)
    entang_edges = np.linspace(*entang_range, entang_bins + 1)

    def get_bin_indices(writhe, entang):
        writhe_idx = np.digitize(writhe, writhe_edges) - 1
        entang_idx = np.digitize(entang, entang_edges) - 1

        writhe_idx = np.clip(writhe_idx, 0, writhe_bins - 1)
        entang_idx = np.clip(entang_idx, 0, entang_bins - 1)
        return writhe_idx, entang_idx
    
    current_state = oriented
    current_writhe = 0
    current_entang = 0
    sampled_states = []

    all_bins = [(i, j) for i in range(writhe_bins) for j in range(entang_bins)]
    filled_bins = set()

    # while len(sampled_states) < sub_samples:
    for target_bin in all_bins:
        while target_bin not in filled_bins:

            proposed_state = pivot(current_state, timesteps=pivot_lag, knot=knot_type)
            proposed_state, proposed_writhe, proposed_entang = BFACF(proposed_state, timesteps=BFACF_lag)

            current_bin = get_bin_indices(current_writhe, current_entang)
            proposed_bin = get_bin_indices(proposed_writhe, proposed_entang)

            if g[proposed_bin] <= g[current_bin] or np.random.rand() < 0.1: #math.exp(g[current_bin] - g[proposed_bin]):

                topo = Q_invariant(proposed_state, 'Uq(sl2)').alexander_polynomial_hash(knot_type) 
                if topo == True:
                    current_writhe = proposed_writhe
                    current_entang = proposed_entang
                    current_bin = proposed_bin
                    sampled_states.append(proposed_state)

                    filled_bins.add(proposed_bin)

            g[current_bin] += math.log(f)
            H[current_bin] += 1

            if np.min(H) > flatness_crit * np.mean(H):
                H.fill(0)
                f = math.sqrt(f)
        
    return sampled_states

def process_wang_landau(args):

    i, oriented, knot_type, writhe_bins, rog_bins, sub_samples = args

    sampled_states = wang_landau_sampling(oriented, knot_type, writhe_bins, rog_bins, sub_samples)

    return sampled_states


def main():
    '''
    Sampling knots according to the autocorrelation findings in knot_data_analysis.py
    Note: pivot decorrelates knots very quickly.
    We keep BFACF to take decorrelated samples and move them towards high writhe configurations.
    '''

    state_space = np.zeros((discretization, discretization, discretization))
    knot = Knot(knot_type, state_space)
    unknot = knot.initialize()

    print('Hashing...')
    array_dict = {}
    iter = np.nditer(unknot, flags=['multi_index'])
    for val in iter:
        if val != 0:
            array_dict[iter.multi_index] = val.item()

    print('Orienting...')
    oriented = orient(array_dict)
    for key, value in oriented.items():
        if value == 1.0:
            oriented[key] = 1  # orientation float issue

    start_time = time.time()

    writhe_bins = sub_samples
    rog_bins = sub_samples

    args_list = [(i, oriented, knot_type, writhe_bins, rog_bins, sub_samples) for i in range(samples)]

    with Pool(processes=num_processes) as pool: 
        results = pool.map(process_wang_landau, args_list)  # Parallelize over `samples`

    run_time = time.time() - start_time
    print(run_time)

    sampled_results = [r for r in results]          # list of all sampled states

    for i, evolved in enumerate(sampled_results):
        # Save coordinates
        for j, state in enumerate(evolved):
            max_x = max(p[0] for p in state) + 1
            max_y = max(p[1] for p in state) + 1
            max_z = max(p[2] for p in state) + 1
            array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
            for (x, y, z), val in state.items():
                array[x, y, z] = val

            coords = np.argwhere(array>0)
            coord_dat = [(array[p[0], p[1], p[2]], p[0], p[1], p[2]) for p in coords]

            elements = sorted(coord_dat, key=lambda x: x[0])
            print(elements)
            
            joggle_scale = 1e-2
            np.random.seed(42)
            elements_jiggled = [np.array([i[1:4] for i in elements], dtype=float) +
            np.random.normal(scale=joggle_scale, size=(len(elements), 3))]

            new_coord = [tuple(row) for row in elements_jiggled[0]]
            w = [i[0] for i in elements]
            new_coord_w = [(w[idx],) + coord for idx, coord in enumerate(new_coord)]

            np.savetxt(f'samples/{knot_type}_{i}_{j}.csv', new_coord_w, delimiter=",", fmt='%.5f')

    writhe_dist = []
    entang_dist = []
    max_writhe = 0
    max_entang = 0
    min_writhe = 500
    min_entang = 500

    for i in range(samples):
        for j in range(sub_samples):

            print(f'Checking: {i},{j}')
            file = np.loadtxt(f'samples/{knot_type}_{i}_{j}.csv', delimiter=',', dtype=int)
            load = read_array(file)
            array = load.copy() # this will load an integer rounded version
            no_points = len(np.argwhere(array)>0)
            projections_111 = points_on_axis(array, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
            projections_1m11 = points_on_axis(array, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
            projections_11m1 = points_on_axis(array, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
            projections_1m1m1 = points_on_axis(array, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))

            writhe = lattice_writhe_Cimasoni(array, no_points,
                                                projections_111=projections_111, 
                                                projections_11m1=projections_1m11,
                                                projections_1m11=projections_11m1,
                                                projections_1m1m1=projections_1m1m1)
            
            if writhe > max_writhe:
                max_writhe = writhe
                print (f"max wr: {max_writhe}, {i}")
            if writhe < min_writhe:
                min_writhe = writhe
                print(f"min wr: {min_writhe}, {i}")
        
            entang_dict = {}
            iter = np.nditer(array, flags=['multi_index'])
            for val in iter:
                if val != 0:
                    entang_dict[iter.multi_index] = val.item()

            entang = long_range_entanglement(entang_dict)

            if entang > max_entang:
                max_entang = entang
                print (f"max entang: {max_entang}, {i}")

            if entang < min_entang:
                min_entang = entang
                print (f"min entang: {min_entang}, {i}")


            writhe_dist.append(writhe)
            entang_dist.append(entang)

    plt.hist(writhe_dist, bins=100, density=True, alpha=0.5, label='Writhe Distribution')
    plt.xlabel('Writhe')
    plt.ylabel('Density')
    plt.title(f'Writhe Distribution for {knot_type}')
    plt.savefig(f'samples/writhe_dist_samples_{knot_type}.png')
    plt.clf()
    
    plt.hist(entang_dist, bins=100, density=True, alpha=0.5, label='Entanglement Distribution')
    plt.xlabel('Entanglement')
    plt.ylabel('Density')
    plt.title(f'Entanglement Distribution for {knot_type}')
    plt.savefig(f'samples/entang_dist_samples_{knot_type}.png')
    plt.clf()

par = ArgumentParser()
'''
    Lets us specify arguements for the code.
'''

par.add_argument("-d", "--discretization", type=int, default=100, help="Discretization of state space y,z axis.")
par.add_argument("-k", "--knot", type=str, default='0_1', help="Knot type.")
par.add_argument("-s", "--sampler", type=str, default='Metropolis', help="Sampling method.")
par.add_argument("-no", "--no_samples", type=int, default=10, help="Number of decorrelated samples to generate.")
par.add_argument("-sub", "--no_sub_samples", type=int, default=10, help="Number of sub-samples per process.")
par.add_argument("-np", "--no_processes", type=int, default=os.cpu_count(), help="Number of cores to run code on.")

args = par.parse_args()

if __name__ == "__main__":
    discretization = args.discretization
    knot_type = args.knot
    sampler = args.sampler
    samples = args.no_samples
    num_processes = args.no_processes
    sub_samples = args.no_sub_samples

    main()