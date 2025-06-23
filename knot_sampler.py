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

def wang_landau_sampling(partition, oriented, knot_type, writhe_bins, entang_bins, sub_samples, f_init=math.e, flatness_crit=0.99):
    '''
    Need a way to randomly implement levels of energy checking to get samples that are highly writhed
    Set no. bins = no. sub_samples, then we want each bin to be filled w exactly one sample.
    '''
    ### Idea, pass the aimed writhe (range per bin) and entanglement (range per bin) as arguments into BFACF and pivot evolvers.
    ### This is much more efficient than randomly evolving and hoping for a sample to fall into the right bin.
    ### Another cool idea could be to enforce change in knot type by selectively evolving components of the knot.
    ### Say if we get a knot with correct writhe bin but wrong type -> perturb into correct type.
    ### Would need to find a way of locating the right components to perturb.

    g = np.zeros((writhe_bins, entang_bins)) # 2d matrix of log probs
    H = np.zeros((writhe_bins, entang_bins)) # 2d matrix of histogram counts
    f = f_init
    pivot_lag = 5000
    BFACF_lag = 20000

    # 2.) Implement dynamical logic to switch between pivot and BFACF based on the current state.
    # 3.) Implement saving knot as correct type if generated.
    # 4.) Implement reduction in writhe or entanglement if value is above the aimed range.
    # 5.) writhe_range and entang_range should be passed as arguments to the sampling function.

    writhe_range = (0, 35)
    entang_range = (500, 3000)

    writhe_edges = np.linspace(*writhe_range, writhe_bins + 1)
    entang_edges = np.linspace(*entang_range, entang_bins + 1)
    
    writhe_ranges = [(writhe_edges[i], writhe_edges[i + 1]) for i in range(len(writhe_edges) - 1)]
    entang_ranges = [(entang_edges[i], entang_edges[i + 1]) for i in range(len(entang_edges) - 1)]
    
    current_state = oriented
    current_writhe = 0
    current_entang = 0
    sampled_states = []

    filled_bins = set()

    for i in range(len(writhe_ranges)):
        for j in range(len(entang_ranges)):
            print(f"Sampling writhe {writhe_ranges[i]} and entanglement {entang_ranges[j]}")

            completed = False
            while not completed:

                ## Currently brute force switching.
                ## 1.) Also, we would like to sample across grid of writhe and entanglement bins, the 
                ## evolution constraints should be dictated by the bin we are sampling.

                proposed_state = pivot(current_state, timesteps=pivot_lag, knot=knot_type, aimed_range=(writhe_ranges[i], entang_ranges[j]))
                proposed_state, proposed_writhe, proposed_entang = BFACF(proposed_state, timesteps=BFACF_lag, aimed_range=(writhe_ranges[i], entang_ranges[j]))
                # proposed_state = pivot(proposed_state, timesteps=pivot_lag, knot=knot_type, aimed_range=(writhe_ranges[i], entang_ranges[j]))
                # proposed_state, proposed_writhe, proposed_entang = BFACF(proposed_state, timesteps=BFACF_lag, aimed_range=(writhe_ranges[i], entang_ranges[j]))

                topo = Q_invariant(proposed_state, 'Uq(sl2)').alexander_polynomial_hash(knot_type) 
                print(f"writhe: {proposed_writhe}, range: {writhe_ranges[i]}")
                print(f"entanglement: {proposed_entang}, range: {entang_ranges[j]}")
                if topo == True:
                    current_writhe = proposed_writhe
                    current_entang = proposed_entang
                    if min(writhe_ranges[i])<=current_writhe<=max(writhe_ranges[i]):
                        if min(entang_ranges[j])<=current_entang<=max(entang_ranges[j]):
                            # sampled_states.append(proposed_state)
                            completed = True
                            print(f"Sampled state with writhe {current_writhe} and entanglement {current_entang} in bin [{writhe_ranges[i]}, {entang_ranges[j]}]")

                            min_x = min(p[0] for p in proposed_state)
                            min_y = min(p[1] for p in proposed_state)
                            min_z = min(p[2] for p in proposed_state)

                            max_x = max(p[0] for p in proposed_state) + 1
                            max_y = max(p[1] for p in proposed_state) + 1
                            max_z = max(p[2] for p in proposed_state) + 1

                            offset_x = abs(min_x) if min_x < 0 else 0
                            offset_y = abs(min_y) if min_y < 0 else 0
                            offset_z = abs(min_z) if min_z < 0 else 0

                            # array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
                            array = np.zeros((max_x + offset_x, max_y + offset_y, max_z + offset_z), dtype=np.float64)
                            for (x, y, z), val in proposed_state.items():
                                # array[x, y, z] = val
                                array[x + offset_x, y + offset_y, z + offset_z] = val

                            coords = np.argwhere(array>0)
                            coord_dat = [(array[p[0], p[1], p[2]], p[0], p[1], p[2]) for p in coords]

                            elements = sorted(coord_dat, key=lambda x: x[0])
                            
                            joggle_scale = 1e-2
                            np.random.seed(42)
                            elements_jiggled = [np.array([el[1:4] for el in elements], dtype=float) +
                            np.random.normal(scale=joggle_scale, size=(len(elements), 3))]

                            new_coord = [tuple(row) for row in elements_jiggled[0]]
                            w = [wx[0] for wx in elements]
                            new_coord_w = [(w[idx],) + coord for idx, coord in enumerate(new_coord)]

                            np.savetxt(f'samples/{knot_type}_{partition}_{int(writhe_ranges[i])}_{int(entang_ranges[j])}.csv', new_coord_w, delimiter=",", fmt='%.5f')
                            sampled_states.append([partition, int((writhe_ranges[i][0]+writhe_ranges[i][1])/2), int((entang_ranges[j][0]+entang_ranges[j][1])/2)])
            
    return sampled_states

def process_wang_landau(args):

    i, oriented, knot_type, writhe_bins, rog_bins, sub_samples = args

    sampled_states = wang_landau_sampling(i, oriented, knot_type, writhe_bins, rog_bins, sub_samples)

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

    # for i, evolved in enumerate(sampled_results):
    #     # Save coordinates
    #     # "offset_*" fixes the issue of negative coordinates being saved with PBC.

    #     for j, state in enumerate(evolved):
    #         min_x = min(p[0] for p in state)
    #         min_y = min(p[1] for p in state)
    #         min_z = min(p[2] for p in state)

    #         max_x = max(p[0] for p in state) + 1
    #         max_y = max(p[1] for p in state) + 1
    #         max_z = max(p[2] for p in state) + 1

    #         offset_x = abs(min_x) if min_x < 0 else 0
    #         offset_y = abs(min_y) if min_y < 0 else 0
    #         offset_z = abs(min_z) if min_z < 0 else 0

    #         # array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    #         array = np.zeros((max_x + offset_x, max_y + offset_y, max_z + offset_z), dtype=np.float64)
    #         for (x, y, z), val in state.items():
    #             # array[x, y, z] = val
    #             array[x + offset_x, y + offset_y, z + offset_z] = val

    #         coords = np.argwhere(array>0)
    #         coord_dat = [(array[p[0], p[1], p[2]], p[0], p[1], p[2]) for p in coords]

    #         elements = sorted(coord_dat, key=lambda x: x[0])
    #         print(elements)
            
    #         joggle_scale = 1e-2
    #         np.random.seed(42)
    #         elements_jiggled = [np.array([i[1:4] for i in elements], dtype=float) +
    #         np.random.normal(scale=joggle_scale, size=(len(elements), 3))]

    #         new_coord = [tuple(row) for row in elements_jiggled[0]]
    #         w = [i[0] for i in elements]
    #         new_coord_w = [(w[idx],) + coord for idx, coord in enumerate(new_coord)]

    #         np.savetxt(f'samples/{knot_type}_{i}_{j}.csv', new_coord_w, delimiter=",", fmt='%.5f')

    writhe_dist = []
    entang_dist = []
    max_writhe = 0
    max_entang = 0
    min_writhe = 500
    min_entang = 500

    for i in sampled_results:
        for j in i:


    # for i in range(samples):
    #     for j in range(sub_samples):

            print(f'Checking: {j[0]},{j[1]},{j[2]}')
            file = np.loadtxt(f'samples/{knot_type}_{j[0]}_{j[1]}_{j[2]}.csv', delimiter=',', dtype=int)
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
                print (f"max wr: {max_writhe}, {j}")
            if writhe < min_writhe:
                min_writhe = writhe
                print(f"min wr: {min_writhe}, {j}")
        
            entang_dict = {}
            iter = np.nditer(array, flags=['multi_index'])
            for val in iter:
                if val != 0:
                    entang_dict[iter.multi_index] = val.item()

            entang = long_range_entanglement(entang_dict)

            if entang > max_entang:
                max_entang = entang
                print (f"max entang: {max_entang}, {j}")

            if entang < min_entang:
                min_entang = entang
                print (f"min entang: {min_entang}, {j}")

            writhe_dist.append(writhe)
            entang_dist.append(entang)

    plt.hist2d(writhe_dist, entang_dist, bins=(len(writhe_dist), len(entang_dist)), density=True, cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel('Writhe')
    plt.ylabel('Entanglement')
    plt.title(f'Writhe vs Entanglement Heatmap for {knot_type}')
    plt.savefig(f'samples/writhe_entang_heatmap_{knot_type}.png')
    plt.clf()

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