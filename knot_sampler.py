import matplotlib.pyplot as plt
from argparse import ArgumentParser
from knot_evolution_hash import *
from knot_reader import *
import time
from multiprocessing import Pool
import os
import numpy as np
import csv
from knot_invs import Q_invariant
from pathlib import Path

def _sampling(partition, oriented, knot_type, fn_1, fn_2, bins_1, bins_2, plot):
    '''
    Need a way to randomly implement levels of energy checking to get samples that are highly writhed
    Set no. bins = no. sub_samples, then we want each bin to be filled w exactly one sample.
    '''

    # make folder for saving:
    sample_dir = Path(f'samples/{knot_type}')
    sample_dir.mkdir(parents=True, exist_ok=True)

    initial_length = len(oriented)

    pivot_lag = 5000
    BFACF_lag = 10000

    fn1_range = (0, 25)
    fn2_range = (0, 1500)
    # entang_range = (750, 2750) # maybe make convergence slightly faster

    fn1_edges = np.linspace(*fn1_range, bins_1 + 1)
    fn2_edges = np.linspace(*fn2_range, bins_2 + 1)
    print(fn1_edges)
    
    fn1_ranges = [(fn1_edges[i], fn1_edges[i + 1]) for i in range(len(fn1_edges) - 1)]
    fn2_ranges = [(fn2_edges[i], fn2_edges[i + 1]) for i in range(len(fn2_edges) - 1)]
    
    current_state = oriented
    current_fn1 = 0
    current_fn2 = 0
    sampled_states = []

    instance_per_range = []

    for i in range(len(fn1_ranges)):
        for j in range(len(fn2_ranges)):
            print(f"Sampling fn1 {fn1_ranges[i]} and fn2 {fn2_ranges[j]}")
            # check if file already exists
            
            if os.path.exists(f'samples/{knot_type}/{knot_type}_{partition}_{int((fn1_ranges[i][0]+fn1_ranges[i][1])/2)}_{int((fn2_ranges[j][0]+fn2_ranges[j][1])/2)}.csv'):
                print(f"File already exists for fn1 {fn1_ranges[i]} and fn2 {fn2_ranges[j]}, skipping...")
                continue

            instance = 0
            completed = False
            while not completed:
                instance += 1

                ## Currently brute force switching.
                ## 1.) Also, we would like to sample across grid of writhe and entanglement bins, the 
                ## evolution constraints should be dictated by the bin we are sampling.

                proposed_state = pivot(current_state, timesteps=pivot_lag, knot=knot_type, aimed_range=(fn1_ranges[i], fn2_ranges[j]), fn_1=fn_1, fn_2=fn_2)
                proposed_state, proposed_fn1, proposed_fn2 = BFACF(proposed_state, timesteps=BFACF_lag, aimed_range=(fn1_ranges[i], fn2_ranges[j]), fn_1=fn_1, fn_2=fn_2)
 
                print(f"function 1: {proposed_fn1}, range: {fn1_ranges[i]}")
                print(f"entanglement: {proposed_fn2}, range: {fn2_ranges[j]}")

                current_fn1 = proposed_fn1
                current_fn2 = proposed_fn2
                if min(fn1_ranges[i])<=current_fn1<=max(fn1_ranges[i]):
                    if min(fn2_ranges[j])<=current_fn2<=max(fn2_ranges[j]):
                        # sampled_states.append(proposed_state)

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
                        
                        joggle_scale = 1
                        np.random.seed(42)
                        elements_jiggled = [np.array([el[1:4] for el in elements], dtype=float) +
                        np.random.normal(scale=joggle_scale, size=(len(elements), 3))]

                        new_coord = [tuple(row) for row in elements_jiggled[0]]
                        w = [wx[0] for wx in elements]
                        new_coord_w = [(w[idx],) + coord for idx, coord in enumerate(new_coord)]

                        # included int(...) for entanglement 

                        ##########################################################
                        state = {tuple(coord[1:]): coord[0] for coord in read_coord(new_coord_w)}
                        topo = Q_invariant(state, 'Uq(sl2)').alexander_polynomial_hash(knot_type, joggle=False) 

                        if topo == True and len(new_coord_w) == initial_length:
                        
                        ###########################################################

                            if plot == True:
                                plot_3d_line(new_coord_w)

                            'Final topology check after jiggling coords off lattice.'
                            completed = True
                            print(f"Sampled state with function 1: {current_fn1} and function 2: {current_fn2} in bin [{fn1_ranges[i]}, {fn2_ranges[j]}]")
                            np.savetxt(f'samples/{knot_type}/{knot_type}_{partition}_{int((fn1_ranges[i][0]+fn1_ranges[i][1])/2)}_{int((fn2_ranges[j][0]+fn2_ranges[j][1])/2)}.csv', new_coord_w, delimiter=",", fmt='%.5f')
                            sampled_states.append([partition, int((fn1_ranges[i][0]+fn1_ranges[i][1])/2), int((fn2_ranges[j][0]+fn2_ranges[j][1])/2)])
                            instance_per_range.append([int((fn1_ranges[i][0]+fn1_ranges[i][1])/2), int((fn2_ranges[j][0]+fn2_ranges[j][1])/2), instance])
            
    return sampled_states, instance_per_range

def process_sampler(args):

    i, oriented, knot_type, fn_1, fn_2, bins_1, bins_2, plot = args
    sampled_states, instance_per_range = _sampling(i, oriented, knot_type, fn_1, fn_2, bins_1, bins_2, plot)
    return sampled_states, instance_per_range


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

    fn_1 = fns[0]
    fn_2 = fns[1]

    bins_1 = sub_samples
    bins_2 = sub_samples

    args_list = [(i, oriented, knot_type, fn_1, fn_2, bins_1, bins_2, plot) for i in range(samples)]

    with Pool(processes=num_processes) as pool: 
        results = pool.map(process_sampler, args_list)  # Parallelize over samples

    run_time = time.time() - start_time
    print(run_time)

    sampled_results = [r[0] for r in results]          # list of all sampled states
    times = [r[1] for r in results]

    with open(f'samples/{knot_type}/time_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Writhe Bin', 'Entanglement Bin', 'Instances'])  # Header row
        for time_entry in times:
            writer.writerow(time_entry)


    writhe_dist = []
    entang_dist = []
    max_writhe = 0
    max_entang = 0
    min_writhe = 500
    min_entang = 500

    for i in sampled_results:
        for j in i:

            print(f'Checking: {j[0]},{j[1]},{j[2]}')
            file = np.loadtxt(f'samples/{knot_type}/{knot_type}_{j[0]}_{j[1]}_{j[2]}.csv', delimiter=',', dtype=int)
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
            # entang = average_curvature(entang_dict)

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
    plt.title(f'Writhe vs Entanglement Heatmap for non-trivial knots')
    plt.savefig(f'samples/writhe_entang_heatmap_{knot_type}.png')
    plt.clf()

    plt.hist(writhe_dist, bins=100, density=True, alpha=0.5, label='Writhe Distribution')
    plt.xlabel('Writhe')
    plt.ylabel('Density')
    plt.title(f'Writhe Distribution for non-trivial knots')
    plt.savefig(f'samples/writhe_dist_samples_{knot_type}.png')
    plt.clf()
    
    plt.hist(entang_dist, bins=100, density=True, alpha=0.5, label='Entanglement Distribution')
    plt.xlabel('Entanglement')
    plt.ylabel('Density')
    plt.title(f'Entanglement Distribution for non-trivial knots')
    plt.savefig(f'samples/entang_dist_samples_{knot_type}.png')
    plt.clf()

par = ArgumentParser()
'''
    Lets us specify arguments for the code.
'''

par.add_argument("-d", "--discretization", type=int, default=100, help="Discretization of state space y,z axis.")
par.add_argument("-k", "--knot", type=str, default='0_1', help="Knot type.")
par.add_argument("-s", "--sampler", type=str, default='Metropolis', help="Sampling method.")
par.add_argument("-no", "--no_samples", type=int, default=1, help="Number of samples to generate per geometric requirement.")
par.add_argument("-sub", "--no_sub_samples", type=int, default=2, help="Number of sub-samples per process.")
par.add_argument("-np", "--no_processes", type=int, default=os.cpu_count(), help="Number of cores to run code on.")
par.add_argument("-fns", "--functions", type=list, default=[None, None], help="Geometric state space metrics to explore. Default = writhe and entanglement (flagged by 'False')")
par.add_argument("-distr", "--state_space_distr", type=list, default=[(0, 25), (0, 1500)], help="End-to-end state space sample distribution.")
par.add_argument("-plot", "--plot", type=bool, default=False, help="Plot output (best used if sampling a single knot).")

args = par.parse_args()

if __name__ == "__main__":
    
    discretization = args.discretization
    knot_type = args.knot
    sampler = args.sampler
    samples = args.no_samples
    num_processes = args.no_processes
    sub_samples = args.no_sub_samples
    plot = args.plot
    distr = args.state_space_distr
    fns = args.functions

    main()