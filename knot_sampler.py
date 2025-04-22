import matplotlib.pyplot as plt
from argparse import ArgumentParser
from knot_evolution import *
from knot_reader import read
import time
from multiprocessing import Pool
import os

def process_sample(args):
    """
    Function to process a single sample in parallel.
    """
    i, unknot, knot_type, sampler = args 
    knot_copy = unknot.copy()
    pivot_lag = 500
    BFACF_lag = 1000

    # Perform pivot and BFACF
    knot_copy = pivot(knot_copy, pivot_lag, knot_type)
    knot_copy, g_w = BFACF(array=knot_copy, timesteps=BFACF_lag, sampler=sampler)

    # Save coordinates
    coords = np.argwhere(knot_copy > 0)
    coord_dat = [(knot_copy[i[0], i[1], i[2]], i[0], i[1], i[2]) for i in coords]
    output_file = f'samples/{knot_type}_{i}.csv'
    np.savetxt(output_file, coord_dat, delimiter=",", fmt='%d')


def main():
    '''
    Sampling knots according to the autocorrelation findings in knot_data_analysis.py
    Note: pivot decorrelates knots very quickly.
    We keep BFACF to take decorrelated samples and move them towards high writhe configurations.
    '''

    # global knot_type, sampler  # Declare global variables for use in `process_sample`

    # state_space = np.zeros((discretization, discretization, discretization))
    # knot = Knot(knot_type, state_space)
    # unknot = knot.initialize()
    # # orient knot
    # print('Orienting...')
    # unknot = orient(unknot)

    # start_time = time.time()
    # args_list = [(i, unknot, knot_type, sampler) for i in range(samples)]

    # with Pool(processes=num_processes) as pool:  # Use all available CPU cores
    #     results = pool.map(process_sample, args_list)  # Parallelize over `samples`

    # run_time = time.time() - start_time
    # print(run_time)

    writhe_dist = []
    r_o_g_dist = []

    for i in range(samples):
        print(f'Checking: {i}')
        file = np.loadtxt(f'samples/{knot_type}_{i}.csv', delimiter=',', dtype=int)
        load = read(file)
        array = load.copy()
        projections_111 = points_on_axis(array, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
        projections_1m11 = points_on_axis(array, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
        projections_11m1 = points_on_axis(array, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
        projections_1m1m1 = points_on_axis(array, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))

        writhe = lattice_writhe_Cimasoni(array, 
                                               projections_111=projections_111, 
                                               projections_11m1=projections_1m11,
                                               projections_1m11=projections_11m1,
                                               projections_1m1m1=projections_1m1m1)
        
        r_o_g = radius_of_gyration(array)
        writhe_dist.append(writhe)
        r_o_g_dist.append(r_o_g)
        

    plt.hist(writhe_dist)
    plt.savefig('samples/writhe_dist_samples')
    plt.clf()

    plt.hist(r_o_g_dist)
    plt.savefig('samples/r_o_g_dist_samples')
    plt.clf()

par = ArgumentParser()
'''
    Lets us specify arguements for the code.
'''

par.add_argument("-d", "--discretization", type=int, default=100, help="Discretization of state space y,z axis.")
par.add_argument("-k", "--knot", type=str, default='0_1', help="Knot type.")
par.add_argument("-s", "--sampler", type=str, default='Metropolis', help="Sampling method.")
par.add_argument("-no", "--no_samples", type=int, default=10, help="Number of decorrelated samples to generate.")
par.add_argument("-np", "--no_processes", type=int, default=os.cpu_count(), help="Number of cores to run code on.")

args = par.parse_args()

if __name__ == "__main__":
    discretization = args.discretization
    knot_type = args.knot
    sampler = args.sampler
    samples = args.no_samples
    num_processes = args.no_processes

    main()