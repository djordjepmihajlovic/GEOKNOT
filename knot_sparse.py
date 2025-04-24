import matplotlib.pyplot as plt
from argparse import ArgumentParser
from knot_evolution_hash import *
import time

def main():


    state_space = np.zeros((discretization, discretization, discretization))
    knot = Knot(knot_type, state_space)
    unknot = knot.initialize()

    array_dict = {}
    iter = np.nditer(unknot, flags=['multi_index'])
    for val in iter:
        if val != 0:
            array_dict[iter.multi_index] = val.item()

    # Orient knot
    print('Orienting...')
    oriented = orient(array_dict)
    for key, value in oriented.items():
        if value == 1.0:
            oriented[key] = 1  # orientation float issue

    print('Running pivot...')
    start_time = time.time()
    # pivot
    evolved = pivot(oriented, timesteps=it*10, knot=knot_type)
    # bfacf (10x pivot)
    print('Running bfacf...')
    evolved, wr, rog = BFACF(evolved, timesteps=it)
    end_time = time.time() - start_time
    print("Simulation time:", end_time)

    # Convert back to array for plotting
    max_x = max(p[0] for p in evolved) + 1
    max_y = max(p[1] for p in evolved) + 1
    max_z = max(p[2] for p in evolved) + 1
    array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (x, y, z), val in evolved.items():
        array[x, y, z] = val

    # Plot result
    coords = np.argwhere(array>0)
    coord_dat = [(array[i[0], i[1], i[2]], i[0], i[1], i[2]) for i in coords]

    np.savetxt(f'examples/config_{knot_type}.csv', coord_dat, delimiter=",", fmt='%d')
    plot_3d(array)


par = ArgumentParser()
'''
    Lets us specify arguements for the code.
'''

par.add_argument("-d", "--discretization", type=int, default=100, help="Discretization of state space y,z axis.")
par.add_argument("-it", "--iterations", type=int, default=1000, help="Iterations of BFACF algorithm.")
par.add_argument("-k", "--knot", type=str, default='0_1', help="Knot type.")
par.add_argument("-s", "--sampler", type=str, default='Metropolis', help="Sampling method.")

args = par.parse_args()

if __name__ == "__main__":
    discretization = args.discretization
    it = args.iterations
    knot_type = args.knot
    sampler = args.sampler

    main()