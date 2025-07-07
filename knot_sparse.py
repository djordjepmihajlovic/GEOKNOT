import matplotlib.pyplot as plt
from argparse import ArgumentParser
from knot_evolution_hash import *
from quantum_knot_invs import *
import time
from defunct.knot_evolution import lattice_writhe_Klenin

def main():
    '''
    A function for running a simulation of pivot and BFACF moves on a given knot.
    '''

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
    # contraints = (writhe, entanglement)
    contraints = ((30, 35), (1000, 1500))
    # pivot
    # evolved = pivot(oriented, timesteps=it*1, knot=knot_type, aimed_range=contraints)
    # bfacf (2x pivot)
    print('Running bfacf 1...')
    evolved = loopBFACF(oriented, timesteps=it*5)
    # print('Running pivot 2...')
    # evolved = pivot(evolved, timesteps=it*10, knot=knot_type, aimed_range=contraints)

    end_time = time.time() - start_time
    print("Simulation time:", end_time)

    topo = Q_invariant(evolved, 'Uq(sl2)').alexander_polynomial_hash(knot_type) 
    print(f"Knot type consistent?: {topo}")

    # Convert back to array for plotting
    min_x = min(p[0] for p in evolved)
    min_y = min(p[1] for p in evolved)
    min_z = min(p[2] for p in evolved)

    max_x = max(p[0] for p in evolved) + 1
    max_y = max(p[1] for p in evolved) + 1
    max_z = max(p[2] for p in evolved) + 1

    offset_x = abs(min_x) if min_x < 0 else 0
    offset_y = abs(min_y) if min_y < 0 else 0
    offset_z = abs(min_z) if min_z < 0 else 0

    array = np.zeros((max_x + offset_x, max_y + offset_y, max_z + offset_z), dtype=np.float64)

    for (x, y, z), val in evolved.items():
        array[x + offset_x, y + offset_y, z + offset_z] = val

    # Plot result
    coords = np.argwhere(array>0)
    coord_dat = [(array[i[0], i[1], i[2]], i[0], i[1], i[2]) for i in coords]

    elements = sorted(coord_dat, key=lambda x: x[0])
    
    joggle_scale = 0
    np.random.seed(42)
    elements_jiggled = [np.array([i[1:4] for i in elements], dtype=float) +
    np.random.normal(scale=joggle_scale, size=(len(elements), 3))]

    new_coord = [tuple(row) for row in elements_jiggled[0]]
    w = [i[0] for i in elements]
    new_coord_w = [(w[idx],) + coord for idx, coord in enumerate(new_coord)]

    im = lattice_writhe_Klenin(new_coord_w)
    print(np.sum(im))
    plt.imshow(im)
    plt.colorbar()

    np.savetxt(f'examples/config_{knot_type}.csv', new_coord_w, delimiter=",", fmt='%.5f')
    plot_3d_line(new_coord_w)

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