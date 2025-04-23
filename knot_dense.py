import matplotlib.pyplot as plt
from argparse import ArgumentParser
from knot_evolution import *
import time

def main():

    state_space = np.zeros((discretization, discretization, discretization))
    knot = Knot(knot_type, state_space)
    unknot = knot.initialize()

    # orient knot
    print('Orienting...')
    unknot = orient(unknot)

    projections_111 = points_on_axis(unknot, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
    projections_1m11 = points_on_axis(unknot, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
    projections_11m1 = points_on_axis(unknot, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
    projections_1m1m1 = points_on_axis(unknot, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))

    print('Initial writhe...')
    print(np.sum(lattice_writhe_Klenin(unknot))/(np.pi**2))
    print(lattice_writhe_Cimasoni(unknot,
                                    projections_111=projections_111,
                                    projections_1m11=projections_1m11,
                                    projections_11m1=projections_11m1,
                                    projections_1m1m1=projections_1m1m1))


    start_time = time.time()
    # unknot, g_w = BFACF(array=unknot, timesteps=it, sampler=sampler)
    unknot = pivot(unknot, it, knot_type)
    end_time = time.time() - start_time
    print(end_time)

    print('Final writhe...')
    projections_111 = points_on_axis(unknot, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
    projections_1m11 = points_on_axis(unknot, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
    projections_11m1 = points_on_axis(unknot, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
    projections_1m1m1 = points_on_axis(unknot, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))

    print(lattice_writhe_Cimasoni(unknot,
                                projections_111=projections_111,
                                projections_1m11=projections_1m11,
                                projections_11m1=projections_11m1,
                                projections_1m1m1=projections_1m1m1))
    
    print(np.sum(lattice_writhe_Klenin(unknot))/(np.pi**2))

    Q_invariant(unknot, 'Uq(sl2)').alexander_polynomial(knot_type) 

    # save coords as required
    coords = np.argwhere(unknot>0)
    coord_dat = [(unknot[i[0], i[1], i[2]], i[0], i[1], i[2]) for i in coords]

    np.savetxt(f'examples/config_{knot_type}_WRTEST.csv', coord_dat, delimiter=",", fmt='%d')
    
    plot_3d(unknot)


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