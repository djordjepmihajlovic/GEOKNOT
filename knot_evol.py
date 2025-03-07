import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm 
import matplotlib.colors as mcolors  
from argparse import ArgumentParser
from knot_init import *

'''
BFACF algorithm for polygonal lattice knot evolution.

Key Features:
    * Oriented lattice knots S^{1} in Z^{3}
    * Writhe calculation (https://www.unige.ch/math/folks/cimasoni/writhe.pdf)
    * MCMC evolution towards highly entangled configurations
    * Visualization

To Implement:
    * Knotoids
    * Bonded knotoids
    * Proteins (spec. bonded knotoids (capture forces) ~ protein_init.py ~ load in protein from PDB and automate)
    * S^{2} in Z^{4}, (S^{n} in Z^{n+2})?
    * TQFT inspired things?

To Do:
    * CNN with the entire state (large knot thats managed to span entire state space) and predict the knot type.

Currently (06/03/25):
    * Need a way to get extremely entangled configurations
'''

@njit()
def neighbours(array, point):
    '''
    Takes in array and specified point, outputs an array of neighbours of point and associated neighbour value.
    '''

    neighbour = np.empty((26, 4))

    idx = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:

                if dx == 0 and dy == 0 and dz == 0:
                    continue

                nx_1, ny_1, nz_1 = point[0].item() + dx, point[1].item() + dy, point[2].item() + dz

                if 0<= nx_1 <array.shape[0] and 0<= ny_1 <array.shape[1] and 0<= nz_1 <array.shape[2]: 
                    neighbour[idx] = [nx_1, ny_1, nz_1, array[nx_1][ny_1][nz_1]]
                    idx += 1

    return neighbour

@njit()
def restricted_neighbours(array, point):
    '''
    Takes in array and point in array and outputs an array of neighbours and neighbour value.
    '''

    neighbour = np.empty((6, 4))

    idx = 0
    for dx in [-1, 1]:
        nx_1, ny_1, nz_1 = point[0].item() + dx, point[1].item(), point[2].item()

        if 0<= nx_1 <array.shape[0]: 
            neighbour[idx] = [nx_1, ny_1, nz_1, array[nx_1][ny_1][nz_1]]
            idx += 1

    for dy in [-1, 1]:
        nx_1, ny_1, nz_1 = point[0].item(), point[1].item() + dy, point[2].item()

        if 0<= ny_1 <array.shape[1]: 
            neighbour[idx] = [nx_1, ny_1, nz_1, array[nx_1][ny_1][nz_1]]
            idx += 1

    for dz in [-1, 1]:
        nx_1, ny_1, nz_1 = point[0].item(), point[1].item(), point[2].item() + dz

        if 0<= nz_1 <array.shape[2]: 
            neighbour[idx] = [nx_1, ny_1, nz_1, array[nx_1][ny_1][nz_1]]
            idx += 1

    return neighbour


def orient(array):
    '''
    For fixing orientation: make copy of array, assign 1 to some point and N to one of its neighbours. Now impose that every point (excluding)
    1 and N must have a neighbour with +1 value and -1 value of current value. 
    Save both oriented and unoriented structure, use unoriented structure (just 1's) for some calcs.
    '''

    indicies = np.argwhere(array == 1)
    oriented_structure = array.copy()
    p1 = indicies[0]

    vec_1 = np.empty((2, 3))

    neighbourhood = neighbours(array, p1)
    idx = 0

    for i in neighbourhood:
        if i[3] == 1:
            vec_1[idx] = [i[0], i[1], i[2]]
            idx += 1

    vec_1 = vec_1.astype(int)

    oriented_structure[vec_1[0][0]][vec_1[0][1]][vec_1[0][2]] = len(indicies)
    oriented_structure[vec_1[1][0]][vec_1[1][1]][vec_1[1][2]] = 2

    neighbourhood = neighbours(oriented_structure, vec_1[1])

    update_number = 3

    neighbourhood = neighbourhood.astype(int)

    for i in neighbourhood:
        if i[3] == 1 and (i[0] != p1[0] or i[1] != p1[1] or i[2] != p1[2]):
            oriented_structure[i[0]][i[1]][i[2]] = update_number
            prev_number = update_number
            update_number += 1
        
    while len(np.argwhere(oriented_structure ==1)) > 1:

        index = np.argwhere(oriented_structure == prev_number)

        for ind in index:
            neighbourhood = neighbours(oriented_structure, ind)
            neighbourhood = neighbourhood.astype(int)
            for i in neighbourhood:
                if i[3] == 1:
                    oriented_structure[i[0]][i[1]][i[2]] = update_number
                    prev_number = update_number
                    update_number += 1

    return oriented_structure


@njit()
def find_new(array, edge):
    '''
    Find valid locations to move edge, nb. needs to be restricted neighbours
    '''

    valid_neighbours = []

    neighbourhood = restricted_neighbours(array, edge)

    for i in neighbourhood:
        if i[3] != 1:
            #and i[0] != 0 or i[1] != 0 or i[2] !=0:
            valid_neighbours.append(i)
    
    new_edge = np.random.choice(len(valid_neighbours))

    return valid_neighbours[new_edge]

@njit()
def check_double_edge(array):
    '''
    Checks for singular points.
    '''
    indicies = np.argwhere(array > 1)
    
    if len(indicies) != 0:
        for i in indicies:
            array[i[0]][i[1]][i[2]] = 0

    return array

@njit()
def check_verticies(array):
    '''
    Checks the verticies of the 3D state space.
    '''
    indicies = np.argwhere(array > 0)

    status = 0

    for i in indicies:
        check = []

        neighbourhood = neighbours(array, i)
        for j in neighbourhood:
            check.append(j[3])

        if len(np.argwhere(np.array(check)>0)) != 2:
            status -= 1

    return status

@njit()
def crumple(array):
    '''
    Markov Chain method to enforce movement toward more crumpled structure.
    Defines energy as sum of distance between all indicies
    '''

    crossing_array = array.copy()
    indicies = np.argwhere(crossing_array>0)
    dist = 0

    for i in indicies:
        for j in indicies:
            if i[0] != j[0] or i[1] != j[1] or i[2] != j[2]:
                dist += (i[0] - j[0]) **2 + (i[1] - j[1]) **2 + (i[2] - j[2]) **2

    energy = dist **(1/2)

    return -energy

def shear(array):
    return array


@njit()
def lattice_writhe(array):
    '''
    Want to explore Tait numbers T(A_{i}) on the two areas (8 areas modulo symmetry) on the indicatrix corresponding to projections on: 
    (x, z) plane, (y, z) plane.
    Additionally, need to have defined direction to capture +,- crossings: cross product of a fixed orientation along knot
    '''

    indicies = np.argwhere(array>0)
    TA_1 = 0
    TA_2 = 0

    for i in indicies:
        '''
        This gives you x, y coordinates of places in projection where crossings occur.
        Nb. limited height so triple crossings don't occur
        Need to build vector for each strand in crossing (3 points) and take cross product
        Also need to define orientation for these vectors
        Current issue: what to do when triple crossing? (lines 257, 264, 319, 312 ignore issue for now)
        : solution - define 'good' projections (0, 1, 2) points, omit any other projections
        '''

        projected_vector_yz = array[:, i[1], i[2]] # list on yz plane projections
        projected_vector_xz = array[i[0], :, i[2]] # list on xz plane projections

        points_yz = np.argwhere(projected_vector_yz > 0) # gives two locations (x) of crossings
        points_xz = np.argwhere(projected_vector_xz > 0) # gives two locations (y) of crossings

        if len(points_yz) == 2: # unsure as to why this doesnt omit the other projections?

            vec_1_yz = np.empty((2, 4))
            vec_2_yz = np.empty((2, 4))

            idx_1_yz = 0
            idx_2_yz = 0

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:

                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        nx_1, ny_1, nz_1 = points_yz[0].item() + dx, i[1] + dy, i[2] + dz
                        nx_2, ny_2, nz_2 = points_yz[1].item() + dx, i[1] + dy, i[2] + dz

                        if 0<= nx_1 <array.shape[0] and 0<= ny_1 <array.shape[1] and 0<= nz_1 <array.shape[2] and array[nx_1][ny_1][nz_1] > 0:

                            if idx_1_yz < 2:
                                vec_1_yz[idx_1_yz] = [nx_1, ny_1, nz_1, array[nx_1][ny_1][nz_1]]

                            idx_1_yz +=1

                        if 0<= nx_2 <array.shape[0] and 0<= ny_2 <array.shape[1] and 0<= nz_2 <array.shape[2] and array[nx_2][ny_2][nz_2] > 0:

                            if idx_2_yz < 2:
                                vec_2_yz[idx_2_yz] = [nx_2, ny_2, nz_2, array[nx_2][ny_2][nz_2]]

                            idx_2_yz += 1

            # define vector to point small -> large
            p1 = vec_1_yz[0]
            p2 = vec_1_yz[1]
            p3 = vec_2_yz[0]
            p4 = vec_2_yz[1]

            arrow_1 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
            arrow_2 = [p3[0] - p4[0], p3[1] - p4[1], p3[2] - p4[2]]

            arrow_1 = np.array(arrow_1)
            arrow_2 = np.array(arrow_2)

            if p4[3]>p3[3]:
                arrow_2 = -1 * arrow_2
            
            if p2[3]>p1[3]:
                arrow_1 = -1 * arrow_1

            cross_prod = np.cross(arrow_1, arrow_2)
            mag = ((arrow_1[0]-arrow_2[0])**2 + (arrow_1[1]-arrow_2[1])**2 +(arrow_1[2]-arrow_2[2])**2)**(1/2)
            sign = np.sign(cross_prod[0])

            TA_1 += sign

        if len(points_xz) == 2:

            vec_1_xz = np.empty((2, 4))
            vec_2_xz = np.empty((2, 4))

            idx_1_xz = 0
            idx_2_xz = 0

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:

                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        nx_1, ny_1, nz_1 = i[0] + dx, points_xz[0].item() + dy, i[2] + dz
                        nx_2, ny_2, nz_2 = i[0] + dx, points_xz[1].item() + dy, i[2] + dz

                        if 0<= nx_1 <array.shape[0] and 0<= ny_1 <array.shape[1] and 0<= nz_1 <array.shape[2] and array[nx_1][ny_1][nz_1] > 0:

                            if idx_1_xz < 2:
                                vec_1_xz[idx_1_xz] = [nx_1, ny_1, nz_1, array[nx_1][ny_1][nz_1]]

                            idx_1_xz +=1

                        if 0<= nx_2 <array.shape[0] and 0<= ny_2 <array.shape[1] and 0<= nz_2 <array.shape[2] and array[nx_2][ny_2][nz_2] > 0:

                            if idx_2_xz < 2:
                                vec_2_xz[idx_2_xz] = [nx_2, ny_2, nz_2, array[nx_2][ny_2][nz_2]]

                            idx_2_xz += 1

            # define vector to point small -> large
            p1 = vec_1_xz[0]
            p2 = vec_1_xz[1]
            p3 = vec_2_xz[0]
            p4 = vec_2_xz[1]

            arrow_1 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
            arrow_2 = [p3[0] - p4[0], p3[1] - p4[1], p3[2] - p4[2]]

            arrow_1 = np.array(arrow_1)
            arrow_2 = np.array(arrow_2)

            if p4[3]>p3[3]:
                arrow_2 = -1 * arrow_2
            
            if p2[3]>p1[3]:
                arrow_1 = -1 * arrow_1

            cross_prod = np.cross(arrow_1, arrow_2)
            sign = np.sign(cross_prod[1])
            mag = ((arrow_1[0]-arrow_2[0])**2 + (arrow_1[1]-arrow_2[1])**2 +(arrow_1[2]-arrow_2[2])**2)**(1/2)

            TA_2 += sign

    return (TA_1 + TA_2)/ 4


@njit()
def metropolis_acceptance(old, new, temperature):
    '''
    Probability to randomly accept update moves that arent crossing increasing.
    '''
    if new > old:
        return True
    else:
        prob = np.exp((new-old)/temperature)
        return np.random.uniform(0, 1) < prob
    
def BFACF_update(array, temperature, time):
    '''
    BFACF algorithm on oriented curve
    '''

    update_array = array.copy()

    # old_energy = compute_energy(update_array)

    old_c_energy = crumple(update_array)
    old_energy = lattice_writhe(update_array)

    valid_indicies = np.argwhere(array > 1)

    random_edge = np.random.choice(len(valid_indicies))
    new_edge = find_new(update_array, valid_indicies[random_edge])

    update_array[valid_indicies[random_edge][0]][valid_indicies[random_edge][1]][valid_indicies[random_edge][2]] = 0

    update_array[int(new_edge[0])][int(new_edge[1])][int(new_edge[2])] += array[valid_indicies[random_edge][0]][valid_indicies[random_edge][1]][valid_indicies[random_edge][2]]

    status = check_verticies(update_array)

    # new_energy = compute_energy(update_array)

    new_c_energy = crumple(update_array)
    new_energy = lattice_writhe(update_array)

    if status < 0:
        return array, 0
    
    else:
        if time< 1000000:
            return update_array, new_energy
        
        elif 900000<time<905000:
            if metropolis_acceptance(old_c_energy, new_c_energy, temperature):
                return update_array, new_energy
            else:
                return array, 0
        
        else: 
            if metropolis_acceptance(old_energy, new_energy, temperature):
                return update_array, new_energy
            else:
                return array, 0


def main():

    animate3D = False
    plot = True
    state_space = np.zeros((8, discretization, discretization))

    knot = Knot(knot_type, state_space)
    unknot = knot.initialize()

    print('Orienting...')
    unknot = orient(unknot)
    writhe = lattice_writhe(unknot)
    print('Initial update...')
    unknot, energy = BFACF_update(unknot, temperature, 0)

    # run BFACF for a bunch of timesteps
    writhe_calc = []
    time_subdiv = 0

    for i in range(it):
        if i%(1000000) == 0:
            time_subdiv = 0
        unknot, energy = BFACF_update(unknot, temperature, time_subdiv)
        time_subdiv += 1

        if i%1000 == 0:
            print(f"simulation: {i/it}%")
            writhe_calc.append(energy)

    coords = np.argwhere(unknot>0)
    coord_dat = [(unknot[i[0], i[1], i[2]], i[0], i[1], i[2]) for i in coords]

    np.savetxt(f'examples/config_{knot_type}.csv', coord_dat, delimiter=",", fmt='%d')

    writhe_calc = [x for x in writhe_calc if x != 0]  # O(n)

    plt.hist(writhe_calc)
    plt.savefig(f'figs/writhe_distn_{knot_type}')

    if plot == True:

        norm = mcolors.Normalize(vmin=np.min(unknot[unknot > 0]), vmax=np.max(unknot))
        cmap = cm.coolwarm  

        # Initialize color array
        colors = np.zeros(unknot.shape + (4,))  # RGBA color array

        # Apply colormap for nonzero values
        mask = unknot > 0  
        colors[mask] = cmap(norm(unknot[mask]))  

        colors[..., 3] = np.where(unknot > 0, 1.0, 0.0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the voxels
        ax.voxels(unknot > 0, facecolors=colors)

        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.set_zlim([0, 100])

        plt.savefig(f'figs/tangle_{knot_type}')

    if animate3D == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            global unknot
            for i in range(1000):  # Perform multiple updates per frame
                unknot = BFACF_update(unknot, temperature) 
            
            ax.clear()  # Clear previous voxels

            # Get voxel positions (nonzero values)
            filled = unknot > 0
            ax.voxels(filled, facecolors='blue', edgecolors='black', alpha=0.5)
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 100])
            ax.set_zlim([0, 100])

            ax.set_title(f"Unknot")  # Optional title update

            return ax,

        ani = animation.FuncAnimation(fig, update, frames=10, interval=500, blit=False)
        plt.show()


'''
    Lets us specify arguements for the code.
'''

par = ArgumentParser()

par.add_argument("-d", "--discretization", type=int, default=100, help="Discretization of state space y,z axis.")
par.add_argument("-t", "--temperature", type=float, default=0.01, help="Temperature of system, lets it vary from MCMC constraint.")
par.add_argument("-it", "--iterations", type=int, default=1000, help="Iterations of BFACF algorithm.")
par.add_argument("-k", "--knot", type=str, default='0_1', help="Knot type.")

args = par.parse_args()

if __name__ == "__main__":
    discretization = args.discretization
    temperature = args.temperature
    it = args.iterations
    knot_type = args.knot

    main()