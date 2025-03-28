import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as mcolors  
from argparse import ArgumentParser
from knot_init import *
from quantum_knot_invs import *

'''
BFACF/Pivot algorithm with Wang-Landau sampler implementation for flat writ dos polygonal lattice knot embeddings.

Relevant literature: 
    *
    *
    *

Key Features:
    * Oriented lattice knots S^{1} in Z^{3}
    * Writhe calculation (https://www.unige.ch/math/folks/cimasoni/writhe.pdf)
    * Wang-Landau sampling toward flat writhe dos (g_w)
    * Visualization

To Implement:
    * Links
    * Knotoids 
    * Bonded knotoids
    * Proteins (spec. bonded knotoids (capture forces) ~ protein_init.py ~ load in protein from PDB and automate)
    * S^{2} in Z^{4}, (S^{n} in Z^{n+2})

Currently (25/03/25):
    * Implemented Wang-Landau sampler to flatten distribution
    * Need to test uncorrelated samples; how many iterations between sample save
    * Need to test writhe calc; make sure it is correct
    *** Neighbourhood can contain 3 points, requirement is that 2 of the neighbours at the previous and next point in the knot ***

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
        if i[3] == 0:
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
        value = array[i[0]][i[1]][i[2]]
        for j in neighbourhood:
            if j[3] == (value+1)%(len(indicies)) or j[3] == (value-1)%(len(indicies)):
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


def sparse_point_density(array):
    '''
    Sparse point density is a method to ensure points from distant ends of knot cluster close to each other.
    Similar to crumple however distance between knots of furthers position is optimized.
    '''

    sparse_array = array.copy()
    for i in sparse_array:
        neighbours = neighbours(sparse_array)
        for j in neighbours:
            neighbours_of_neighbours = neighbours[j]

            for point in neighbours_of_neighbours:
                if point[3] != 0:
                    total_diff = 0


@njit()
def lattice_writhe(array):
    '''
    Want to explore Tait numbers T(A_{i}) on the two areas (8 areas modulo symmetry) on the indicatrix corresponding to projections on: 
    (x, z) plane, (y, z) plane [(x, y)??].
    Additionally, need to have defined direction to capture +,- crossings: cross product of a fixed orientation along knot
    '''

    indicies = np.argwhere(array>0)
    TA_1 = 0
    TA_2 = 0
    TA_3 = 0

    for i in indicies:
        '''
        This gives you x, y coordinates of places in projection where crossings occur.
        Nb. limited height so triple crossings don't occur
        Need to build vector for each strand in crossing (3 points) and take cross product
        Also need to define orientation for these vectors
        Current issue: what to do when triple crossing? (lines 257, 264, 319, 312 ignore issue for now)
        : solution - define 'good' projections (0, 1, 2) points, omit any other projections
        ! Projections are wrong, these are projections onto planes (1, 1, 0), (1, 0, 1), (0, 1, 1)
        Projections of 8 quadrant indicatrix:
        1. (1, 1, 1)
        2. (1, 1, -1)
        3. (1, -1, 1)
        4. (1, -1, -1) 
        '''

        projected_vector_yz = array[:, i[1], i[2]] # list on yz plane projections
        projected_vector_xz = array[i[0], :, i[2]] # list on xz plane projections
        projected_vector_xy = array[i[0], i[1], :] # list on xy plane projections

        points_yz = np.argwhere(projected_vector_yz > 0) # gives two locations (x) of crossings
        points_xz = np.argwhere(projected_vector_xz > 0) # gives two locations (y) of crossings
        points_xy = np.argwhere(projected_vector_xy > 0)

        ## yz plane
        if len(points_yz) == 2: 

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

            ##
            writhe_distance = abs(vec_1_yz[0][3] - vec_2_yz[0][3]) 
            ##

            arrow_1 = np.array(arrow_1)
            arrow_2 = np.array(arrow_2)

            if p4[3]>p3[3]:
                arrow_2 = -1 * arrow_2
            
            if p2[3]>p1[3]:
                arrow_1 = -1 * arrow_1

            cross_prod = np.cross(arrow_1, arrow_2)
            mag = (arrow_1[0]-arrow_2[0])**2 + (arrow_1[1]-arrow_2[1])**2 + (arrow_1[2]-arrow_2[2])**2
            if mag != 0:
                sign = np.sign(cross_prod[0]) * writhe_distance/mag
            else:
                sign = 0
            TA_1 += sign

        ## xz plane
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

            ##
            writhe_distance = abs(vec_1_xz[0][3] - vec_2_xz[0][3]) 
            ##

            arrow_1 = np.array(arrow_1)
            arrow_2 = np.array(arrow_2)

            if p4[3]>p3[3]:
                arrow_2 = -1 * arrow_2
            
            if p2[3]>p1[3]:
                arrow_1 = -1 * arrow_1

            cross_prod = np.cross(arrow_1, arrow_2)
            mag = (arrow_1[0]-arrow_2[0])**2 + (arrow_1[1]-arrow_2[1])**2 + (arrow_1[2]-arrow_2[2])**2
            if mag != 0:
                sign = np.sign(cross_prod[1]) * writhe_distance/mag
            else:
                sign = 0
            TA_2 += sign

        ## xy plane, not sure if this is necessary.
        if len(points_xy) == 2:

            vec_1_xy = np.empty((2, 4))
            vec_2_xy = np.empty((2, 4))

            idx_1_xy = 0
            idx_2_xy = 0

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:

                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        nx_1, ny_1, nz_1 = i[0] + dx, i[1] + dy, points_xy[0].item() + dz
                        nx_2, ny_2, nz_2 = i[0] + dx, i[1] + dy, points_xy[1].item() + dz

                        if 0<= nx_1 <array.shape[0] and 0<= ny_1 <array.shape[1] and 0<= nz_1 <array.shape[2] and array[nx_1][ny_1][nz_1] > 0:

                            if idx_1_xy < 2:
                                vec_1_xy[idx_1_xy] = [nx_1, ny_1, nz_1, array[nx_1][ny_1][nz_1]]

                            idx_1_xy +=1

                        if 0<= nx_2 <array.shape[0] and 0<= ny_2 <array.shape[1] and 0<= nz_2 <array.shape[2] and array[nx_2][ny_2][nz_2] > 0:

                            if idx_2_xy < 2:
                                vec_2_xy[idx_2_xy] = [nx_2, ny_2, nz_2, array[nx_2][ny_2][nz_2]]

                            idx_2_xy += 1

            # define vector to point small -> large
            p1 = vec_1_xy[0]
            p2 = vec_1_xy[1]
            p3 = vec_2_xy[0]
            p4 = vec_2_xy[1]

            arrow_1 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
            arrow_2 = [p3[0] - p4[0], p3[1] - p4[1], p3[2] - p4[2]]

            ##
            writhe_distance = abs(vec_1_xy[0][3] - vec_2_xy[0][3]) 
            ##

            arrow_1 = np.array(arrow_1)
            arrow_2 = np.array(arrow_2)

            if p4[3]>p3[3]:
                arrow_2 = -1 * arrow_2
            
            if p2[3]>p1[3]:
                arrow_1 = -1 * arrow_1

            cross_prod = np.cross(arrow_1, arrow_2)
            mag = (arrow_1[0]-arrow_2[0])**2 + (arrow_1[1]-arrow_2[1])**2 + (arrow_1[2]-arrow_2[2])**2
            if mag != 0:
                sign = np.sign(cross_prod[2]) * writhe_distance/mag
            else:
                sign = 0
            TA_3 += sign

    return (TA_1 + TA_2 + TA_3)/6

def metropolis_acceptance(old_energy, new_energy, temperature):
    '''
    Metropolis acceptance criterion.
    '''

    if new_energy < old_energy:
        return True
    else:
        return np.random.rand() < np.exp((old_energy - new_energy)/temperature)

    
def BFACF(array, timesteps, sampler):
    '''
    BFACF with chosen sampling methods
    '''

    if sampler == "Wang-Landau":
        '''
        Wang-Landau sampling algorithm for flat distributions of dos (energy).
        '''

        old_energy = lattice_writhe(array) # initial writhe computation

        bins = 50
        f = np.exp(1)
        writhe_min, writhe_max = -20, 20
        bin_edges = np.linspace(writhe_min, writhe_max, bins + 1)
        g_w = np.zeros(bins)
        H_w = np.zeros(bins)

        def get_bin(w):
            return np.digitize(w, bin_edges) -1
        
        def is_flat(H):
            avg = np.mean(H[H>0])
            return np.std(H[H>0])/ avg < 0.2
        
        for time in range(timesteps):

            if time % 1000 == 0:
                print(f"simulation: {time/timesteps}")

            update_array = array.copy()
            valid_indicies = np.argwhere(array > 1)

            random_edge = np.random.choice(len(valid_indicies))
            new_edge = find_new(update_array, valid_indicies[random_edge])

            update_array[valid_indicies[random_edge][0]][valid_indicies[random_edge][1]][valid_indicies[random_edge][2]] = 0
            update_array[int(new_edge[0])][int(new_edge[1])][int(new_edge[2])] += array[valid_indicies[random_edge][0]][valid_indicies[random_edge][1]][valid_indicies[random_edge][2]]
            status = check_verticies(update_array)

            new_energy = lattice_writhe(update_array)

            if status < -2:
                continue

            if new_energy < writhe_min or new_energy > writhe_max:
                continue

            current_state = get_bin(old_energy)
            new_state = get_bin(new_energy)

            if np.random.rand() < min(1, np.exp(g_w[current_state]-g_w[new_state])):
                old_energy = new_energy
                current_state = new_state
                array = update_array

            g_w[current_state] += f ### NOT SURE ###
            H_w[current_state] += 1

            if time % 1000 == 0 and is_flat(H_w):
                H_w[:] = 0
                f *= 0.5
    
    elif sampler == "Metropolis":
        '''
        Metropolis acceptance algorithm.
        '''

        g_w = []
        old_energy = lattice_writhe(array) # initial writhe computation
        
        for time in range(timesteps):

            if time % 1000 == 0:
                print(f"simulation: {time/timesteps}")

            update_array = array.copy()
            valid_indicies = np.argwhere(array > 1)

            random_edge = np.random.choice(len(valid_indicies))
            new_edge = find_new(update_array, valid_indicies[random_edge])

            update_array[valid_indicies[random_edge][0]][valid_indicies[random_edge][1]][valid_indicies[random_edge][2]] = 0
            update_array[int(new_edge[0])][int(new_edge[1])][int(new_edge[2])] += array[valid_indicies[random_edge][0]][valid_indicies[random_edge][1]][valid_indicies[random_edge][2]]
            status = check_verticies(update_array)

            new_energy = lattice_writhe(update_array)

            if status < -2:
                continue
            else:
                if metropolis_acceptance(old_energy, new_energy, 0.01):
                    array = update_array
                    old_energy = new_energy
                    g_w.append(new_energy)

    return array, g_w
        
def pivot(array):
    '''
    Pivot algorithm to increase autocorrelation of samples.
    Notice: valid pivots occur on a shared axis in Z^{3}
    '''
    update_array = array.copy()
    valid_indicies = np.argwhere(array > 1)

    def check_axis(coordinates_1, coordinates_2):
        shared_axis = []
        for i in range(3):
            if coordinates_1[i] == coordinates_2[i]:
                shared_axis.append(i)

        if len(shared_axis) > 1:
            return True

    random_edge_1, random_edge_2 = np.random.choice(len(valid_indicies), size=2, replace=False)

    if random_edge_2>random_edge_1:
        w1 = [i for i in range(random_edge_1, random_edge_2+1)]
        axis = np.argwhere(array == random_edge_2) - np.argwhere(array == random_edge_1)

    else:
        w1 = [i for i in range(random_edge_2, random_edge_1+1)]
        axis = np.argwhere(array == random_edge_1) - np.argwhere(array == random_edge_2)

    u = axis[0]/np.linalg.norm(axis[0])
    ux, uy, uz = u[0], u[1], u[2]

    pivot_point = np.argwhere(array == random_edge_1)
    ang = np.random.choice([np.pi/2, -np.pi/2, np.pi])

    R = np.array([
        [ux**2*(1-np.cos(ang))+np.cos(ang), ux*uy*(1-np.cos(ang))-uz*np.sin(ang), ux*uz*(1-np.cos(ang))+uy*np.sin(ang)],
        [ux*uy*(1-np.cos(ang))+uz*np.sin(ang), uy**2*(1-np.cos(ang))+np.cos(ang), uy*uz*(1-np.cos(ang))-ux*np.sin(ang)],
        [ux*uz*(1-np.cos(ang))-uy*np.sin(ang), uy*uz*(1-np.cos(ang))+ux*np.sin(ang), uz**2*(1-np.cos(ang))+np.cos(ang)]])
    
    for x in w1:
        index = np.argwhere(array == x)
        translated_index = index - pivot_point 

        if len(translated_index)>0:

            new_index = np.dot(R, translated_index[0])
            new_index = np.round(new_index + pivot_point).astype(int) 
            update_array[index[0][0]][index[0][1]][index[0][2]] = 0
            update_array[new_index[0][0]][new_index[0][1]][new_index[0][2]] = x


    status = check_verticies(update_array)

    
    if status < -2:
        return array
    else:
        print(random_edge_1, random_edge_2)
        return update_array


def main():

    plot = False
    state_space = np.zeros((discretization, discretization, discretization))

    knot = Knot(knot_type, state_space)
    unknot = knot.initialize()

    # orient knot
    print('Orienting...')
    unknot = orient(unknot)

    # run test pivot:
    for i in range(0, 1000):
        unknot = pivot(unknot)

    for x in range(0, 100):
    # run BFACF for a bunch of timesteps
        unknot, g_w = BFACF(array=unknot, timesteps=it, sampler="Wang-Landau")

        # save coords as required
        coords = np.argwhere(unknot>0)
        coord_dat = [(unknot[i[0], i[1], i[2]], i[0], i[1], i[2]) for i in coords]

        np.savetxt(f'examples/config_{knot_type}_{x}.csv', coord_dat, delimiter=",", fmt='%d')

        plt.hist(g_w)
        plt.xlabel('Writhe')
        plt.ylabel('Frequency')
        plt.title('Writhe Distribution Lattice Unknot')
        plt.savefig(f'figs/writhe_distn_{knot_type}_{x}')

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
        plt.show()


par = ArgumentParser()
'''
    Lets us specify arguements for the code.
'''

par.add_argument("-d", "--discretization", type=int, default=100, help="Discretization of state space y,z axis.")
par.add_argument("-it", "--iterations", type=int, default=1000, help="Iterations of BFACF algorithm.")
par.add_argument("-k", "--knot", type=str, default='0_1', help="Knot type.")

args = par.parse_args()

if __name__ == "__main__":
    discretization = args.discretization
    it = args.iterations
    knot_type = args.knot

    main()