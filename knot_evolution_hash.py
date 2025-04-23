import numpy as np
from numba import njit, prange
from knot_init import *
from quantum_knot_invs import *
from multiprocessing import Pool
import networkx as nx
from numba.typed import List
import random
import copy

'''
BFACF/Pivot algorithm with Wang-Landau sampler implementation for flat wrt dos polygonal lattice knot embeddings.

Relevant literature: 
    * Lattice Knots: 
    * BFACF: 
    * Pivot: 
    * Wang-Landau: 
    * Cimasoni Writhe calculation O(nlog(n)): 
    * Klenin Writhe calculation O(n^{2}): 
    * Quantum invariants of knots

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

    *** Urgently need to fix topology changing in BFACF ***
    *** Need to fix speed (can be massively parallelized), maybe we want to use C code. ***

'''

def neighbours(array_dict, point):
    """
    Returns the full 3x3x3 neighborhood of a point in 3D space.
    Each neighbor is a (x, y, z, value) tuple.
    """
    x, y, z = point
    neighbours = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue  # skip center
                nx, ny, nz = x + dx, y + dy, z + dz
                val = array_dict.get((nx, ny, nz), 0)
                neighbours.append((nx, ny, nz, val))

    return neighbours

def restricted_neighbours(array_dict, point):
    '''
    Takes in array and point in array and outputs an array of neighbours and neighbour value.
    '''

    x, y, z = point
    neighbours = []

    directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]

    for dx, dy, dz in directions:
        nx, ny, nz = x + dx, y + dy, z + dz
        val = array_dict.get((nx, ny, nz), 0)
        neighbours.append((nx, ny, nz, val))

    return neighbours

def orient(hash_table):
    '''
    For fixing orientation: make copy of hash table, assign 1 to some point and N to one of its neighbours. Now impose that every point (excluding)
    1 and N must have a neighbour with +1 value and -1 value of current value.
    Save both oriented and unoriented structure, use unoriented structure (just 1's) for some calculations.
    '''

    # Find the coordinates where the value is 1 (this replaces np.argwhere)
    indicies = [coord for coord, value in hash_table.items() if value == 1]
    oriented_structure = hash_table.copy()  # Create a copy of the hash table
    p1 = indicies[0]  # First point with value 1

    vec_1 = []

    # Get the neighbors of the point p1 (assuming the neighbours function is adapted for hash table)
    neighbourhood = neighbours(hash_table, p1)
    idx = 0

    for i in neighbourhood:
        if i[3] == 1:  # This checks if the neighbor has a value of 1
            vec_1.append([i[0], i[1], i[2]])
            idx += 1

    vec_1 = np.array(vec_1, dtype=int)

    # Assign values to the neighbors (same as original code)
    oriented_structure[tuple(vec_1[0])] = len(indicies)
    oriented_structure[tuple(vec_1[1])] = 2

    neighbourhood = neighbours(oriented_structure, tuple(vec_1[1]))

    update_number = 3

    # Ensure the neighbourhood is in the correct format
    neighbourhood = np.array(neighbourhood, dtype=int)

    for i in neighbourhood:
        if i[3] == 1 and tuple(i[:3]) != p1:
            oriented_structure[tuple(i[:3])] = update_number
            prev_number = update_number
            update_number += 1

    # Loop to update the structure until there’s only one point with value 1
    while len([value for value in oriented_structure.values() if value == 1]) > 1:
        # Get the index where the value is the last assigned number (prev_number)
        index = [coord for coord, value in oriented_structure.items() if value == prev_number]

        for ind in index:
            neighbourhood = neighbours(oriented_structure, ind)
            neighbourhood = np.array(neighbourhood, dtype=int)
            for i in neighbourhood:
                if i[3] == 1:
                    oriented_structure[tuple(i[:3])] = update_number
                    prev_number = update_number
                    update_number += 1

    return oriented_structure


def find_new(array, edge):
    '''
    Find valid locations to move edge, nb. needs to be restricted neighbours
    '''

    valid_neighbours = []
    neighbourhood = restricted_neighbours(array, edge)

    for i in neighbourhood:
        if array.get(i[:3], 0) == 0:
            valid_neighbours.append(i)
    
    if not valid_neighbours:
        return (-1, -1, -1, -1) # fixes no valid neighbour issue.

    return random.choice(valid_neighbours)

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

def check_verticies(array_dict):
    """
    Checks the vertices of the 3D state space using dictionary-based storage.
    """
    indicies = [pos for pos, val in array_dict.items() if val > 0]
    status = 0
    total = len(indicies)

    for i in indicies:
        check = []

        neighbourhood = neighbours(array_dict, i)
        value = array_dict[i]

        for j in neighbourhood:
            neighbor_val = j[3]
            if neighbor_val == (value + 1) % total or neighbor_val == (value - 1) % total:
                check.append(neighbor_val)

        if len([v for v in check if v > 0]) != 2:
            status -= 1

    return status


def build_correlation_graph(array):
    '''
    Build correlation graph between points for parallelization.
    '''
    indicies = np.argwhere(array > 0)
    G = nx.Graph()

    for idx, point in enumerate(indicies):
        G.add_node(idx, coords=tuple(point))

def crumple(array_dict):
    '''
    Markov Chain method to enforce movement toward more crumpled structure.
    Defines energy as sum of distance between all indices in the dictionary
    '''
    # Extract the keys (coordinates) and values (array values) from the dictionary
    coords = list(array_dict.keys())
    dist = 0

    # Iterate over all pairs of coordinates to calculate the distance
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):  # Avoid double-counting by only looking at pairs i < j
            coord_i = coords[i]
            coord_j = coords[j]

            # Calculate the squared Euclidean distance between the two coordinates
            dist += ((coord_i[0] - coord_j[0]) ** 2 + 
                     (coord_i[1] - coord_j[1]) ** 2 + 
                     (coord_i[2] - coord_j[2]) ** 2)

    # Energy is the negative of the square root of the distance sum
    energy = -dist ** (1 / 2)

    return energy

def radius_of_gyration(array):
    indicies = np.argwhere(array>0)
    center_of_mass = np.mean(indicies, axis=0)
    return np.sqrt(np.mean(np.sum((indicies - center_of_mass)**2, axis=1)))

@njit()
def positional_difference(array, update_array):
    indicies_1 = np.argwhere(array > 0)
    indicies_2 = np.argwhere(update_array > 0)

    # Extract the values at the indices
    values_1 = np.empty(len(indicies_1), dtype=np.float64)
    values_2 = np.empty(len(indicies_2), dtype=np.float64)

    for idx in range(len(indicies_1)):
        x, y, z = indicies_1[idx]
        values_1[idx] = array[x, y, z]

    for idx in range(len(indicies_2)):
        x, y, z = indicies_2[idx]
        values_2[idx] = update_array[x, y, z]

    # Sort indices based on the values
    sorted_indices_1 = indicies_1[np.argsort(values_1)]
    sorted_indices_2 = indicies_2[np.argsort(values_2)]
    differences = np.sum((sorted_indices_1 - sorted_indices_2) ** 2, axis=1)
    return np.sum(differences)


@njit()
def lattice_writhe_Cimasoni_defunct(array):
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
            sign = np.sign(cross_prod[0])
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
            sign = np.sign(cross_prod[1]) 

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
            sign = np.sign(cross_prod[2]) 

            TA_3 += sign

    return (TA_1 + TA_2 + TA_3)/6

def metropolis_acceptance(old_energy, new_energy, temperature):
    '''
    Metropolis acceptance criterion.
    '''

    if new_energy > old_energy: # (new writhe is larger)
        return True
    else:
        # Fix this! Accepting way too many falses
        # return False
        return np.random.rand() < 0.1

def points_on_axis(array, axis):
    '''
    Projects points onto a plane defined by axis as local 2D coordinate system aligned with plane.
    '''

    indicies = np.argwhere(array>0)
    axis = axis/np.linalg.norm(axis)

    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(axis, [1, 0, 0]) else np.array([0, 1, 0])
    u = np.cross(axis, arbitrary_vector)
    u /= np.linalg.norm(u)

    v = np.cross(axis, u)
    v /= np.linalg.norm(v)

    projected_points = []
    just_proj = []

    for idx, i in enumerate(indicies):
        dot_product = np.dot(i, axis)
        projection_coordinates = i - np.outer(dot_product, axis)
        '''
        Express 3D coords as 2D coords  
        '''
        u_comp = np.dot(projection_coordinates, u)
        v_comp = np.dot(projection_coordinates, v)

        projected_points.append([[u_comp, v_comp], i, array[i[0]][i[1]][i[2]]])
        just_proj.append([u_comp, v_comp])

    projected_points = np.array(projected_points, dtype=object)
    projected_points = sorted(projected_points, key=lambda x: x[2])

    '''
    clean format for numba
    '''
    projected_points = [
    [float(x) for x in row[0]] + row[1].tolist() + [row[2]]
    for row in projected_points
    ]

    ## Plot for debugging

    # plt.scatter([i[0] for i in just_proj],[i[1] for i in just_proj])
    # plt.plot([i[0] for i in projected_points],[i[1] for i in projected_points], linestyle = '-')

    # for pt in projected_points:
    #     x, y = pt[0], pt[1]
    #     value = pt[5]
    #     plt.text(x, y, str(value), fontsize=9, ha='left', va='bottom')

    # plt.show()

    '''
    projected_points has (ordered) structure: 
        projected_points[x][0:1] = projected coords (2d coordinate system aligned w/ plane)
        projected_points[x][2:4] = original coords (determine over under)
        projected_points[x][5] = value at original coords (sequence)
    '''

    return np.array(projected_points, dtype=np.float64)


@njit()
def lattice_writhe_Cimasoni(array, projections_111, projections_1m11, projections_11m1, projections_1m1m1):
    '''
    Want to explore Tait numbers T(A_{i}) on the 4 areas (8 areas modulo symmetry) on the indicatrix corresponding to projections on: 
    (pi, e/2, sqrt(2)/2), (pi, -e/2, sqrt(2)/2), (pi, e/2, -sqrt(2)/2), (pi, -e/2, -sqrt(2)/2).
    Additionally, need to have defined direction to capture +,- crossings: cross product of a fixed orientation along knot.
    '''

    TA = 0
    projections = np.stack((projections_111, projections_1m11, projections_11m1, projections_1m1m1))

    for x_th, x_th_proj in enumerate(projections):

        if x_th == 0:
            proj = np.array([np.pi,np.e/2,np.sqrt(2)/2], dtype=np.float64) # (pi, e, sqrt(2))
        elif x_th == 1:
            proj = np.array([np.pi,-np.e/2,np.sqrt(2)/2], dtype=np.float64)
        elif x_th == 2:
            proj = np.array([np.pi,np.e/2,-np.sqrt(2)/2], dtype=np.float64)
        else:
            proj = np.array([np.pi,-np.e/2,-np.sqrt(2)/2], dtype=np.float64)

        intersections = []
        wr = 0
        # for idx, i in enumerate(x_th_proj):
        for idx in prange(len(x_th_proj)):
            i = x_th_proj[idx]
            '''
            1. projections (1, 1, 1)
            Method:
            '''

            x1 = i[0]
            y1 = i[1] 
            val1 = i[5]

            x2 = x_th_proj[(idx + 1)%len(x_th_proj)][0]
            y2 = x_th_proj[(idx + 1)%len(x_th_proj)][1] 

            orig_x1 = i[2]
            orig_y1 = i[3]
            orig_z1 = i[4]

            orig_x2 = x_th_proj[(idx + 1)%len(x_th_proj)][2]
            orig_y2 = x_th_proj[(idx + 1)%len(x_th_proj)][3]
            orig_z2 = x_th_proj[(idx + 1)%len(x_th_proj)][4]

            '''
            Ignoring conditions.
            '''
            
            # for jdx, j in enumerate(x_th_proj):
            for jdx in range(len(x_th_proj)):
                
                '''
                Logic here to avoid including crossings occuring between sequential segments.
                Additional logic to avoid including crossings from lines lying on top of each other.
                '''
                if jdx != idx and jdx!=(idx-1)%len(x_th_proj) and jdx!=(idx+1)%len(x_th_proj):

                    j = x_th_proj[jdx]
                    x3 = j[0]
                    y3 = j[1] 

                    val2 = j[5]
                    x4 = x_th_proj[(jdx + 1)%len(x_th_proj)][0] 
                    y4 = x_th_proj[(jdx + 1)%len(x_th_proj)][1] 

                    dt = ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
                    ds = ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
                    if dt == 0 or ds == 0:
                        continue

                    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4))/dt
                    s = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2))/ds
                    
                    if 0<=s<=1:
                        if 0<=t<=1:

                            ip_x = x1 + t * (x2 - x1)
                            ip_y = y1 + t * (y2 - y1)

                            if [np.round(ip_x, decimals=2), np.round(ip_y, decimals=2)] not in intersections:
                                intersections.append([np.round(ip_x, decimals=2), np.round(ip_y, decimals=2)])

                                '''
                                Determine + or -.
                                Req. over under and orientation.
                                '''
                                
                                orig_x3 = j[2]
                                orig_y3 = j[3]
                                orig_z3 = j[4]

                                orig_x4 = x_th_proj[(jdx + 1)%len(x_th_proj)][2]
                                orig_y4 = x_th_proj[(jdx + 1)%len(x_th_proj)][3]
                                orig_z4 = x_th_proj[(jdx + 1)%len(x_th_proj)][4]
            
                                vect_1 = [orig_x2-orig_x1, orig_y2-orig_y1, orig_z2-orig_z1]
                                vect_2 = [orig_x4-orig_x3, orig_y4-orig_y3, orig_z4-orig_z3]

                                cross = np.cross(vect_1, vect_2)
                                dot_v = np.dot(cross, proj)
                                sign_vector_orientation = np.sign(dot_v)

                                distance = np.array([orig_x3-orig_x1, orig_y3-orig_y1, orig_z3-orig_z1], dtype=np.float64)
                                dot_d = np.dot(distance, proj)
                                sign_distance = np.sign(dot_d)

                                sign = sign_distance * sign_vector_orientation
                                # rh, lh convention
                                if sign > 0:
                                    wr -= 1 * (abs(val1-val2)%(len(np.argwhere(array>1))/2))/(np.linalg.norm(distance))
                                elif sign < 0: 
                                    wr += 1 * (abs(val1-val2)%(len(np.argwhere(array>1))/2))/(np.linalg.norm(distance))

        TA += wr
    TA = TA/4
    return TA

def BFACF(array_dict, timesteps):
    '''
    BFACF with chosen sampling methods
    '''
    alpha = 0

    init_array = dict(array_dict)
    max_x = max(p[0] for p in init_array) + 1
    max_y = max(p[1] for p in init_array) + 1
    max_z = max(p[2] for p in init_array) + 1
    init2array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (x, y, z), val in init_array.items():
        init2array[x, y, z] = val

    projections_111 = points_on_axis(init2array, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
    projections_1m11 = points_on_axis(init2array, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
    projections_11m1 = points_on_axis(init2array, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
    projections_1m1m1 = points_on_axis(init2array, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))

    old_writhe_energy = lattice_writhe_Cimasoni(init2array, 
                                            projections_111=projections_111,
                                            projections_1m11=projections_1m11,
                                            projections_11m1=projections_11m1,
                                            projections_1m1m1=projections_1m1m1)
    
    old_crumple_energy = crumple(init_array)
    
    old_energy = alpha * old_crumple_energy + (1-alpha) * old_writhe_energy

    for time in range(timesteps):
        
        print(f"simulation: {time/timesteps}")

        update_array = dict(array_dict)
        valid_indicies = [pos for pos, val in array_dict.items() if val > 1]

        random_edge = random.choice(valid_indicies)
        new_edge = find_new(update_array,random_edge)

        if new_edge == (-1, -1, -1, -1):
            continue

        old_val = array_dict[random_edge]
        del update_array[random_edge]
        update_array[new_edge[:3]] = update_array.get(new_edge[:3], 0) + old_val

        status = check_verticies(update_array)
        if status < -2:
            continue
        else:
            max_x = max(p[0] for p in update_array) + 1
            max_y = max(p[1] for p in update_array) + 1
            max_z = max(p[2] for p in update_array) + 1
            update2array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
            for (x, y, z), val in update_array.items():
                update2array[x, y, z] = val

            projections_111 = points_on_axis(update2array, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
            projections_1m11 = points_on_axis(update2array, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
            projections_11m1 = points_on_axis(update2array, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
            projections_1m1m1 = points_on_axis(update2array, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))

            new_writhe_energy = lattice_writhe_Cimasoni(update2array, 
                                                    projections_111=projections_111,
                                                    projections_1m11=projections_1m11,
                                                    projections_11m1=projections_11m1,
                                                    projections_1m1m1=projections_1m1m1)
            
            new_crumple_energy = crumple(update_array)
            new_energy = alpha * new_crumple_energy + (1-alpha) * new_writhe_energy
            

            if metropolis_acceptance(old_energy=old_energy, new_energy=new_energy, temperature=0.01):
                array_dict = update_array
                old_energy = new_energy

    print(new_energy)
    return array_dict
        
def pivot(array_dict, timesteps, knot):
    '''
    Pivot algorithm to increase autocorrelation of samples.
    Notice: valid pivots occur on a shared axis in Z^{3}
    Can implement writhe here to try pivot to more writhed config.
    Need to change so that it uses dictionary for speed.
    '''

    init_array = dict(array_dict)
    max_x = max(p[0] for p in init_array) + 1
    max_y = max(p[1] for p in init_array) + 1
    max_z = max(p[2] for p in init_array) + 1
    init2array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (x, y, z), val in init_array.items():
        init2array[x, y, z] = val

    projections_111 = points_on_axis(init2array, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
    projections_1m11 = points_on_axis(init2array, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
    projections_11m1 = points_on_axis(init2array, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
    projections_1m1m1 = points_on_axis(init2array, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))

    old_writhe_energy = lattice_writhe_Cimasoni(init2array, 
                                            projections_111=projections_111,
                                            projections_1m11=projections_1m11,
                                            projections_11m1=projections_11m1,
                                            projections_1m1m1=projections_1m1m1)

    for time in range(timesteps):

        update_dict = dict(array_dict)
        valid_indicies = [pos for pos, val in update_dict.items() if val > 0]

        random_edge_1, random_edge_2 = random.sample(valid_indicies, 2)

        if update_dict[random_edge_2]>update_dict[random_edge_1]:
            w1 = [i for i in range(update_dict[random_edge_1], update_dict[random_edge_2]+1)]
            axis = np.array(random_edge_2) - np.array(random_edge_1)

        else:
            w1 = [i for i in range(update_dict[random_edge_2], update_dict[random_edge_1]+1)]
            axis = np.array(random_edge_1) - np.array(random_edge_2)

        u = axis/np.linalg.norm(axis)
        ux, uy, uz = u[0], u[1], u[2]

        pivot_point = random_edge_1
        ang = np.random.choice([np.pi/2, -np.pi/2, np.pi])

        R = np.array([
            [ux**2*(1-np.cos(ang))+np.cos(ang), ux*uy*(1-np.cos(ang))-uz*np.sin(ang), ux*uz*(1-np.cos(ang))+uy*np.sin(ang)],
            [ux*uy*(1-np.cos(ang))+uz*np.sin(ang), uy**2*(1-np.cos(ang))+np.cos(ang), uy*uz*(1-np.cos(ang))-ux*np.sin(ang)],
            [ux*uz*(1-np.cos(ang))-uy*np.sin(ang), uy*uz*(1-np.cos(ang))+ux*np.sin(ang), uz**2*(1-np.cos(ang))+np.cos(ang)]])
        
        invalid = False
        for x in w1:
            index = next((k for k, v in update_dict.items() if np.array_equal(v, x)), None)
            if index == None:
                continue

            translated_index = np.array(index) - pivot_point 

            if len(translated_index)>0:

                new_index = np.dot(R, translated_index)
                new_index = np.round(new_index + pivot_point).astype(int) 
                del update_dict[index]  # Remove the old key
                update_dict[tuple(new_index)] = x 

        status = check_verticies(update_dict)
        print(status)
        if invalid == True:
            status = -np.inf
        
        if status < -2:
            continue
        else:
            topo = Q_invariant(update_dict, 'Uq(sl2)').alexander_polynomial_hash(knot) 
            if topo == True:
                max_x = max(p[0] for p in update_dict) + 1
                max_y = max(p[1] for p in update_dict) + 1
                max_z = max(p[2] for p in update_dict) + 1
                update2array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
                for (x, y, z), val in update_dict.items():
                    update2array[x, y, z] = val

                projections_111 = points_on_axis(update2array, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
                projections_1m11 = points_on_axis(update2array, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
                projections_11m1 = points_on_axis(update2array, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
                projections_1m1m1 = points_on_axis(update2array, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))

                new_writhe_energy = lattice_writhe_Cimasoni(update2array, 
                                                        projections_111=projections_111,
                                                        projections_1m11=projections_1m11,
                                                        projections_11m1=projections_11m1,
                                                        projections_1m1m1=projections_1m1m1)
            

                if metropolis_acceptance(old_energy=old_writhe_energy, new_energy=new_writhe_energy, temperature=0.01):
                    array_dict = update_dict
                    old_writhe_energy = new_writhe_energy
                    print(random_edge_1, random_edge_2)
            
            else:
                continue

    return array_dict
