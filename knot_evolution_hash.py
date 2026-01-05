import numpy as np
from numba import njit, prange
from knot_init import *
from knot_invs import *
import math
import random

'''
BFACF/Pivot algorithm with strictly evolving sampler implementation for flat wrt dos polygonal lattice knot embeddings.

Relevant literature: 
    * Lattice Knots: 
    * BFACF: 
    * Pivot: 
    * Cimasoni Writhe calculation O(nlog(n)): 
    * Klenin Writhe calculation O(n^{2}): 

Key Features:
    * Oriented lattice knots S^{1} in Z^{3}
    * Geometrically selective BFACF/Pivot updates
    * Writhe calculation (https://www.unige.ch/math/folks/cimasoni/writhe.pdf)
    * Visualization

Future Implementations:
    * Links
    * Knotoids 
    * Bonded knotoids
    * Proteins (spec. bonded knotoids (capture forces) ~ protein_init.py ~ load in protein from PDB and automate)
    * S^{2} in Z^{4}, (S^{n} in Z^{n+2})

To do (23/06/25):
    * I want to implement a mechanism (probably a bias in the BFACF moves) to ensure `threading' occurs to the knots
    * Idea:
    * define arcs/loops (some subsection of the knot) 
    * calculate the center of mass of chosen loop. find plane that arc closes 
    * move a chosen point through the perturbed center of mass.

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
def check_singularities(array_dict, new_edge, val):
    '''
    Checks for singular points.
    '''
    # new edge
    # get neighbours of new edge
    # construct new edge 'forward' and 'backward' vectors
    # first find position of forward and backward points
    forward_pos = list(array_dict.keys())[list(array_dict.values()).index(val+1)]
    backward_pos = list(array_dict.keys())[list(array_dict.values()).index(val-1)]
    forward_vector = np.array(forward_pos) - np.array(new_edge[:3])
    backward_vector = np.array(new_edge[:3]) - np.array(backward_pos)

    neighbours = neighbours(array_dict, new_edge)
    for i in neighbours:
        # construct the vector between neighbour and subsequent point
        if array_dict[tuple(i[:3])] == val or array_dict[tuple(i[:3])] == val+1 or array_dict[tuple(i[:3])] == val-1:
            continue

        else:
            neighbour_pos = np.array(i[:3]) 
            n_val = array_dict[tuple(i[:3])]
            n_forward_pos = list(array_dict.keys())[list(array_dict.values()).index(n_val+1)]
            n_backward_pos = list(array_dict.keys())[list(array_dict.values()).index(n_val-1)]
            # check if the vectors intersect
            n_forward_vector = np.array(n_forward_pos) - neighbour_pos
            n_backward_vector = neighbour_pos - np.array(n_backward_pos)
            

        
    singularity_status = True

    return singularity_status

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
            #if neighbor_val != 0:
                check.append(neighbor_val) # only have neighbours with value +1 or -1 (a bit more restricting but removes possibility of X intersections)

        if len([v for v in check if v > 0]) != 2:
            status -= 1

    return status


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

def threading(array_dict, min_length = 5, max_length = 20):
    '''
    Define arcs/loops (some subsection of the knot) 
    calculate the center of mass of chosen loop. find plane that arc closes 
    move a chosen point through the perturbed center of mass.
    '''

    sequence_length = random.randint(min_length, max_length)
    coords = list(array_dict.keys())
    vals = [array_dict[c] for c in coords]
    random_point_loop = random.choice(vals)

    sample_sequence = []
    for i in range(random_point_loop, random_point_loop+sequence_length):
        for coord in coords:
            if array_dict[coord] == i:
                sample_sequence.append(coord)

    print(f'Selected sequence: {sample_sequence}, with values: {[array_dict[c] for c in sample_sequence]}')

    centre_of_mass = np.mean(np.array(sample_sequence), axis=0)
    print(f'Centre of mass: {centre_of_mass}')

    random_point_thread = random.choice(vals)
    print(f'Random point for threading: {random_point_thread}')

    # find shared plane of points
    # perturb centre of mass along normal to plane
    print(sample_sequence[0])
    AC = np.array(sample_sequence[0]) - np.array(sample_sequence[1])
    AB = np.array(sample_sequence[0]) - np.array(sample_sequence[2])
    normal_vector = np.cross(AC, AB).astype(np.float64)  # Ensure float64 type
    normal_vector /= np.linalg.norm(normal_vector)
    perturbation = np.random.normal(scale=0.2, size=3)  # Small perturbation

    target_point = centre_of_mass + normal_vector * perturbation

    print(random_point_thread)

    return target_point, random_point_thread

def long_range_entanglement(array_dict, sequence_threshold=10, distance_threshold=5):
    '''
    Measures spatial proximity between distant points in sequence.
    
    sequence_threshold: minimum "sequence" distance between points to be considered long-range
    distance_threshold: maximum Euclidean distance in space to count as close contact
    Returns:
    score counting long-range spatial entanglements
    '''
    coords = list(array_dict.keys())
    vals = [array_dict[c] for c in coords]
    score = 0
    total_points = len(vals)

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            # Our knot is cyclic
            sequence_dist = min(abs(vals[i] - vals[j]), total_points - abs(vals[i] - vals[j]))
            if sequence_dist >= sequence_threshold:
                # Euclidean distance
                spatial_dist = np.sqrt(
                    (coords[i][0] - coords[j][0]) ** 2 +
                    (coords[i][1] - coords[j][1]) ** 2 +
                    (coords[i][2] - coords[j][2]) ** 2
                )
                if spatial_dist <= distance_threshold:
                    score += 1
    return score


def average_curvature(array_dict):
    '''
    Computes the average curvature of a 3D curve defined by coords.
    '''
    coords = list(array_dict.keys())
    n = len(coords)

    # Compute first and second derivatives 
    first_derivative = np.gradient(coords, axis=0)
    second_derivative = np.gradient(first_derivative, axis=0)

    # Compute curvature at each point
    curvature = []
    for i in range(n):
        cross_product = np.cross(first_derivative[i], second_derivative[i])
        numerator = np.linalg.norm(cross_product)
        denominator = np.linalg.norm(first_derivative[i]) ** 3
        if denominator != 0:
            curvature.append(numerator / denominator)
        else:
            curvature.append(0)
    return np.mean(curvature)

@njit()
def radius_of_gyration(array):
    '''
    Radius of gyration 
    '''
    indicies = np.argwhere(array > 0)
    center_of_mass = np.mean(indicies, axis=0)
    return np.sqrt(np.mean(np.sum((indicies - center_of_mass)**2, axis=1)))

@njit()
def gyration_tensor_and_eigenvalues(coords):
    '''
    Computes the gyration tensor and its eigenvalues.
    '''

    center_of_mass = np.mean(coords, axis=0)
    
    shifted_coords = coords - center_of_mass
    gyration_tensor = np.zeros((3, 3))
    for coord in shifted_coords:
        gyration_tensor += np.outer(coord, coord)
    gyration_tensor /= len(coords)
    eigenvalues = np.linalg.eigvalsh(gyration_tensor)
    
    return gyration_tensor, eigenvalues

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

def metropolis_acceptance(old_energy, new_energy, temperature):
    '''
    Metropolis acceptance criterion.
    '''

    if new_energy > old_energy: # (new writhe is larger)
        return True
    else:

        ### Want to implement a dynamically changing temperature
        acceptance_probability = np.exp((new_energy - old_energy) / temperature)
        return np.random.rand() < acceptance_probability
        # return False


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

    '''
    projected_points has (ordered) structure: 
        projected_points[x][0:1] = projected coords (2d coordinate system aligned w/ plane)
        projected_points[x][2:4] = original coords (determine over under)
        projected_points[x][5] = value at original coords (sequence)
    '''

    return np.array(projected_points, dtype=np.float64)


@njit()
def lattice_writhe_Cimasoni(array, no_points, projections_111, projections_1m11, projections_11m1, projections_1m1m1):
    '''
    Want to explore Tait numbers T(A_{i}) on the 4 areas (8 areas modulo symmetry) on the indicatrix corresponding to projections on: 
    (pi, e/2, sqrt(2)/2), (pi, -e/2, sqrt(2)/2), (pi, e/2, -sqrt(2)/2), (pi, -e/2, -sqrt(2)/2).
    Additionally, need to have defined direction to capture +,- crossings: cross product of a fixed orientation along knot.
    '''

    TA = 0
    projections = np.stack((projections_111, projections_1m11, projections_11m1, projections_1m1m1))
    lens = no_points

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
                                    wr -= 1 # * (abs(val1-val2)%(lens))/(np.linalg.norm(distance)**2)
                                elif sign < 0: 
                                    wr += 1 # * (abs(val1-val2)%(lens))/(np.linalg.norm(distance)**2)

        TA += wr
    TA = TA/4
    return TA

def lattice_writhe_Klenin(coord_list):
    '''
    Computes writhe using Klenin formulation.
    Input: list of points in 3D space and value.
    '''
    
    ringx = np.array([(x, y, z) for _, x, y, z in coord_list])
    vals = np.array([val for val, _, _, _ in coord_list])

    sorted_indices = np.argsort(vals)
    ring1 = ringx[sorted_indices]
    ring2 = ring1.copy()
    matrix = np.zeros((ring1.shape[0], ring2.shape[0]))
    # Loop on the first ring
    for i in prange(ring1.shape[0]):
        # Loop on the second ring
        for j in prange(ring2.shape[0]):
            matrix[i,j] = compute_single_sts_writhe(ring1, ring2, i, j, 2)
        print(i)
    return matrix


def compute_single_sts_writhe(ring1, ring2, i, j, lw):

    wr = 0
    
    # Loop over the segment of the first ring
    for it in prange(-np.int64(lw/2)+i,np.int64(lw/2)+i):
        # Loop over the segment of the second ring
        for jt in prange(-np.int64(lw/2)+j, np.int64(lw/2)+j): 
            
            one = ring1[np.mod(it-1,ring1.shape[0]),:]
            three = ring2[np.mod(jt-1,ring2.shape[0]),:]
            two = ring1[np.mod(it,ring1.shape[0]),:]
            four = ring2[np.mod(jt,ring2.shape[0]),:]

            r12=two-one
            r34=four-three
            r23=three-two
            r13=three-one
            r14=four-one
            r24=four-two

            n1 = np.cross(r13,r14)
            if np.linalg.norm(n1)==0:
                continue
            n1 = n1 / np.linalg.norm(n1)

            n2 = np.cross(r14,r24)
            if np.linalg.norm(n2)==0:
                continue
            n2 = n2 / np.linalg.norm(n2)

            n3 = np.cross(r24,r23)
            if np.linalg.norm(n3)==0:
                continue
            n3 = n3 / np.linalg.norm(n3)

            n4 = np.cross(r23,r13)
            if np.linalg.norm(n4)==0:
                continue
            n4 = n4 / np.linalg.norm(n4)

            n1n2=np.dot(n1,n2)
            n2n3=np.dot(n2,n3)
            n3n4=np.dot(n3,n4)
            n4n1=np.dot(n4,n1)

            cvec = np.cross(r34,r12)
            dprcvec = np.dot(cvec,r13)

            if dprcvec == 0:
                continue

            omega = (np.arcsin( n1n2 ) + np.arcsin( n2n3 ) + np.arcsin( n3n4 ) + np.arcsin( n4n1 ) ) * dprcvec/np.abs(dprcvec);
            
            wr+=omega/(4*np.pi)

    return 2*wr

### can reduce clutter here
### also need to make it so updates are hard

def BFACF(array_dict, timesteps, aimed_range):
    '''
    BFACF with chosen sampling methods
    '''
    # Gradient descent towards entanglement first.
    # Need to generalize this.
    
    wr_data = []
    ent_data = []
    count_data = []

    alpha = 1

    max_wr = max(aimed_range[0])
    min_wr = min(aimed_range[0])
    max_entang = max(aimed_range[1])
    min_entang = min(aimed_range[1])
    
    random_number = random.uniform(0, 1)
    target_wr = (max_wr + min_wr) / 2 + random_number * (max_wr - min_wr) / 2
    target_entang = (max_entang + min_entang) / 2 + random_number * (max_entang - min_entang) / 2

    init_array = dict(array_dict)
    no_points = len(init_array)
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

    old_writhe_energy = lattice_writhe_Cimasoni(init2array, no_points, 
                                            projections_111=projections_111,
                                            projections_1m11=projections_1m11,
                                            projections_11m1=projections_11m1,
                                            projections_1m1m1=projections_1m1m1)
    
    # old_entanglement_energy = long_range_entanglement(init_array)
    old_entanglement_energy = average_curvature(init_array)
    
    old_energy = alpha * old_entanglement_energy + (1-alpha) * old_writhe_energy

    # old_curve_energy = average_curvature(init2array)

    phase = 1

    for time in range(timesteps):
        if phase == 1 and target_entang - 10*(target_entang/100) < old_entanglement_energy < target_entang + 10*(target_entang/100):
            print(f"Moved to phase 2 at: {time} steps")
            alpha = 0
            phase = 2
            old_energy = old_writhe_energy

        elif phase == 2 and target_wr - 10*(target_wr/100) < old_writhe_energy < target_wr + 10*(target_wr/100):
            print(f"Target reached in: {time} steps")
            break

        wr_data.append(old_writhe_energy)
        ent_data.append(old_entanglement_energy)
        count_data.append(time)
        
        # print(f"simulation: {time/timesteps}")
        if time % (timesteps/10) == 0:
            print(f"BFACF: {time/timesteps}")

        update_array = dict(array_dict)
        valid_indicies = [pos for pos, val in array_dict.items() if val > 1]

        random_edge = random.choice(valid_indicies)
        new_edge = find_new(update_array,random_edge)

        if new_edge == (-1, -1, -1, -1):
            continue

        old_val = array_dict[random_edge]
        del update_array[random_edge]
        update_array[new_edge[:3]] = update_array.get(new_edge[:3], 0) + old_val

        # New function to check for singular points, takes in the updated array edge checks connecting strands (forward and backward) does a sweep to check for intersections
        # Only needs to sweep neighbours of the new edge not the entire array

        status = check_verticies(update_array)
        if status < -2:#< -2: Valid configurations.
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

            new_writhe_energy = lattice_writhe_Cimasoni(update2array, no_points, 
                                                    projections_111=projections_111,
                                                    projections_1m11=projections_1m11,
                                                    projections_11m1=projections_11m1,
                                                    projections_1m1m1=projections_1m1m1)
            
            # new_entanglement_energy = long_range_entanglement(update_array)
            new_entanglement_energy = average_curvature(update_array)

            new_energy = alpha * new_entanglement_energy + (1-alpha) * new_writhe_energy
            
            temp = 0.00001

            if phase == 1:
                if new_entanglement_energy < target_entang:

                    if metropolis_acceptance(old_energy=old_energy, new_energy=new_energy, temperature=temp):
                        array_dict = update_array
                        old_entanglement_energy = new_entanglement_energy
                        old_writhe_energy = new_writhe_energy
                        old_energy = new_energy
                    else:
                        continue
                else:
                    if metropolis_acceptance(old_energy=-old_energy, new_energy=-new_energy, temperature=temp):
                        array_dict = update_array
                        old_entanglement_energy = new_entanglement_energy
                        old_writhe_energy = new_writhe_energy
                        old_energy = new_energy
                    else:
                        continue

            elif phase == 2:
                # Hard constraint only accept if deviation within 10% of target
                tol_ent = 0.1
                if new_writhe_energy < target_wr:
                    if not (target_entang*(1 - tol_ent) <= new_entanglement_energy <= target_entang*(1 + tol_ent)):
                        continue
                    if metropolis_acceptance(old_energy=old_energy, new_energy=new_energy, temperature=temp):
                        array_dict = update_array
                        old_entanglement_energy = new_entanglement_energy
                        old_writhe_energy = new_writhe_energy
                        old_energy = new_energy
                    else:
                        continue

    return array_dict, old_writhe_energy, old_entanglement_energy


def pivot(array_dict, timesteps, knot, aimed_range):
    '''
    Pivot algorithm to increase autocorrelation of samples.
    Notice: valid pivots occur on a shared axis in Z^{3}
    '''

    wr_data = []
    ent_data = []
    count_data = []

    # Randomly pick number that determines minimizing/ maximizing/ doing nothing for writhe
    max_wr = max(aimed_range[0])
    min_wr = min(aimed_range[0])
    max_entang = max(aimed_range[1])
    min_entang = min(aimed_range[1])

    random_number = random.uniform(0, 1)

    target_wr = (max_wr + min_wr) / 2 + random_number * (max_wr - min_wr) / 2
    target_entang = (max_entang + min_entang) / 2 + random_number * (max_entang - min_entang) / 2

    init_array = dict(array_dict)
    no_points = len(init_array)
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

    old_writhe_energy = lattice_writhe_Cimasoni(init2array, no_points,
                                            projections_111=projections_111,
                                            projections_1m11=projections_1m11,
                                            projections_11m1=projections_11m1,
                                            projections_1m1m1=projections_1m1m1)
    
    # old_entanglement_energy = long_range_entanglement(init_array)
    old_entanglement_energy = average_curvature(init_array)

    phase = 1
    alpha = 1

    old_energy = alpha * old_entanglement_energy + (1-alpha) * old_writhe_energy

    for time in range(timesteps):
        if phase == 1 and  target_entang - 10*(target_entang/100) < old_entanglement_energy < target_entang + 10*(target_entang/100):
            print(f"Entanglement target reached in: {time} steps")
            phase = 2
            alpha = 0
            old_energy = old_writhe_energy
            break 

        count_data.append(time)
        wr_data.append(old_writhe_energy)
        ent_data.append(old_entanglement_energy)

        if time % (timesteps/10) == 0:
            print(f"Pivot: {time/timesteps}")

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
        if invalid == True:
            status = -np.inf
        
        if status < -2: 
            continue
        else:
            
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

            new_writhe_energy = lattice_writhe_Cimasoni(update2array, no_points,
                                                    projections_111=projections_111,
                                                    projections_1m11=projections_1m11,
                                                    projections_11m1=projections_11m1,
                                                    projections_1m1m1=projections_1m1m1)
            
            # new_entanglement_energy = long_range_entanglement(update_dict)
            new_entanglement_energy = average_curvature(update_dict)
            new_energy = alpha * new_entanglement_energy + (1-alpha) * new_writhe_energy
            
            # if new_writhe_energy < target_wr:
            #     if new_entanglement_energy < target_entang:

            temp = 0.0001 * (target_wr - new_writhe_energy) / target_wr
            # temp = 0
            if old_entanglement_energy < target_entang:
                if metropolis_acceptance(old_energy=old_energy, new_energy=new_energy, temperature=temp):
                    array_dict = update_dict
                    old_writhe_energy = new_writhe_energy
                    old_entanglement_energy = new_entanglement_energy
                    old_energy = new_energy
                    
                else:
                    continue
            else:
                if metropolis_acceptance(old_energy=-old_energy, new_energy=-new_energy, temperature=temp):
                    # Reverse search
                    array_dict = update_dict
                    old_writhe_energy = new_writhe_energy
                    old_entanglement_energy = new_entanglement_energy
                    old_energy = new_energy
                    
                else:
                    continue

    return array_dict

def bias_energy(ent, wr, target_ent, target_wr, k_ent=1.0, k_wr=1.0):
    """
    Harmonic bias centered on targets
    """
    return k_ent * (ent - target_ent)**2 + k_wr * (wr - target_wr)**2


def metropolis_biased_accept(old_ent, old_wr, new_ent, new_wr,
                             target_ent, target_wr,
                             k_ent=1.0, k_wr=1.0, temp=1e-3):
    """
    Compute biased energies and perform a Metropolis accept/reject.
    """
    old_E = bias_energy(old_ent, old_wr, target_ent, target_wr, k_ent, k_wr)
    new_E = bias_energy(new_ent, new_wr, target_ent, target_wr, k_ent, k_wr)

    dE = new_E - old_E
    if dE <= 0:
        return True, new_E, old_E
    else:
        if temp <= 0:
            return False, new_E, old_E
        p = math.exp(-dE / temp)
        if random.random() < p:
            return True, new_E, old_E
        else:
            return False, new_E, old_E

def dict_to_dense(array_dict):
    """
    Turn dict into dense numpy array.
    """
    max_x = max(p[0] for p in array_dict) + 1
    max_y = max(p[1] for p in array_dict) + 1
    max_z = max(p[2] for p in array_dict) + 1
    arr = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (x, y, z), val in array_dict.items():
        arr[x, y, z] = val
    return arr

def compute_observables(update_dict, alpha, no_points):

    update2array = dict_to_dense(update_dict)

    projections_111 = points_on_axis(update2array, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
    projections_1m11 = points_on_axis(update2array, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
    projections_11m1 = points_on_axis(update2array, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
    projections_1m1m1 = points_on_axis(update2array, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))

    new_writhe_energy = lattice_writhe_Cimasoni(update2array, no_points,
                                            projections_111=projections_111,
                                            projections_1m11=projections_1m11,
                                            projections_11m1=projections_11m1,
                                            projections_1m1m1=projections_1m1m1)
    
    # new_entanglement_energy = long_range_entanglement(update_dict)
    new_entanglement_energy = average_curvature(update_dict)
    new_energy = alpha * new_entanglement_energy + (1-alpha) * new_writhe_energy

    return new_energy

def bias_energy(wr, ent, target_wr, target_ent, sigma_wr, sigma_ent):
    return 0.5 * ((wr - target_wr)**2 / sigma_wr**2 
                  + (ent - target_ent)**2 / sigma_ent**2)