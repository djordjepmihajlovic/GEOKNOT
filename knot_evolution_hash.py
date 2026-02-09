import numpy as np
from numba import njit, prange
from knot_init import *
from knot_invs import *
import random
from knot_functions import *

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

'''

def neighbours(array_dict, point):
    """
    Input:
    Returns the full 3x3x3 neighborhood of a point in 3D space.
    Each neighbor is a (x, y, z, value) tuple.
    Output:
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
    Input:
    Takes in array and point in array and outputs an array of neighbours and neighbour value.
    Output:
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
    Input: 
    For fixing orientation: make copy of hash table, assign 1 to some point and N to one of its neighbours. Now impose that every point (excluding)
    1 and N must have a neighbour with +1 value and -1 value of current value.
    Save both oriented and unoriented structure, use unoriented structure (just 1's) for some calculations.
    Output: 
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
    Input: 
    Find valid locations to move edge, nb. needs to be restricted neighbours
    Output: 
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
    Input:
    Checks for singular points.
    Output:
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
    Input:
    Checks the vertices of the 3D state space using dictionary-based storage.
    Output:
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
    Input:
    Markov Chain method to enforce movement toward more crumpled structure.
    Defines energy as sum of distance between all indices in the dictionary.
    Output:
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
    Input:
    Define arcs/loops (some subsection of the knot) 
    calculate the center of mass of chosen loop. find plane that arc closes 
    move a chosen point through the perturbed center of mass.
    Output:
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
    Input:
    Measures spatial proximity between distant points in sequence.
    
    sequence_threshold: minimum "sequence" distance between points to be considered long-range
    distance_threshold: maximum Euclidean distance in space to count as close contact
    Returns:
    score counting long-range spatial entanglements
    Output:
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
def lattice_writhe_Cimasoni(projections_111, projections_1m11, projections_11m1, projections_1m1m1):
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


def _fn1_wrapper(array_dict, fn_1=None):
    """
    Compute first geometric functional for bias. If fn_1=None will automatically set to writhe.
    """
    # build dense array
    max_x = max(p[0] for p in array_dict) + 1
    max_y = max(p[1] for p in array_dict) + 1
    max_z = max(p[2] for p in array_dict) + 1
    dense = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (x, y, z), val in array_dict.items():
        dense[x, y, z] = val

    # legacy default
    if fn_1 is None:
        projections_111 = points_on_axis(dense, np.array([np.pi, np.e/2, np.sqrt(2)/2])) 
        projections_1m11 = points_on_axis(dense, np.array([np.pi, -(np.e)/2, np.sqrt(2)/2])) 
        projections_11m1 = points_on_axis(dense, np.array([np.pi, np.e/2, -(np.sqrt(2))/2]))
        projections_1m1m1 = points_on_axis(dense, np.array([np.pi, -(np.e)/2, -(np.sqrt(2))/2]))
        return lattice_writhe_Cimasoni(projections_111=projections_111,
                                      projections_1m11=projections_1m11,
                                      projections_11m1=projections_11m1,
                                      projections_1m1m1=projections_1m1m1)

    # try flexible call signatures for custom function
    try:
        return fn_1(array_dict)
    except Exception as e:
        raise RuntimeError(f"Provided fn_1 raised an error: {e}")


def _fn2_wrapper(array_dict, fn_2=None):
    """
    Compute entanglement-like observable. `entanglement_fn` should accept
    either `array_dict` or a dense numpy array. If None, falls back to
    `long_range_entanglement`.
    """
    if fn_2 is None:
        return long_range_entanglement(array_dict)

    # try flexible call signatures for custom function
    try:
        return fn_2(array_dict)
    except Exception as e:
        raise RuntimeError(f"Provided fn_2 raised an error: {e}")


def BFACF(array_dict, timesteps, aimed_range, fn_1=None, fn_2=None):
    '''
    BFACF with chosen sampling methods
    '''
    
    fn1_data = []
    fn2_data = []
    count_data = []

    alpha = 1

    max_fn1 = max(aimed_range[0])
    min_fn1 = min(aimed_range[0])
    max_fn2 = max(aimed_range[1])
    min_fn2 = min(aimed_range[1])
    
    random_number = random.uniform(0, 1)
    target_fn1 = (max_fn1 + min_fn1) / 2 + random_number * (max_fn1 - min_fn1) / 2
    target_fn2 = (max_fn2 + min_fn2) / 2 + random_number * (max_fn2 - min_fn2) / 2

    init_array = dict(array_dict)
    max_x = max(p[0] for p in init_array) + 1
    max_y = max(p[1] for p in init_array) + 1
    max_z = max(p[2] for p in init_array) + 1
    init2array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (x, y, z), val in init_array.items():
        init2array[x, y, z] = val

    # compute initial observables via wrappers (allows user-supplied functions)
    old_fn1_energy = _fn1_wrapper(init_array, fn_1=fn_1)
    old_fn2_energy = _fn2_wrapper(init_array, fn_2=fn_2)
    old_energy = alpha * old_fn1_energy + (1 - alpha) * old_fn2_energy

    phase = 1

    for time in range(timesteps):
        if phase == 1 and target_fn2 - 10*(target_fn2/100) < old_fn2_energy < target_fn2 + 10*(target_fn2/100):
            print(f"Moved to phase 2 at: {time} steps")
            alpha = 0
            phase = 2
            old_energy = old_fn1_energy

        elif phase == 2 and target_fn1 - 10*(target_fn1/100) < old_fn1_energy < target_fn1 + 10*(target_fn1/100):
            print(f"Target reached in: {time} steps")
            break

        fn1_data.append(old_fn1_energy)
        fn2_data.append(old_fn2_energy)
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

            # compute new observables using wrappers
            new_fn1_energy = _fn1_wrapper(update_array, fn_1=fn_1)
            new_fn2_energy = _fn2_wrapper(update_array, fn_2=fn_2)
            new_energy = alpha * new_fn2_energy + (1 - alpha) * new_fn1_energy
            
            temp = 0.00001

            if phase == 1:
                if new_fn2_energy < target_fn2:

                    if metropolis_acceptance(old_energy=old_energy, new_energy=new_energy, temperature=temp):
                        array_dict = update_array
                        old_fn2_energy = new_fn2_energy
                        old_fn1_energy = new_fn1_energy
                        old_energy = new_energy
                    else:
                        continue
                else:
                    if metropolis_acceptance(old_energy=-old_energy, new_energy=-new_energy, temperature=temp):
                        array_dict = update_array
                        old_fn2_energy = new_fn2_energy
                        old_fn1_energy = new_fn1_energy
                        old_energy = new_energy
                    else:
                        continue

            elif phase == 2:
                # Hard constraint only accept if deviation within 10% of target
                tol_fn2 = 0.1
                if new_fn1_energy < target_fn1:
                    if not (target_fn2*(1 - tol_fn2) <= new_fn2_energy <= target_fn2*(1 + tol_fn2)):
                        continue
                    if metropolis_acceptance(old_energy=old_energy, new_energy=new_energy, temperature=temp):
                        array_dict = update_array
                        old_fn2_energy = new_fn2_energy
                        old_fn1_energy = new_fn1_energy
                        old_energy = new_energy
                    else:
                        continue

    return array_dict, old_fn1_energy, old_fn2_energy


def pivot(array_dict, timesteps, knot, aimed_range, fn_1=None, fn_2=None):
    '''
    Pivot algorithm to increase autocorrelation of samples.
    Notice: valid pivots occur on a shared axis in Z^{3}
    '''

    fn1_data = []
    fn2_data = []
    count_data = []

    # Randomly pick number that determines minimizing/ maximizing/ doing nothing for writhe
    max_fn1 = max(aimed_range[0])
    min_fn1 = min(aimed_range[0])
    max_fn2 = max(aimed_range[1])
    min_fn2 = min(aimed_range[1])

    random_number = random.uniform(0, 1)

    target_fn1 = (max_fn1 + min_fn1) / 2 + random_number * (max_fn1 - min_fn1) / 2
    target_fn2 = (max_fn2 + min_fn2) / 2 + random_number * (max_fn2 - min_fn2) / 2

    init_array = dict(array_dict)
    max_x = max(p[0] for p in init_array) + 1
    max_y = max(p[1] for p in init_array) + 1
    max_z = max(p[2] for p in init_array) + 1
    init2array = np.zeros((max_x, max_y, max_z), dtype=np.float64)
    for (x, y, z), val in init_array.items():
        init2array[x, y, z] = val

    # compute initial observables via wrappers
    old_fn1_energy = _fn1_wrapper(init_array, fn_1=fn_1)
    old_fn2_energy = _fn2_wrapper(init_array, fn_2=fn_2)
    phase = 1
    alpha = 1
    old_energy = alpha * old_fn2_energy + (1 - alpha) * old_fn1_energy

    for time in range(timesteps):
        if phase == 1 and  target_fn2 - 10*(target_fn2/100) < old_fn2_energy < target_fn2 + 10*(target_fn2/100):
            print(f"Entanglement target reached in: {time} steps")
            phase = 2
            alpha = 0
            old_energy = old_fn1_energy
            break 

        count_data.append(time)
        fn1_data.append(old_fn1_energy)
        fn2_data.append(old_fn2_energy)

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

            # compute new observables using wrappers
            new_fn1_energy = _fn1_wrapper(update_dict, fn_1=fn_1)
            new_fn2_energy = _fn2_wrapper(update_dict, fn_2=fn_2)
            new_energy = alpha * new_fn2_energy + (1 - alpha) * new_fn1_energy
            
            # if new_writhe_energy < target_wr:
            #     if new_entanglement_energy < target_entang:

            temp = 0.0001 * (target_fn1 - new_fn1_energy) / target_fn1
            # temp = 0
            if old_fn2_energy < target_fn2:
                if metropolis_acceptance(old_energy=old_energy, new_energy=new_energy, temperature=temp):
                    array_dict = update_dict
                    old_fn1_energy = new_fn1_energy
                    old_fn2_energy = new_fn2_energy
                    old_energy = new_energy
                    
                else:
                    continue
            else:
                if metropolis_acceptance(old_energy=-old_energy, new_energy=-new_energy, temperature=temp):
                    # Reverse search
                    array_dict = update_dict
                    old_fn1_energy = new_fn1_energy
                    old_fn2_energy = new_fn2_energy
                    old_energy = new_energy
                    
                else:
                    continue

    return array_dict