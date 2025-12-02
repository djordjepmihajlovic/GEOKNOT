import numpy as np
from numba import njit
import kymoknot 
from kymoknot.searchtype import SearchType
from knot_init import *

'''
This module calculates quantum invariants of a knot to determine knot type (not preserved in pivot algorithm).
Takes in state space array and returns a Q_invariant object.

Method:
    * Take in array of knot.
    * Project onto flat plane (non rational normal to avoid problems).
    * Define Tensor category.
    * Forming the sequence.
    * Build equation.
    * Solves final equation (sequence of tensors multiplied by R^{x}).
        
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

    projected_points = [
    [float(x) for x in row[0]] + row[1].tolist() + [row[2]]
    for row in projected_points
    ]

    ##################### ISSUE HERE BECAUSE I UPDATED HOW KNOT DATA WAS PARSED ###########################
    ##################### Need to order knots according to their value ####################################
    print(np.array(projected_points, dtype=np.float64))

    return np.array(projected_points, dtype=np.float64)


def scan(projection):
    '''
    Scans from bottom to top and extracts different horizontal splittings of diagram to compose tensor equation.
    Need to jitify this function.
    '''

    splits = np.linspace(np.min(projection[:, 1:2]), np.max(projection[:, 1:2]), num=1000)

    proj = projection[:, 0:2]
    proj_scan = []
    splittings = []

    for jdx, j in enumerate(splits):

        x3 = 1000
        y3 = j
        x4 = -1000
        y4 = j

        intersections = []

        for idx, i in enumerate(proj):

            x1 = i[0]
            y1 = i[1]
            x2 = proj[(idx + 1)%len(proj)][0]
            y2 = proj[(idx + 1)%len(proj)][1]

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

                    ## check y increasing or decreasing
                    diff_y = y2 - y1
                    value = np.sign(diff_y)

                    point = [np.round(ip_x, 2), np.round(ip_y, 2)]
                    if not any(existing[:2] == point for existing in intersections):
                        intersections.append(point + [value])
        
        intersections = sorted(intersections, key=lambda x: x[0])
        if len(proj_scan) == 0:
            if len(intersections)%2==0:
                # splittings.append([intersections[0][1], [i[2] for i in intersections]])
                splittings.append([i for i in intersections])
                proj_scan.append([i[2] for i in intersections])
        
        else:
            if proj_scan[-1] != [i[2] for i in intersections]:
                if len(intersections)%2==0:
                    proj_scan.append([i[2] for i in intersections])
                    # splittings.append([intersections[0][1], [i[2] for i in intersections]])
                    splittings.append([i for i in intersections])

    '''
    splittings is a list of form [[[x, y, v], [x, y, dv]] ... []]
    '''
    return splittings

def crossing(projection, axis):

    intersections = []
    for idx, i in enumerate(projection):
        '''
        1. projections (1, 1, 1)
        Method:

        '''

        x1 = i[0]
        y1 = i[1] 
        x2 = projection[(idx + 1)%len(projection)][0]
        y2 = projection[(idx + 1)%len(projection)][1] 

        orig_x1 = i[2]
        orig_y1 = i[3]
        orig_z1 = i[4]

        orig_x2 = projection[(idx + 1)%len(projection)][2]
        orig_y2 = projection[(idx + 1)%len(projection)][3]
        orig_z2 = projection[(idx + 1)%len(projection)][4]

        '''
        Ignoring conditions.
        '''
        
        for jdx, j in enumerate(projection):
            '''
            Logic here to avoid including crossings occuring between sequential segments.
            Additional logic to avoid including crossings from lines lying on top of each other.
            '''
            if jdx != idx and jdx!=(idx-1)%len(projection) and jdx!=(idx+1)%len(projection):

                x3 = j[0]
                y3 = j[1] 
                x4 = projection[(jdx + 1)%len(projection)][0] 
                y4 = projection[(jdx + 1)%len(projection)][1] 

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

                        point = [np. round(ip_x, 5), np.round(ip_y, 5)]

                        # if [np.round(ip_x, decimals=2), np.round(ip_y, decimals=2)] not in intersections:
                        #     intersections.append([np.round(ip_x, decimals=2), np.round(ip_y, decimals=2)])
                        if not any(existing[:2] == point for existing in intersections):

                            '''
                            Determine + or -.
                            Req. over under and orientation.
                            '''
                            
                            orig_x3 = j[2]
                            orig_y3 = j[3]
                            orig_z3 = j[4]

                            orig_x4 = projection[(jdx + 1)%len(projection)][2]
                            orig_y4 = projection[(jdx + 1)%len(projection)][3]
                            orig_z4 = projection[(jdx + 1)%len(projection)][4]
        
                            vect_1 = [orig_x2-orig_x1, orig_y2-orig_y1, orig_z2-orig_z1]
                            vect_2 = [orig_x4-orig_x3, orig_y4-orig_y3, orig_z4-orig_z3]

                            cross = np.cross(vect_1, vect_2)
                            dot_v = np.dot(cross, axis)
                            sign_vector_orientation = np.sign(dot_v)

                            distance = np.array([orig_x3-orig_x1, orig_y3-orig_y1, orig_z3-orig_z1], dtype=np.float64)
                            dot_d = np.dot(distance, axis)
                            sign_distance = np.sign(dot_d)

                            sign = sign_distance * sign_vector_orientation
                            # rh, lh convention
                            if sign > 0:
                                wr = -1
                            elif sign < 0: 
                                wr = 1

                            intersections.append(point + [wr])
    return intersections
        

class Q_invariant:
    def __init__(self, array, q_group):
        '''
        Knot array of state space, specified quantum group and projection.
        '''
        self.array = array
        self.q_group = q_group

    def alexander_polynomial(self, knot):
        '''
        KymoKnot: https://github.com/luca-tubiana/KymoKnot
        '''

        index = np.argwhere(self.array>0)
        # easy change to dicts
        elements = []
        for i in index:
            elements.append([self.array[i[0]][i[1]][i[2]], i[0], i[1], i[2]])
        
        elements = sorted(elements, key=lambda x: x[0])
        elements.append(elements[0])
        
        joggle_scale = 1e-2
        np.random.seed(42)
        elements = [np.array([i[1:4] for i in elements], dtype=float) +
        np.random.normal(scale=joggle_scale, size=(len(elements), 3))]

        kl = kymoknot.KymoKnotSearch(
        seed=0,
        closure_type=kymoknot.CL_QHULLHYB,
        close_subchain=kymoknot.CL_QHULLHYB,
        search_type=[SearchType.BU],
        )

        chain_res = kl.search(elements, kymoknot.INP_LINEAR)
        res = chain_res[SearchType.BU]

        x = knot
        for ke in res[0]:
            p = ke.knot_ids
            p = p[3:]
            p = ''.join(p.split())
            x = ''.join(x.split())
            if p == 'UN':
                p = '0_1'
            if p == x:
                l = True
            else:
                l = False
                
        print(x)
        return l
    
    def alexander_polynomial_hash(self, knot, joggle = True):
        '''
        KymoKnot: https://github.com/luca-tubiana/KymoKnot
        '''

        index = [pos for pos, val in self.array.items() if val > 0]
        # easy change to dicts
        elements = []
        for i in index:
            elements.append([self.array[i], i[0], i[1], i[2]])
        
        elements = sorted(elements, key=lambda x: x[0])
        elements.append(elements[0])
        
        if joggle == True:
            joggle_scale = 1e-2
        else:
            joggle_scale = 0

        elements = [np.array([i[1:4] for i in elements], dtype=float) +
        np.random.normal(scale=joggle_scale, size=(len(elements), 3))]

        kl = kymoknot.KymoKnotSearch(
        seed=0,
        closure_type=kymoknot.CL_QHULLHYB,
        close_subchain=kymoknot.CL_QHULLHYB,
        search_type=[SearchType.BU],
        )

        chain_res = kl.search(elements, kymoknot.INP_LINEAR)
        res = chain_res[SearchType.BU]

        x = knot
        for ke in res[0]:
            p = ke.knot_ids
            p = p[3:]
            p = ''.join(p.split())
            x = ''.join(x.split())
            if p == 'UN':
                p = '0_1'
            if p == x:
                l = True
            else:
                l = False
                
        print(x)
        print(p)
        return l