import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from sympy import symbols, simplify
from tensor_algebra import *
import kymoknot 
from kymoknot.searchtype import SearchType
from kymoknot.knotentry import KnotEntry
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
        self.axis = np.array([np.pi, -np.e/2, np.sqrt(2)/2])
        self.projection = points_on_axis(self.array, self.axis)
        scan(self.projection)


    def build_equation(self):
        '''
        Build equation from splittings and crossings.
        '''

        tensors = scan(self.projection)
        crossings = crossing(self.projection, self.axis)

        equation = tensors
        for i in crossings:
            equation.append([i])

        equation = sorted(equation, key = lambda x: x[0][1]) # sort by y values 

        # ## debugging plot

        q = symbols('q')
        e_1 = symbols('e_1') ## e1: (e_{1})
        e_2 = symbols('e_2')
        de_1 = symbols('de_1') ## dual e1: (e^{1})
        de_2 = symbols('de_2')
        V = symbols('V')
        dV = symbols('dV')

        if self.q_group == 'Uq(sl2)':

            R_table_VV = {
                TensorProduct(e_1, e_1): q**(1/4)*TensorProduct(e_1, e_1),
                TensorProduct(e_1, e_2): q**(-1/4)*TensorProduct(e_2, e_1),
                TensorProduct(e_2, e_1): q**(-1/4)*TensorProduct(e_1, e_2) + (q**(1/4) - q**(-3/4))*TensorProduct(e_2, e_1),
                TensorProduct(e_2, e_2): q**(1/4)*TensorProduct(e_2, e_2),
            }

            inv_R_table_VV = {
                TensorProduct(e_1, e_1): q**(-1/4)*TensorProduct(e_1, e_1),
                TensorProduct(e_1, e_2): q**(1/4)*TensorProduct(e_2, e_1) + (q**(-1/4) - q**(3/4))*TensorProduct(e_1, e_2),
                TensorProduct(e_2, e_1): q**(1/4)*TensorProduct(e_1, e_2),
                TensorProduct(e_2, e_2): q**(-1/4)*TensorProduct(e_2, e_2),
            }

            R_table_dVdV = {
                TensorProduct(de_1, de_1): q**(-1/4)*TensorProduct(de_1, de_1),
                TensorProduct(de_1, de_2): q**(1/4)*TensorProduct(de_2, de_1),
                TensorProduct(de_2, de_1): q**(1/4)*TensorProduct(de_1, de_2) + (q**(-1/4) - q**(3/4))*TensorProduct(de_2, de_1),
                TensorProduct(de_2, de_2): q**(-1/4)*TensorProduct(de_2, de_2),
            }

            inv_R_table_dVdV = {
                TensorProduct(de_1, de_1): q**(1/4)*TensorProduct(de_1, de_1),
                TensorProduct(de_1, de_2): q**(-1/4)*TensorProduct(de_2, de_1) + (q**(1/4) - q**(-3/4))*TensorProduct(de_1, de_2),
                TensorProduct(de_2, de_1): q**(-1/4)*TensorProduct(de_1, de_2),
                TensorProduct(de_2, de_2): q**(1/4)*TensorProduct(de_2, de_2),
            }


        evaluation_table = {
            TensorProduct(dV, V): q**(-1/2)*TensorProduct(de_1, e_1) + q**(1/2)*TensorProduct(de_2, e_2),
            TensorProduct(V, dV): TensorProduct(e_1, de_1) + TensorProduct(e_2, de_2),
        }

        coevaluation_table = {
            TensorProduct(de_1, e_1): 1,
            TensorProduct(de_1, e_2): 0,
            TensorProduct(de_2, e_1): 0,
            TensorProduct(de_2, e_2): 1,
            
            TensorProduct(e_1, de_1): q**(1/2),
            TensorProduct(e_1, de_2): 0,
            TensorProduct(e_2, de_1): 0,
            TensorProduct(e_2, de_2): q**(-1/2),
        }

        basis_table = {
            1.0: V,
            -1.0: dV,
        }

        def evaluate(tensor_product):
            return evaluation_table.get(tensor_product, tensor_product)
        
        def coevaluate(tensor_product):
            return coevaluation_table.get(tensor_product, tensor_product)
        
        def RMatrix(tensor_product):
            return R_table_VV.get(tensor_product, tensor_product)
        
        def detect_change(equation):
            '''
            Observes what changes are happening between different partitions of knot diagram.
            '''
            for idx in range(0, len(equation)-1):
                '''
                Cup left of original.
                '''
                print(equation[idx])
                print(equation[idx+1])

            return None
        
        detect_change(equation)
        
        '''
        Logic to read in equation.
        Build equation and alter as we move through the changes.
        'Detect change' function (between steps).
            1. Get initial state
                [V, dV] -> Create tensor
            2. Check operation for next state:
                Cup and location
                    [V, dV] -> [V, dV, V, dV]
                    = insert tensor factors accordingly (location = index), (function = cap or cup) using the insert_tensor function
                Crossing and location
                    [V, dV, V, dV] -> [V, dV, dV, V]
                    = crossing, apply correct R matrix at location
        '''

        for idx, i in enumerate(equation):
            elements = []
            vals = []
            product = []
            if len(i)> 1:
                for j in i:
                    elements.append(basis_table.get(j[2]))
                print(elements)
                prev_elements = elements

                for p in range(0, len(elements)-1, 2):
                    vals.append(evaluate(TensorProduct(elements[p], elements[p+1])))

                
                if len(vals)>1:
                    tensor_equation = 0
                    for dxd in range(0, len(vals)-1):
                        
                        if tensor_equation == 0:
                            tensor_equation = TensorProduct(vals[dxd], vals[dxd+1])
                        
                        else:
                            tensor_equation = TensorProduct(tensor_equation, vals[dxd+1])
                else:
                    tensor_equation = vals[0]

                print(tensor_equation)
            
            else:
                # x coord
                ### Need to work on this logic...
                order = equation[idx-1]
                order.append(i[0])
                order = sorted(order, key = lambda x: x[0])
                crossing_index  = order.index(i[0])
                ### Crossing is between elements at crossing_index and crossing_index-1
                ### Crossing type (apply R matrix correctly)
                print(crossing_index)
                print(prev_elements[crossing_index-1], prev_elements[crossing_index])
        

        plt.plot([i[0] for i in self.projection],[i[1] for i in self.projection], linestyle = '-', c='blue')
        plt.plot([self.projection[0][0], self.projection[-1][0]], [self.projection[0][1], self.projection[-1][1]], c='blue')

        for pt in self.projection:
            x, y = pt[0], pt[1]
            value = pt[5]
            plt.text(x, y, str(value), fontsize=9, ha='left', va='bottom')

        for i in tensors:
            plt.hlines(i[0][1], xmin=min(self.projection[:, 0]), xmax=max(self.projection[:, 0]), color='red', linestyle='--')

        plt.title('Projection of Knot: Quantum Invariant')
        
        plt.show()

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
    
discretization = 100
knot_type = '3_1'
state_space = np.zeros((discretization, discretization, discretization))
knot = Knot(knot_type, state_space)
knot = knot.initialize()
knot = orient(knot)

Q = Q_invariant(knot, 'Uq(sl3)')
Q.build_equation()