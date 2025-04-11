import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

'''
This module calculates quantum invariants of a knot to determine knot type (not preserved in pivot algorithm).
Takes in state space array and returns a Q_invariant object.

Method:
    * Take in array of knot.
    * Project onto flat plane (non rational normal to avoid problems).
    * Forming the sequence.
    * Build equation.
    * Solves final equation (sequence of tensors multiplied by R^{x}).
        
'''

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

    return np.array(projected_points, dtype=np.float64)

@njit
def crossings(projection, axis):

        intersections = [] # coordinate information
        orientation = [] # over/under information

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

                            if [np.round(ip_x, decimals=2), np.round(ip_y, decimals=2)] not in intersections:
                                intersections.append([np.round(ip_x, decimals=2), np.round(ip_y, decimals=2)])

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
                                orientation.append(sign)
        

class Q_invariant:
    def __init__(self, array, q_group):
        '''
        Knot array of state space, specified quantum group and projection.
        '''
        self.array = array
        self.q_group = q_group
        self.axis = np.array([np.pi, np.e/2, np.sqrt(2)/2], dtype=np.int64)
        self.projection = points_on_axis(self.array, self.axis)
        self.crossings = crossings(self.projection, self.axis)


