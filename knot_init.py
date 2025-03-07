import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

class Knot:
    def __init__(self, knot, array):

        self.knot = knot
        self.array = array

    def initialize(self):

        if self.knot == '0_1':
            return k0_1_initialization(self.array)
        
        elif self.knot == '3_1':
            return k3_1_initialization(self.array)
        

def draw_line(grid, z, x1, y1, x2, y2, size):
    '''
    Bresenham's line algorithm; draws lines between points in array
    '''
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= x1 < size and 0 <= y1 < size:
            grid[z, y1, x1] = 1
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def k0_1_initialization(array):
    '''
    Initializes an unknot on the boundary of the state space i.e.
    '''

    length = len(array[0])
    for i in prange(25, length - 24):
        for j in prange(25, length - 24):
            if i == 25 or i == length-25:
                array[4][i][j] = 1

            if j == 25 or j == length-25:

                array[4][i][j] = 1

    array[4][25][25] = array[4][25][75] = array[4][75][75] = array[4][75][25] = 0

    print(len(np.argwhere(array == 1)))

    return array


def k0_1_initialization_2(array):
    '''
    Initializes an thin unknot on the boundary of the state space i.e.
    '''

    length = len(array[0])
    for i in prange(25, length - 24):
        for j in prange(70, length - 24):
            if i == 25 or i == length-25:
                array[4][i][j] = 1

            if j == 70 or j == length-25:

                array[4][i][j] = 1

    array[4][25][75] = array[4][75][70] = array[4][75][75] = array[4][25][70] = 0

    print(len(np.argwhere(array == 1)))

    return array


def k3_1_initialization(array):
    '''
    Initializes a 3_1 in state space
    '''

    draw_line(array, 6, 51, 49, 51, 38, 100) # 6 
    draw_line(array, 6, 50, 37, 44, 37, 100) # 6

    array[5][37][43] = array[4][37][43] = array[3][37][43] = 1

    draw_line(array, 2, 43, 38, 43, 45, 100) # 2

    array[5][46][43] = array[4][46][43] = array[3][46][43] = 1

    draw_line(array, 6, 43, 47, 43, 52, 100) # 6 dw
    draw_line(array, 6, 44, 53, 52, 53, 100) # 6
    draw_line(array, 6, 53, 52, 53, 41, 100) # 6

    array[5][40][53] = 1 # 5

    draw_line(array, 4, 52, 40, 41, 40, 100) # 4 dw
    draw_line(array, 4, 40, 41, 40, 49, 100) # 4
    draw_line(array, 4, 41, 50, 49, 50, 100) # 4

    array[5][50][50] = 1    

    return array


def k4_1_initialization(array):
    '''
    Initializes a 4_1 in state space
    '''

    return array

def k5_1_initialization(array):
    '''
    Initializes a 5_1 in state space
    '''

    return array

def k5_2_initialization(array):
    '''
    Initializes a 5_2 in state space
    '''

    return array