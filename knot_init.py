import numpy as np
from numba import prange
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as mcolors  
from mpl_toolkits.mplot3d import Axes3D

class Knot:
    def __init__(self, knot, array):

        self.knot = knot
        self.array = array

    def initialize(self):

        if self.knot == '0_1':
            return k0_1_initialization(self.array)
        
        elif self.knot == '3_1':
            return k3_1_initialization_L(self.array)
        
def plot_3d(array):

    norm = mcolors.Normalize(vmin=np.min(array[array > 0]), vmax=np.max(array))
    cmap = cm.coolwarm  

    # Initialize color array
    colors = np.zeros(array.shape + (4,)) 

    # Apply colormap for nonzero values
    mask = array > 0  
    colors[mask] = cmap(norm(array[mask]))  

    colors[..., 3] = np.where(array > 0, 1.0, 0.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the voxels
    ax.voxels(array > 0, facecolors=colors)

    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])

    plt.show()
    plt.clf()


def plot_3d_line(array):

    indices = np.argwhere(array > 0)
    values = array[tuple(indices.T)]
    sorted_indices = indices[np.argsort(values)]

    norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
    cmap = cm.coolwarm
    colors = cmap(norm(np.sort(values)))

    x, y, z = sorted_indices[:, 0], sorted_indices[:, 1], sorted_indices[:, 2]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot line
    ax.plot3D(x, y, z, color='gray', linewidth=1.5)

    # Scatter with colors based on values
    ax.scatter(x, y, z, c=np.sort(values), cmap=cmap, s=10)

    ax.set_xlim([0, array.shape[0]])
    ax.set_ylim([0, array.shape[1]])
    ax.set_zlim([0, array.shape[2]])

    plt.show()
    plt.clf()
        

def draw_line_xy(grid, z, x1, y1, x2, y2, size):
    '''
    Bresenham's line algorithm; draws lines between points in array.
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

def draw_line_xyz(grid, x1, y1, z1, x2, y2, z2, size):
    '''
    3D Bresenham's line algorithm; draws a line between two 3D points in a 3D grid.
    '''
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    sz = 1 if z1 < z2 else -1

    if dx >= dy and dx >= dz:        # x 
        err_y = dx // 2
        err_z = dx // 2
        while x1 != x2:
            if 0 <= x1 < size and 0 <= y1 < size and 0 <= z1 < size:
                grid[z1, y1, x1] = 1
            err_y -= dy
            err_z -= dz
            if err_y < 0:
                y1 += sy
                err_y += dx
            if err_z < 0:
                z1 += sz
                err_z += dx
            x1 += sx
    elif dy >= dx and dy >= dz:      # y 
        err_x = dy // 2
        err_z = dy // 2
        while y1 != y2:
            if 0 <= x1 < size and 0 <= y1 < size and 0 <= z1 < size:
                grid[z1, y1, x1] = 1
            err_x -= dx
            err_z -= dz
            if err_x < 0:
                x1 += sx
                err_x += dy
            if err_z < 0:
                z1 += sz
                err_z += dy
            y1 += sy
    else:                            # z 
        err_x = dz // 2
        err_y = dz // 2
        while z1 != z2:
            if 0 <= x1 < size and 0 <= y1 < size and 0 <= z1 < size:
                grid[z1, y1, x1] = 1
            err_x -= dx
            err_y -= dy
            if err_x < 0:
                x1 += sx
                err_x += dz
            if err_y < 0:
                y1 += sy
                err_y += dz
            z1 += sz

    # Set the final point
    if 0 <= x2 < size and 0 <= y2 < size and 0 <= z2 < size:
        grid[z2, y2, x2] = 1

def k0_1_initialization(array):
    '''
    Initializes an unknot on the boundary of the state space i.e.
    '''

    length = len(array[0])
    for i in prange(50, length - 24):
        for j in prange(50, length - 24):
            if i == 50 or i == length-25:
                array[50][i][j] = 1

            if j == 50 or j == length-25:

                array[50][i][j] = 1

    array[50][50][50] = array[50][50][75] = array[50][75][75] = array[50][75][50] = 0

    print(len(np.argwhere(array == 1)))

    return array

def k0_1_initialization_corners(array):
    '''
    Initializes an unknot on the boundary of the state space i.e.
    '''

    length = len(array[0])
    for i in prange(50, length - 24):
        for j in prange(50, length - 24):
            if i == 50 or i == length-25:
                array[50][i][j] = 1

            if j == 50 or j == length-25:

                array[50][i][j] = 1

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

def k0_1_initialization_3(array):
    '''
    Writhe = 1, 0_1
    '''

    draw_line_xy(array, 6, 50, 47, 50, 32, 100)
    draw_line_xy(array, 5, 49, 31, 21, 48, 100)
    draw_line_xy(array, 6, 20, 47, 20, 32, 100)
    draw_line_xy(array, 7, 21, 31, 49, 48, 100)

    return array

def k0_1_initialization_4(array):  
    '''
    Writhe = 1, 0_1, wide
    '''

    draw_line_xyz(array, 50, 47, 6, 50, 32, 6, 100)
    draw_line_xyz(array, 50, 32, 6, 50, 32, 20, 100)
    draw_line_xyz(array, 49, 31, 21, 21, 48, 21, 100)
    draw_line_xyz(array, 20, 47, 6, 20, 32, 6, 100)
    draw_line_xyz(array, 21, 31, 7, 49, 48, 7, 100)
    draw_line_xyz(array, 21, 47, 6, 21, 47, 20, 100)


    plot_3d(array)
    

    return array

def k0_1_initialization_L(array):
    '''
    Initializes an unknot on the boundary of the state space i.e.
    '''

    length = len(array[0])
    for i in prange(25, length - 24):
        for j in prange(25, length - 24):
            if i == 25 or i == length-25:
                array[50][i][j] = 1

            if j == 25 or j == length-25:

                array[50][i][j] = 1

    array[50][25][25] = array[50][25][75] = array[50][75][75] = array[50][75][25] = 0

    print(len(np.argwhere(array == 1)))
    plot_3d(array)

    return array

def k3_1_initialization(array):
    '''
    Initializes a 3_1 in state space
    '''

    draw_line_xy(array, 6, 51, 49, 51, 38, 100) # 6 
    draw_line_xy(array, 6, 50, 37, 44, 37, 100) # 6

    array[5][37][43] = array[4][37][43] = array[3][37][43] = 1

    draw_line_xy(array, 2, 43, 38, 43, 45, 100) # 2

    array[5][46][43] = array[4][46][43] = array[3][46][43] = 1

    draw_line_xy(array, 6, 43, 47, 43, 52, 100) # 6 
    draw_line_xy(array, 6, 44, 53, 52, 53, 100) # 6
    draw_line_xy(array, 6, 53, 52, 53, 41, 100) # 6

    array[5][40][53] = 1 # 5

    draw_line_xy(array, 4, 52, 40, 41, 40, 100) # 4 
    draw_line_xy(array, 4, 40, 41, 40, 49, 100) # 4
    draw_line_xy(array, 4, 41, 50, 49, 50, 100) # 4

    array[5][50][50] = 1    

    print(len(np.argwhere(array == 1)))

    return array

def k3_1_initialization_L(array):
    '''
    Initializes a 3_1 in state space
    '''

    draw_line_xy(array, 50, 50, 55, 50, 33, 100) # 50
    draw_line_xy(array, 50, 49, 32, 38, 32, 100) # 50

    array[49][32][37] = array[48][32][37] = array[47][32][37] = 1

    draw_line_xy(array, 46, 37, 33, 37, 45, 100) # 46

    array[49][46][37] = array[48][46][37] = array[47][46][37] = 1

    draw_line_xy(array, 50, 37, 47, 37, 57, 100) # 50
    draw_line_xy(array, 50, 38, 58, 52, 58, 100) # 50
    draw_line_xy(array, 50, 53, 57, 53, 41, 100) # 50

    array[49][40][53] = 1 # 49

    draw_line_xy(array, 48, 52, 40, 32, 40, 100) # 48
    draw_line_xy(array, 48, 31, 41, 31, 55, 100) # 48
    draw_line_xy(array, 48, 32, 56, 49, 56, 100) # 48

    array[49][56][50] = 1 # 49

    print(len(np.argwhere(array == 1)))

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

