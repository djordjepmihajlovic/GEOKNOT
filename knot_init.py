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
            return k0_1_initialization_100(self.array)
        
        elif self.knot == '3_1':
            return k3_1_initialization_100(self.array)
        
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


def plot_3d_line_array(array):

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

def plot_3d_line(coord_list):
    """
    Plots a 3D line and scatter plot based on a list of tuples (val, x, y, z).
    """
    # Extract values and coordinates
    #values = [item[0] for item in coord_list]
    values = np.arange(0, len(coord_list))
    x = [item[0] for item in coord_list]
    y = [item[1] for item in coord_list]
    z = [item[2] for item in coord_list]

    # Normalize values for coloring
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.coolwarm
    colors = cmap(norm(values))

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot line
    ax.plot3D(x, y, z, color='gray', linewidth=1.5)

    # Scatter with colors based on values
    ax.scatter(x, y, z, c=values, cmap=cmap, s=10)
    plt.show()
        

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

def k0_1_initialization_100(array):
    '''
    Initializes an unknot on the boundary of the state space (size 100).
    '''

    length = len(array[0])
    for i in prange(36, length - 37):
        for j in prange(36, length - 37):
            if i == 36 or i == length-38:
                array[50][i][j] = 1

            if j == 36 or j == length-38:

                array[50][i][j] = 1

    array[50][36][36] = array[50][36][62] = array[50][62][62] = array[50][62][36] = 0

    print(len(np.argwhere(array == 1)))

    return array


def k3_1_initialization_100(array):
    '''
    Initializes a 3_1 in state space (size 100).
    '''

    draw_line_xy(array, 56, 51, 49, 51, 38, 100) # length = 9
    draw_line_xy(array, 56, 50, 37, 44, 37, 100) # length = 6

    array[55][37][43] = array[54][37][43] = array[53][37][43] = 1

    draw_line_xy(array, 52, 43, 38, 43, 45, 100) # length = 7

    array[55][46][43] = array[54][46][43] = array[53][46][43] = 1

    draw_line_xy(array, 56, 43, 47, 43, 56, 100) # length = 5
    draw_line_xy(array, 56, 44, 57, 52, 57, 100) # length = 8
    draw_line_xy(array, 56, 53, 56, 53, 41, 100) # length = 11

    array[55][40][53] = 1 # 5

    draw_line_xy(array, 54, 52, 40, 41, 40, 100) # length = 11
    draw_line_xy(array, 54, 40, 41, 40, 49, 100) # length = 8
    draw_line_xy(array, 54, 41, 50, 49, 50, 100) # length = 8

    # total length = 81

    array[55][50][50] = 1    

    print(len(np.argwhere(array == 1)))
    
    return array