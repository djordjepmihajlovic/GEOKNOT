import numpy as np
import matplotlib.pyplot as plt
from knot_init import *
from knot_evolution_hash import lattice_writhe_Klenin
from knot_invs import *
from scipy.linalg import expm
from scipy.linalg import logm

'''
This code contains functions to compute various features of knots; some of which are hard coded into the evolution step to ensure flattened distributions.
https://arxiv.org/pdf/1102.5658
'''

# My Example knot
knot = np.loadtxt('/Users/s1910360/Desktop/0_1_2506/0_1_0_32_678.csv', delimiter=',', dtype=np.float64)

def read_coord(knot):
    coord_list = [(float(i[0]), float( i[1]), float(i[2]), float(i[3])) for i in knot]
    coord_list = sorted(coord_list, key=lambda x: x[0])

    return coord_list

# Writhe matrix; Klenin et.al formulation
# im = lattice_writhe_Klenin(read_coord(knot))
# print(np.sum(im))
# plt.imshow(im)
# plt.colorbar()

def map(knot):
    '''
    Computes writhe using Klenin formulation.
    Input: list of points in 3D space and value.
    '''
    
    ringx = np.array([(x, y, z) for _, x, y, z in knot])
    vals = np.array([val for val, _, _, _ in knot])

    sorted_indices = np.argsort(vals)
    ring1 = ringx[sorted_indices]
    ring2 = ring1.copy()
    matrix = np.zeros((ring1.shape[0], ring2.shape[0]))
    # Loop on the first ring
    for i in prange(ring1.shape[0]):
        # Loop on the second ring
        for j in prange(ring2.shape[0]):
            matrix[i,j] = dist(ring1, ring2, i, j)
        print(i)

    return matrix

def angle(ring1, ring2, i, j):

    prev_one = ring1[i-1]
    one = ring1[i]
    two = ring2[j]

    vec_1 = one-prev_one
    vec_2 = two-one

    dot_prod = np.dot(vec_1, vec_2)
    norm_vec1 = np.linalg.norm(vec_1)
    norm_vec2 = np.linalg.norm(vec_2)
    angle_rad = np.arccos(dot_prod/(norm_vec1 * norm_vec2))
    angle_rad= np.degrees(angle_rad)
    angle_rad = np.nan_to_num(angle_rad, nan=0.0)

    return angle_rad

def dist(ring1, ring2, i, j):
    one = ring1[i]
    two = ring2[j]
    dist = two - one
    length = np.linalg.norm(dist)

    return length

def frenet_frames(knot):
    '''
    Computes frenet frame
    '''

    ringx = np.array([(x, y, z) for _, x, y, z in knot])
    vals = np.array([val for val, _, _, _ in knot])

    sorted_indices = np.argsort(vals)
    ring1 = ringx[sorted_indices]

    # Loop on the first ring
    frenet_frame = []
    for i in prange(ring1.shape[0]):  

        tangent_1 = (ring1[i] - ring1[i-1])/np.linalg.norm(ring1[i] - ring1[i-1])
        tangent_2 = (ring1[(i+1)%ring1.shape[0]] - ring1[i])/np.linalg.norm(ring1[(i+1)%ring1.shape[0]] - ring1[i])

        binormal = np.cross(tangent_1, tangent_2)/np.linalg.norm(np.cross(tangent_1, tangent_2))
        normal = np.cross(tangent_1, binormal)

        frenet_frame.append(np.array([tangent_1, binormal, normal]))

    return frenet_frame


def complex_hasimoto(frenet_frame):

    pauli_x = np.array([[0, 1],[1, 0]])
    pauli_y = np.array([[0, -1j],[1j, 0]])

    sigma_p = pauli_x + 1j*pauli_y
    sigma_n = pauli_x - 1j*pauli_y
    alpha = 0

    sl2c = []
    holonomy = np.eye(2)

    for idx, frame in enumerate(frenet_frame):
        sign = np.sign(np.dot(np.cross(frenet_frame[idx-1][0], frame[0]), frenet_frame[idx-1][2]))
        phi = sign * np.arccos(np.dot(frenet_frame[idx-1][0], frame[0]))
        theta = np.arccos(np.dot(frenet_frame[idx-1][1], frame[1]))
        alpha += theta

        psi = phi * np.exp(1j * alpha)

        A = 1j/2 * (psi * sigma_p + psi.conj() * sigma_n)
        U = expm(A)
        sl2c.append(U)
        holonomy = holonomy @ U

    print(holonomy)

    connections = []
    for i in range(len(sl2c) - 1):
        g1 = sl2c[i]
        g2 = sl2c[i+1]
        A = logm(np.linalg.inv(g1) @ g2) / 1.0
        connections.append(A)

    return sl2c


def plot_eigenvalues(sl2c):
    '''
    Plots the eigenvalues of SL(2,C) matrices in 3D space.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for matrix in sl2c:
        eigenvalues = np.linalg.eigvals(matrix)
        ax.scatter(np.real(eigenvalues[0]), np.imag(eigenvalues[0]), np.real(eigenvalues[1]), color='b')
        ax.scatter(np.real(eigenvalues[1]), np.imag(eigenvalues[1]), np.real(eigenvalues[0]), color='r')

    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_zlabel('Eigenvalue Magnitude')
    plt.show()

def plot_frenet_frames_3d_with_knot(frenet_frame, knot, scale=0.01):
    '''
    Plots the Frenet frames in 3D space with scaled quivers starting at the knot points.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract knot coordinates
    ringx = np.array([(x, y, z) for _, x, y, z in knot])

    for i, frame in enumerate(frenet_frame):
        tangent, binormal, normal = frame
        start_point = ringx[i]  # Starting point for the quiver

        ax.quiver(
            start_point[0], start_point[1], start_point[2],
            tangent[0] * scale, tangent[1] * scale, tangent[2] * scale,
            color='r', label='Tangent' if i == 0 else ""
        )
        ax.quiver(
            start_point[0], start_point[1], start_point[2],
            binormal[0] * scale, binormal[1] * scale, binormal[2] * scale,
            color='g', label='Binormal' if i == 0 else ""
        )
        ax.quiver(
            start_point[0], start_point[1], start_point[2],
            normal[0] * scale, normal[1] * scale, normal[2] * scale,
            color='b', label='Normal' if i == 0 else ""
        )

    # Plot the knot embedding
    ax.plot(ringx[:, 0], ringx[:, 1], ringx[:, 2], color='k', label='Knot')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Frenet Frames on Knot Embedding (Scaled Quivers)')
    plt.legend()
    plt.show()

# Example usage:
knot_coords = read_coord(knot)
frenet_frame = frenet_frames(knot_coords)

# ang_map = map(read_coord(knot))
# print(np.sum(ang_map))
# plt.imshow(ang_map)
# plt.colorbar()

# # 3D plot of embedding
# plot_3d_line(read_coord(knot))

sl2c = complex_hasimoto(frenet_frames(read_coord(knot)))
plot_eigenvalues(sl2c)

