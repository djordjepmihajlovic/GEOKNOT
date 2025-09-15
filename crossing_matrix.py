import numpy as np
import torch
from numba import njit
import random

### This is old code to find a 'crossing matrix' for polygonal knots.

@njit
def compute_crossing_matrix(knot, projection, dim):
    cross_matrix = np.zeros((dim, dim), dtype=np.float32)

    inter_distance = []
    crossings = []

    # Compute the squared norm of the projection vector
    projection_norm = 0.0
    for i in range(3):
        projection_norm += projection[i] * projection[i]
    projection_norm = np.sqrt(projection_norm)

    # Normalize the projection vector
    norm = projection / projection_norm

    # Project the knot onto the plane orthogonal to the projection
    for i in range(100):
        # Compute the dot product of the knot point and the norm (projection)
        dot_product = 0.0
        for j in range(3):
            dot_product += knot[i, j] * norm[j]

        # Subtract the projection of the knot point along the norm
        for j in range(3):
            knot[i, j] -= dot_product * norm[j]

    # Detect crossings
    for i in range(0, 100):
        crossing_per_segment = 0
        vec_x1 = knot[i, 0]
        vec_x2 = knot[(i + 1) % 100, 0]
        vec_y1 = knot[i, 1]
        vec_y2 = knot[(i + 1) % 100, 1]

        for j in range(0, 100):  # Looping forward
            if j != i and j != (i + 1) % 100 and j != (i - 1) % 100:
                vec_x3 = knot[j % 100, 0]
                vec_x4 = knot[(j + 1) % 100, 0]
                vec_y3 = knot[j % 100, 1]
                vec_y4 = knot[(j + 1) % 100, 1]

                # Linear algebra to detect intersections
                denominator = ((vec_x1 - vec_x2) * (vec_y3 - vec_y4) - (vec_y1 - vec_y2) * (vec_x3 - vec_x4))
                if denominator == 0:
                    continue
                t = ((vec_x1 - vec_x3) * (vec_y3 - vec_y4) - (vec_y1 - vec_y3) * (vec_x3 - vec_x4)) / denominator
                s = ((vec_x1 - vec_x3) * (vec_y1 - vec_y2) - (vec_y1 - vec_y3) * (vec_x1 - vec_x2)) / denominator

                if 0 <= s <= 1 and 0 <= t <= 1:
                    crossings.append([i, j])
                    crossing_per_segment += 1

                    inter_x = vec_x1 + t * (vec_x2 - vec_x1)
                    inter_y = vec_y1 + t * (vec_y2 - vec_y1)

                    x = abs(inter_x - vec_x1)
                    y = abs(inter_y - vec_y1)
                    inter_distance.append([(x**2 + y**2)**(1/2), i])

    # Sorting crossings based on inter-distance
    for indx, i in enumerate(crossings):
        for indy, j in enumerate(crossings):
            if i[0] == j[0] and indy > indx:
                if inter_distance[indx][0] > inter_distance[indy][0]:
                    crossings[indx], crossings[indy] = crossings[indy], crossings[indx]

    cross_values = np.zeros(len(crossings), dtype=np.float32)

    for idx, i in enumerate(crossings):
        cross_values[idx] = knot[i[0], 2] - knot[i[1], 2]

    crossy = 1
    for idx, i in enumerate(cross_values):
        for idy, j in enumerate(cross_values):
            if idx < idy:
                if abs(i) == abs(j):
                    if i > j:
                        cross_values[idx] = crossy
                        cross_values[idy] = -crossy
                    else:
                        cross_values[idx] = -crossy
                        cross_values[idy] = crossy
                    crossy += 1

    for idx, i in enumerate(crossings):
        cross_matrix[i[0], i[1]] = cross_values[idx]

    return cross_matrix, len(crossings)  # Return both the matrix and the number of crossings

# Function to check different projections and return the matrix with the most crossings
@njit
def find_best_projection(knot):
    # List of different projection vectors to check (could add more)
    projections = [
        np.array([1, 1, 1], dtype=np.float32),
        np.array([-1, 1, 1], dtype=np.float32),
        np.array([1, -1, 1], dtype=np.float32),
        np.array([-1, -1, 1], dtype=np.float32)
    ]
    
    best_crossings = -1
    best_matrix = None

    for projection in projections:
        matrix, num_crossings = compute_crossing_matrix(knot, projection)
        
        # Check if this projection has more crossings
        if num_crossings > best_crossings:
            best_crossings = num_crossings
            best_matrix = matrix

    return best_matrix

# Wrapper to convert numpy array to tensor (used in PyTorch)
def tensor_crossing_matrix(knot):

    return torch.tensor(find_best_projection(knot.numpy()), dtype=torch.float32)