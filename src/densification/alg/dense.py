"""
This Python module contains a Python translation of dense.cpp in the SAPCU
algorithm.

It is translated for debugging purposes as the algorithm does not current operate
correctly / consistently on large datasets and a lot of the hyperparameters are
hardcoded in the C++ code, and additional hard-coded values limit the algorithm
to failing on datasets with more than 5000 points.
"""

import numpy as np
from scipy.spatial import KDTree
import heapq
import sys

# Constants
MAX_NUM_POINTS = 8 * 131072
MAX_KD_TREE_SIZE = 4 * MAX_NUM_POINTS
k = 10

# Helper functions for vector operations
def dot(a, b):
    return np.dot(a, b)

def dist(a, b):
    return np.linalg.norm(a - b)

def cross(a, b):
    return np.cross(a, b)

# Point-Triangle Distance
def point_tri(a, b, c, p):
    ab = b - a
    ac = c - a
    bc = c - b

    snom = dot(p - a, ab)
    sdenom = dot(p - b, a - b)
    tnom = dot(p - a, ac)
    tdenom = dot(p - c, a - c)

    if snom <= 0.0 and tnom <= 0.0:
        return a

    unom = dot(p - b, bc)
    udenom = dot(p - c, b - c)
    if sdenom <= 0.0 and unom <= 0.0:
        return b
    if tdenom <= 0.0 and udenom <= 0.0:
        return c

    n = cross(b - a, c - a)
    vc = dot(n, cross(a - p, b - p))

    if vc <= 0.0 and snom >= 0.0 and sdenom >= 0.0:
        return a + ab * snom / (snom + sdenom)

    va = dot(n, cross(b - p, c - p))
    if va <= 0.0 and unom >= 0.0 and udenom >= 0.0:
        return b + bc * unom / (unom + udenom)

    vb = dot(n, cross(c - p, a - p))
    if vb <= 0.0 and tnom >= 0.0 and tdenom >= 0.0:
        return a + ac * tnom / (tnom + tdenom)

    u = va / (va + vb + vc)
    v = vb / (va + vb + vc)
    w = 1.0 - u - v
    return a * u + b * v + c * w

# Main function
def main():
    # Read arguments
    cell = float(sys.argv[1])
    pnumber = int(sys.argv[2])
    dl = float(sys.argv[3])
    du = float(sys.argv[4])
    boxsize = int(round(1 / cell))

    # Read points
    points = np.loadtxt("test.xyz")
    kd_tree = KDTree(points)

    # Initialize variables
    visited = set()
    to_visit = set()
    output_points = []

    # Initial population of the to_visit set
    for point in points:
        grid_index = tuple(np.floor((point + 0.5) / cell).astype(int))
        to_visit.add(grid_index)

    # Process grid cells
    while to_visit:
        grid_index = to_visit.pop()
        if grid_index in visited:
            continue

        visited.add(grid_index)
        center = np.array(grid_index) * cell + 0.5 * cell - 0.5

        # Nearest neighbor search
        distances, indexes = kd_tree.query(center, k=k)
        nearest_points = points[indexes]

        # Compute distance to nearest triangle
        closest_dist = float('inf')
        for i in range(len(nearest_points) - 2):
            closest_point = point_tri(nearest_points[i], nearest_points[-2], nearest_points[-1], center)
            temp_dist = dist(closest_point, center)
            if temp_dist < closest_dist:
                closest_dist = temp_dist

        # Check distance constraints
        if dl <= closest_dist <= du:
            output_points.append(center)

        # Add neighboring cells to to_visit
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            neighbor_index = (grid_index[0] + dx, grid_index[1] + dy, grid_index[2] + dz)
            if neighbor_index not in visited:
                to_visit.add(neighbor_index)

    # Write output points to file
    np.savetxt("target.xyz", np.array(output_points), fmt='%f')

# Run the program
if __name__ == "__main__":
    main()
