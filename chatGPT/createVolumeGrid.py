import numpy as np

# Coordinate arrays (replace with your actual data)
x_coords = np.array([1, 4, 7])
y_coords = np.array([2, 5, 8])
z_coords = np.array([3, 6, 9])

# Create a 3D meshgrid
X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

# Stack the coordinates into a single array
centroids = np.stack((X, Y, Z), axis=-1)

def calculate_volumes_vectorized(centroids):
    dx = np.diff(centroids[:, :, :, 0], axis=0)
    dy = np.diff(centroids[:, :, :, 1], axis=1)
    dz = np.diff(centroids[:, :, :, 2], axis=2)

    volumes = np.zeros_like(dx[:-1, :, :-1])  # Initialize the volumes array with the same shape as the centroids array

    for i in range(dx.shape[0] - 1):
        for j in range(dy.shape[1] - 1):
            for k in range(dz.shape[2] - 1):
                volumes[i, j, k] = dx[i, j, k] * dy[i, j, k] * dz[i, j, k]

    return volumes

volumes = calculate_volumes_vectorized(centroids)
print(volumes)
