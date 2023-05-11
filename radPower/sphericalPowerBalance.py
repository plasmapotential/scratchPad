import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import plotly.graph_objects as go



#arbitrary points in 3D that represent target mesh
Nj = 100000
x = np.random.uniform(-1500, 1500, Nj)
y = np.random.uniform(-1500, 1500, Nj)
z = np.random.uniform(-1500, 1500, Nj)
xyz = np.vstack([x,y,z]).T

#source point
src = np.array([0.0, 0.0, 0.0])

#vector between source and targets
r_ij = np.zeros((Nj,3))
r_ij = xyz - src
rMag = np.linalg.norm(r_ij, axis=1)
rNorm = r_ij / rMag.reshape((-1,1))

#calculate spherical coordinates for each target point
theta = np.arccos( (rNorm[:,2]) )
phi = np.arctan2( rNorm[:,1], rNorm[:,0]  )
points1 = np.vstack([phi, -np.cos(theta)]).T
points2 = np.vstack([phi, theta]).T
#calculate the convex hull
hull = ConvexHull(points1)
hull2 = ConvexHull(points2)
print("perimeter: ")
print(hull.area) #perimeter for 2D points is area
print("area: ")
print(hull.volume) #area for 2D points is volume
print("normalized area: ")
print(hull.volume / (4*np.pi))
print(hull2.volume / (2*np.pi**2))
