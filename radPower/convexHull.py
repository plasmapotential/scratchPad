import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import plotly.graph_objects as go

rng = np.random.default_rng()
points = rng.random((30, 2))
hull = ConvexHull(points)
print("perimeter: ")
print(hull.area) #perimeter for 2D points is area
print("area: ")
print(hull.volume) #area for 2D points is volume


fig = go.Figure()
fig.add_trace(go.Scatter(x=points[:,0], y=points[:,1], mode='markers',))
fig.add_trace(go.Scatter(x=points[hull.vertices,0], y=points[hull.vertices,1], mode='lines'))
fig.show()
