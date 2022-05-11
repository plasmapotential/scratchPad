#coverPoly.py
#Description:   cover a polygon with tesselation of shapes
#Engineer:      T Looby
#Date:          20220426

import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
import plotly.graph_objects as go

#this is an example shape used for testing
polyX = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
polyY = np.array([0.0, 1.0, 2.0, 0.5, 0.0])

#we will tesselate with rectangles / squares here
def square(x, y, s):
    return Polygon([(x, y), (x+s, y), (x+s, y+s), (x, y+s)])

poly = MultiPoint(np.vstack([polyX,polyY]).T).convex_hull
polyCoords = np.array(poly.exterior.coords)

grid_size = 0.1
ibounds = np.array(poly.bounds)//grid_size
ibounds[2:4] += 1
xmin, ymin, xmax, ymax = ibounds*grid_size
xrg = np.arange(xmin, xmax, grid_size)
yrg = np.arange(ymin, ymax, grid_size)
mp = MultiPolygon([square(x, y, grid_size) for x in xrg for y in yrg])
solution = MultiPolygon(list(filter(poly.intersects, mp)))


fig = go.Figure(go.Scatter(x=polyCoords[:,0], y=polyCoords[:,1], fill="toself"))

for geom in solution.geoms:
    xs, ys = np.array(geom.exterior.xy)
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", line=dict(color="seagreen")))


fig.show()
