import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import sys
# daily build binary freecad path
FreeCADPath = '/usr/lib/freecad-daily/lib'
#append FreeCAD to python path
sys.path.append(FreeCADPath)
import FreeCAD
#set compound merge on STP imports to Off
FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Import/hSTEP").SetBool("ReadShapeCompoundMode", False)
import Part
import Mesh
import MeshPart
import Import


source = np.array([0.0,0.0,0.0])

#get mesh centers of box
f = '/home/tom/HEATruns/NSTXU/radTests/nstx/boxMesh.stl'
mesh = Mesh.Mesh(f)
N_facets = mesh.CountFacets
x = np.zeros((N_facets,3))
y = np.zeros((N_facets,3))
z = np.zeros((N_facets,3))
for i,facet in enumerate(mesh.Facets):
    #mesh points
    for j in range(3):
        x[i][j] = facet.Points[j][0]
        y[i][j] = facet.Points[j][1]
        z[i][j] = facet.Points[j][2]
centers = np.zeros((len(x), 3))
centers[:,0] = np.sum(x,axis=1)/3.0
centers[:,1] = np.sum(y,axis=1)/3.0
centers[:,2] = np.sum(z,axis=1)/3.0
Nj = len(centers)

#vector between source and targets
r_ij = np.zeros((Nj,3))
r_ij = centers - source
rMag = np.linalg.norm(r_ij, axis=1)
rNorm = r_ij / rMag.reshape((-1,1))

#calculate spherical coordinates for each target point
theta = np.arccos( (rNorm[:,2]) )
phi = np.arctan2( rNorm[:,1], rNorm[:,0]  )
points = np.vstack([phi+2*np.pi, -np.cos(theta)]).T

#calculate the convex hull
hull = ConvexHull(points)



#
fig = go.Figure()
fig.add_trace(go.Scatter(x=points[:,0], y=points[:,1], mode='markers', marker_size=20,))
fig.update_yaxes(title_text=r'$\theta$')
fig.update_xaxes(title_text=r'$\phi$')
fig.update_layout(
#    legend=dict(
#    yanchor="middle",
#    y=0.9,
#    xanchor="left",
#    x=0.1
#    ),
    font=dict(
#        family="Courier New, monospace",
        size=30,
    )
#
    )

for simplex in hull.simplices:
    fig.add_trace(go.Scatter(x=points[simplex, 0], y=points[simplex, 1], mode='lines', line=dict(width=4,color='red')))
fig.show()
input()


I = np.arange(len(x)*3)[0::3]
J = np.arange(len(x)*3)[1::3]
K = np.arange(len(x)*3)[2::3]
fig = go.Figure(data=[go.Mesh3d(x=x.flatten(order='C'), y=y.flatten(order='C'), z=z.flatten(order='C'), i=I, j=J, k=K, opacity=0.50)])
for i,row in enumerate(r_ij):
    p0 = source
    p1 = row + p0
    vecX = [p0[0], p1[0]]
    vecY = [p0[1], p1[1]]
    vecZ = [p0[2], p1[2]]
    fig.add_trace(go.Scatter3d(x=vecX, y=vecY, z=vecZ, mode='lines+markers'))
#fig.add_trace(go.Scatter3d(x=[source[0]], y=[source[1]], z=[source[2]], mode='markers'))

# Set up 100 points. First, do angles
theta = np.linspace(0,2*np.pi,100)
phi = np.linspace(0,np.pi,100)

# Set up coordinates for points on the sphere
x0 = 100.0*np.outer(np.cos(theta),np.sin(phi))
y0 = 100.0*np.outer(np.sin(theta),np.sin(phi))
z0 = 100.0*np.outer(np.ones(100),np.cos(phi))

fig.add_trace(go.Surface(x=x0, y=y0, z=z0, opacity=0.5))
fig.show()
