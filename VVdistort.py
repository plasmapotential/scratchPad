import numpy as np
import plotly.graph_objects as go


n = 4 #toroidal mode number
h = 1.6 #reference height a*kappa = h
deltaR = 0.2 #deviation r
deltaB = 0.1 #deviation cone
R0 = 0.88 #using the nominal divertor R (where it starts)
Z = 1.602 #on a single z surface here.  use z of each vertex in cad

theta = np.linspace(0,2*np.pi,100)
x = np.cos(theta)
y = np.sin(theta)

xDist = (np.sin(n*theta)*deltaR/R0 + deltaB*Z/h + 1)*x
yDist = (np.cos(n*theta)*deltaR/R0 + deltaB*Z/h + 1)*y


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, name="undistorted",))
fig.add_trace(go.Scatter(x=xDist, y=yDist, name="distorted",))
fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
fig.show()
