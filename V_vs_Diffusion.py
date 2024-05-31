#plots 0.5 location as a function of time (from Ficks law)
#overlays a constant velocity

import numpy as np
import plotly.graph_objects as go

v = 0.7 #m/s
D = 0.00004676195389 #[m^2/s]

tMax = 1.0e-3 #
t = np.linspace(0, tMax, 100)

#use erf(0.5) = 0.5 approximation from Fick's law for diffusion rate:
x = np.sqrt(D*t)

#print time of intersection
tIntersect = D/v**2
xIntersect = v*tIntersect
print("Time of intersection: {:f} [s]".format(tIntersect))
print("Position of intersection: {:f} [m]".format(xIntersect))


colors = []
symbols = ['x', 'star', 'diamond', 'asterisk', 'bowtie', 'hourglass', 'circle-x', 'hexagram' ]

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=x, name='Diffusion 1/2', line=dict(width=2,),
                     mode='lines+markers', marker_size=10, marker_symbol=symbols[1], 
                     marker=dict(maxdisplayed=30)))

fig.add_trace(go.Scatter(x=t, y=v*t, name='Velocity', line=dict(width=2,),
                     mode='lines+markers', marker_size=10, marker_symbol=symbols[2], 
                     marker=dict(maxdisplayed=30)))
fig.show()
