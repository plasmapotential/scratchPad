import numpy as np
import plotly.graph_objects as go


omega = 2*np.pi*100.0
dtMax = 0.01
t = np.linspace(0,dtMax, 11)
y = np.sin(omega * t)
y2 = np.sin(omega * t + np.pi/4.0)



fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=y))
fig.add_trace(go.Scatter(x=t, y=y2))
fig.show()