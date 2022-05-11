import numpy as np
import plotly.graph_objects as go


v = 1.0
gyroPhase = np.linspace(0,2*np.pi,100)
vPhase = np.linspace(0,np.pi,100)

#u,v = np.meshgrid(gyroPhase, vPhase)
#x = v * np.cos(u) * np.sin(v)
#y = v * np.sin(u) * np.sin(v)
#z = v * np.cos(v)
#fig = go.Figure(data=[go.Scatter3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), mode='markers')])


fig = go.Figure()
x = v * np.outer(np.cos(gyroPhase),np.sin(vPhase))
y = v * np.outer(np.sin(gyroPhase),np.sin(vPhase))
z = v * np.outer(np.ones(100),np.cos(vPhase))
fig.add_trace(go.Surface(x=x, y=y, z=z))


gyroPhase = np.ones(100) * np.pi / 4
x1 = v * np.cos(gyroPhase) * np.sin(vPhase)
y1 = v * np.sin(gyroPhase) * np.sin(vPhase)
z1 = v * np.ones(100) * np.cos(vPhase)

fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=dict(color='blue')))


fig.show()
