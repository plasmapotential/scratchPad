import plotly.graph_objects as go
import numpy as np


xs = np.array([0,0,1,1])
xNew = np.array([2,2,3,3])
ys = np.array([0,1,1,0])
yNew = np.array([0,1,1,0])

fig = go.Figure()
trace = go.Scatter(x=xs, y=ys, mode='lines+markers', marker_size=2, fill="toself")

trace['x'] = np.append(None, trace['x'])
trace['y'] = np.append(None, trace['y'])
trace['x'] = np.append(xNew, trace['x'])
trace['y'] = np.append(yNew, trace['y'])

print(trace['x'])
fig.add_trace(trace)
fig.show()
