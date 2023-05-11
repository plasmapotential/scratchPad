import plotly.graph_objects as go
import numpy as np

grid = np.zeros((2000,4000))
grid[850:1150,1500:2500] = 1.0

fig = go.Figure(data=go.Heatmap(z=grid))

fig.add_trace(go.Scatter(x=[1000, 1500], y=[1000, 2000]))
fig.show()
