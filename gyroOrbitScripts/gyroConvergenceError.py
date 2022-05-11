#plots for time benchmarking resolution vs # mesh elements
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



x = [1, 2**3, 3**3, 4**3, 5**3]
y1 = [
601.258999,
4013.002023,
13958.50474,
35752.21036,
76027.5784,
]
y2 = [
15.91,
3.24,
0.88,
1.18,
0.59,
]

#fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=x, y=y2, name="Error %", line=dict(color='rgb(17,119,51)', width=5, dash='dot'),
                         mode='lines+markers', marker_symbol='cross', marker_size=14),
             secondary_y=False)
fig.add_trace(go.Scatter(x=x, y=y1, name="Simulation Time [s]", line=dict(color='royalblue', width=5, dash='solid'),
                         mode='lines+markers', marker_symbol='circle', marker_size=14),
              secondary_y=True)

fig.update_layout(xaxis_tickformat = 'd')
fig.update_layout(
#title="Temperature Probe Time Evolution",
xaxis_title="N Simulations: N = NgP*NvP*NvS",
#yaxis_title="% Error (log scale)",
font=dict(
    family="Arial",
    size=24,
    color="Black"
),
margin=dict(
    l=5,
    r=5,
    b=5,
    t=5,
    pad=2
),
)

fig.update_yaxes(title_text="Simulation Time [s]", secondary_y=True)
fig.update_yaxes(title_text="Error From Optical Prediction [%]", secondary_y=False)



fig.update_layout(legend=dict(
    yanchor="middle",
    y=0.9,
    xanchor="center",
    x=0.5
))


fig.show()
#epsFile = '/home/tom/phd/dissertation/diss/figures/gyroError.eps'
#fig.write_image(epsFile)
