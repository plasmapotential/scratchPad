#plots comparing MT, MT Hybrid, SV, SV Hybrid from intersections.py
#created during benchmark of MT vs SV on North Haven Island, ME

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



x = [1,10,100,1000,10000,11000]
y1 = [
0.0002,
0.001098,
0.009909,
0.156431,
7.684268,
10.989756,
]
y2 = [
0.000179,
0.001618,
0.016214,
0.274099,
15.971446,
25.257792,
]
y3 = [
0.00019,
0.000234,
0.002156,
0.340941,
35.052392,
50.709693,
]


#fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": False}]])
fig.add_trace(go.Scatter(x=x, y=y1, name="MT Hyrbid", line=dict(color='royalblue', width=4, dash='solid'),
                         mode='lines+markers', marker_symbol='circle', marker_size=14),
              secondary_y=False)
fig.add_trace(go.Scatter(x=x, y=y2, name="SV Hybrid", line=dict(color='red', width=4, dash='dot'),
                         mode='lines+markers', marker_symbol='square', marker_size=18),
             secondary_y=False)
fig.add_trace(go.Scatter(x=x, y=y2, name="SV Matrix", line=dict(color='green', width=4, dash='solid'),
                         mode='lines+markers', marker_symbol='cross', marker_size=14),
             secondary_y=False)

fig.update_layout(
#title="Temperature Probe Time Evolution",
xaxis_title="N Line Segs AND N Faces",
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
fig.update_xaxes(type="log")
fig.update_yaxes(title_text="Time [s]", secondary_y=False, type='log')
#fig.update_yaxes(title_text="# Faces", secondary_y=True)


fig.update_layout(legend=dict(
    yanchor="middle",
    y=0.9,
    xanchor="right",
    x=0.5
))


fig.show()
