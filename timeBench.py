#plots for time benchmarking resolution vs # mesh elements
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



x = [0,5,10,15,20,25]
y1 = [
54.215024,
84.214273,
115.884458,
135.507866,
146.059326,
177.935793,
]
y2 = [
4389.76,
37.93,
16.53,
3.71,
2.62,
2.37,
]

#fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=x, y=y2, name="Error %", line=dict(color='rgb(17,119,51)', width=5, dash='dot'),
                         mode='lines+markers', marker_symbol='cross', marker_size=14),
             secondary_y=False)
fig.add_trace(go.Scatter(x=x, y=y1, name="Simulation Time [s]", line=dict(color='royalblue', width=5, dash='solid'),
                         mode='lines+markers', marker_symbol='circle', marker_size=14),
              secondary_y=True)


fig.update_layout(
#title="Temperature Probe Time Evolution",
xaxis_title="Length of Intersection Check Trace [degrees] ",
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
fig.update_yaxes(title_text="Error %", secondary_y=False, type='log')



fig.update_layout(legend=dict(
    yanchor="middle",
    y=0.9,
    xanchor="right",
    x=0.5
))


fig.show()
epsFile = '/home/tom/phd/dissertation/diss/figures/timeBenchmark1.eps'
fig.write_image(epsFile)
