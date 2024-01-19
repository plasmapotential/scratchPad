#makes a fake trace as a function of time.  heaviside functions
import numpy as np
import plotly.graph_objects as go

t = np.linspace(0, 10, 100)
trace1 = np.zeros((len(t)))
trace2 = np.zeros((len(t)))
trace3 = np.zeros((len(t)))

trace1[5:30] = 1.0
trace2[31:78] = 1.0
trace3[80:98] = 1.0





# Plot using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=trace1, name='SEED1_FWD'))
fig.add_trace(go.Scatter(x=t, y=trace2, name='SEED2_FWD'))
fig.add_trace(go.Scatter(x=t, y=trace3, name='SEED3_FWD'))

fig.update_layout(xaxis_title='[s]', yaxis_title='Sequence State', font=dict(family="Arial",size=20,))
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
fig.show()