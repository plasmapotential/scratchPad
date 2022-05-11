import numpy as np
import plotly.graph_objects as go

N=30
x = np.linspace(0,N,31)
pfc1 = np.zeros((N))
pfc1[9] = 1
pfc1[10] = 1
pfc1[11] = 1
pfc1[12] = 1

pfc2 = np.zeros((N))
pfc2[20] = 1
pfc2[21] = 1
pfc2[22] = 1
pfc2[23] = 1

pfc4Plot1 = pfc1.copy()
pfc4Plot1[12] = 0
pfc4Plot2 = pfc2.copy()
pfc4Plot2[23] = 0

pfc4Plot = pfc1+pfc2
pfc4Plot[12] = 0
pfc4Plot[23] = 0

bigPFC = pfc1+pfc2

f1 = np.array([
0,
1.0/3.0,
1.0/3.0,
1.0/3.0,
0,
])

f2 = np.array([
0,
1.0/7.0,
1.0/7.0,
1.0/7.0,
1.0/7.0,
1.0/7.0,
1.0/7.0,
1.0/7.0,
0,
])

f3 = np.array([0,1,0])

conv1_1 = np.convolve(f1,pfc1,'same')
conv2_1 = np.convolve(f2,pfc1,'same')
conv1_2 = np.convolve(f1,pfc2,'same')
conv2_2 = np.convolve(f2,pfc2,'same')

convBig = np.convolve(f2,bigPFC,'same')



fig = go.Figure(data=go.Scatter(x=x, y=pfc4Plot1, name="PFC1", line=dict(color='royalblue', width=4, dash='solid', shape='hv'),
                         mode='lines', marker_symbol='circle', marker_size=6))
fig.add_trace(go.Scatter(x=x, y=pfc4Plot2, name="PFC2", line=dict(color='royalblue', width=4, dash='solid', shape='hv'),
                         mode='lines', marker_symbol='cross', marker_size=6))
fig.add_trace(go.Scatter(x=x, y=conv1_1, name="fSlim*PFC1", line=dict(color='rgb(17,119,51)', width=4, dash='solid'),
                         mode='lines', marker_symbol='cross', marker_size=6))
fig.add_trace(go.Scatter(x=x, y=conv1_2, name="fSlim*PFC2", line=dict(width=4, dash='solid'),
                         mode='lines', marker_symbol='cross', marker_size=6))
fig.add_trace(go.Scatter(x=x, y=conv2_1, name="fWide*PFC1", line=dict(width=4, dash='solid'),
                         mode='lines', marker_symbol='cross', marker_size=6))
fig.add_trace(go.Scatter(x=x, y=conv2_2, name="fWide*PFC2", line=dict(width=4, dash='solid'),
                         mode='lines', marker_symbol='cross', marker_size=6))
fig.add_trace(go.Scatter(x=x, y=convBig, name="fSlim*AllPFCs", line=dict(width=4, dash='solid'),
                         mode='lines', marker_symbol='cross', marker_size=6))

#x=12
#r = 3.5
#y = 0.6
#fig.add_shape(type="circle",
#    xref="x", yref="y",
#    fillcolor="PaleTurquoise",
#    opacity=0.6,
#    x0=x-r, y0=0.6, x1=x+r, y1=1.0,
#    line_color="LightSeaGreen",
#    name="circ1"
#)

fig.show()
