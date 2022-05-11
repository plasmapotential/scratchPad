#plots HF from csv file along a toroidal slice, ie the s direction

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys


#location of HF csv files (here we have three)
file1 = '/home/tom/HEAT/data/nstx_204118/000100/smallPolOut/HF_optical.csv'
file2 = '/home/tom/HEAT/data/nstx_204118/000100/smallPolOut/HF_gyro.csv'
file3 = '/home/tom/HEAT/data/nstx_204118/000100/smallPolOut/HF_allSources.csv'
#file = '/home/tom/HEAT/data/nstx_204118/001004/narrowSlice/HF_gyro.csv'
data = pd.read_csv(file1)
xyz = data.iloc[:,0:3].values
q_opt = data.iloc[:,3].values
data2 = pd.read_csv(file2)
q_gyro = data2.iloc[:,3].values
data3 = pd.read_csv(file3)
q_all = data3.iloc[:,3].values


#location of toroidal slice / chord
theta = -3 #deg
#max distance from that toroidal slice that we include points from on either side
threshDist = 3 #mm

#w = np.array([0,1,0])
w = np.array([-np.sin(np.radians(theta)),np.cos(np.radians(theta)),0])

orig = np.array([0,0,0])
d = -np.dot(w,orig)
dist = np.dot(xyz, w)
norm = np.repeat(w[np.newaxis, :], len(dist), axis=0)
projected = xyz - np.multiply(norm, dist[:,np.newaxis])


if np.all(w==[0,0,1]):
    u = np.cross(w,[0,1,0]) #prevent failure if bhat = [0,0,1]
else:
    u = np.cross(w,[0,0,1]) #this would fail if bhat = [0,0,1] (rare)
v = np.cross(w,u)
#normalize
u = u / np.sqrt(u.dot(u))
v = v / np.sqrt(v.dot(v))
w = w / np.sqrt(w.dot(w))

x_u = np.dot(projected, u)
y_v = np.dot(projected, v)


distFromPlane = dist + d
use = np.where(np.abs(distFromPlane) < threshDist)[0]

x_use = x_u[use]
y_use = y_v[use]

order = np.zeros((len(use)), dtype=int)
idxOld = 0
idxClosest = 0
i=0
order[0] = use[idxClosest]
use2 = use
while True:
    x0 = x_use[idxClosest]
    y0 = y_use[idxClosest]
    x_use = np.delete(x_use,idxClosest)
    y_use = np.delete(y_use,idxClosest)
    use2 = np.delete(use2,idxClosest)
    dX = x0-x_use
    dY = y0-y_use
    if len(dX) == 0:
        break
    else:
        dist = dX**2+dY**2
        idxClosest = np.argmin(dist)
        #NEED TO HANDLE CASES WHEN dist IS SHORTER THAN RESOLUTION HERE
        order[i] = use2[idxClosest]
        i+=1


x = x_u[order]
y = y_v[order]

#plot of all points
#fig = go.Figure(data=go.Scatter(x=x_use, y=y_use, mode='markers', hovertext=list(map(str, use))))

#this plot shows the perimeter of the PFC if everything went correctly
#fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', hovertext=list(map(str, order))))
#fig.show()

#this plot shows HF along the perimeter
distX = np.diff(x)
distY = np.diff(y)
s = np.sqrt(distX**2+distY**2)
s = np.insert(s,0,0)
s = np.cumsum(s)
fig = go.Figure(data=go.Scatter(x=s, y=q_gyro[order], name="Only Gyro", line=dict(color='royalblue', width=4, dash='solid'),
                         mode='lines', marker_symbol='circle', marker_size=6))
fig.add_trace(go.Scatter(x=s, y=q_opt[order], name="Only Optical", line=dict(color='rgb(17,119,51)', width=4, dash='solid'),
                         mode='lines', marker_symbol='cross', marker_size=6))

fig.update_layout(
#title="Temperature Probe Time Evolution",
xaxis_title="Distance Along PFC [m]",
yaxis_title="Heat Flux $MW/m^2$",
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




fig.update_layout(legend=dict(
    yanchor="middle",
    y=0.9,
    xanchor="right",
    x=0.9
))



fig.show()
