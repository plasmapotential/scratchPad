#plots HF from csv file along a toroidal slice, ie the s direction

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os
HEATPath = '/home/tom/source/HEAT/github/source'
sys.path.append(HEATPath)
import toolsClass
tools = toolsClass.tools()
tools.rootDir = HEATPath

#================== USER INPUTS ======================================
#Choose between 'poloidal' or 'toroidal' chord
mode = 'toroidal'

#point on center of PFC
pt = np.array([384.421, 396.983, -1603.5])
rPt = np.sqrt(pt[0]**2 + pt[1]**2)

#0 point for plots - corner of PFC where s=0.  get it from mesh pt
#toroidal
pt0 = np.array([380.524,400.596,-1603.51])
#poloidal
#pt0 = np.array([385.815,399.036,-1603.52])

#max distance from the slice that we include points from on either side
threshDist = 0.5 #mm

#use this to flip plot around (right=>left starting point)
flip = True

#use this to plot all the points in a plane as projected
pointsInPlane = False

#use this to create a VTK object with points used in plot
vtk = False

#location of HF csv files (here we have three)
#file1 = '/home/tom/HEAT/data/nstx_204118/001004/Cube/HF_optical.csv'
#file2 = '/home/tom/HEAT/data/nstx_204118/001004/Cube/HF_gyro.csv'
#file3 = '/home/tom/HEAT/data/nstx_204118/001004/Cube/HF_allSources.csv'
file1 = '/home/tom/results/gyroConvergence2/10eV/5gP5vP5vS/HF_optical.csv'
file2 = '/home/tom/results/gyroConvergence2/10eV/5gP5vP5vS/HF_gyro.csv'
file3 = '/home/tom/results/gyroConvergence2/10eV/5gP5vP5vS/HF_allSources.csv'
#file1 = '/home/tom/results/gyroConvergence2/10eV/3gP3vP3vS/HF_optical.csv'
#file2 = '/home/tom/results/gyroConvergence2/10eV/3gP3vP3vS/HF_gyro.csv'
#file3 = '/home/tom/results/gyroConvergence2/10eV/3gP3vP3vS/HF_allSources.csv'
#file = '/home/tom/HEAT/data/nstx_204118/001004/narrowSlice/HF_gyro.csv'
#===============================================================================

data = pd.read_csv(file1)
xyz = data.iloc[:,0:3].values
q_opt = data.iloc[:,3].values
data2 = pd.read_csv(file2)
q_gyro = data2.iloc[:,3].values
data3 = pd.read_csv(file3)
q_all = data3.iloc[:,3].values

#translate coordinates to this point
xyz2 = xyz - pt

#construct plane
rVec = np.array([pt[0],pt[1],0])
zVec =np.array([0,0,1])
perpVec = np.cross(rVec,zVec)

#create plane for projections
#do this for poloidal slice
if mode == 'poloidal':
    w = perpVec / np.linalg.norm(perpVec)
#do this for toroidal slice
elif mode == 'toroidal':
    w = rVec / np.linalg.norm(rVec)
#old example plane
else:
    w = np.array([0,1,0])

orig = np.array([0,0,0])
d = -np.dot(w,orig)
dist = np.dot(xyz2, w)
norm = np.repeat(w[np.newaxis, :], len(dist), axis=0)
projected = xyz2 - np.multiply(norm, dist[:,np.newaxis])

#create coordinate system where x,y are directions in plane
#(see plots below which will print all points in x,y plane)
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

#plot of all points as projected onto the plane
if pointsInPlane == True:
    fig = go.Figure(data=go.Scatter(x=x_use, y=y_use, mode='markers', hovertext=list(map(str, use))))
    #this plot shows the perimeter of the PFC only if everything went correctly
    #fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', hovertext=list(map(str, order))))
    fig.show()

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



#if desired, flip arrays
if flip == True:
    x = np.flip(xyz[order,0])
    y = np.flip(xyz[order,1])
    z = np.flip(xyz[order,2])
    q_gyro = np.flip(q_gyro[order])
    q_opt = np.flip(q_opt[order])
    q_all = np.flip(q_all[order])

else:
    x = xyz[order,0]
    y = xyz[order,1]
    z = xyz[order,2]
    q_gyro = q_gyro[order]
    q_opt = q_opt[order]
    q_all = q_all[order]

#now project these freshly ordered points to plane (u,v)
xyzOrdered = np.vstack([x,y,z]).T
dist = np.dot(xyzOrdered, w)
norm = np.repeat(w[np.newaxis, :], len(dist), axis=0)
projected = xyzOrdered - np.multiply(norm, dist[:,np.newaxis])
x_uOrdered = np.dot(projected, u)
y_vOrdered = np.dot(projected, v)

#location of 0 pt
loc0 = np.argmin(np.linalg.norm(xyzOrdered - pt0, axis=1))
#print(loc0)
#print(xyz[order][loc0])

#use this to create a VTK object with points used in plot
if vtk == True:
    #make pointcloud of x and y for viewing in PV
    pcfile = '/home/tom/source/test/toroidalChord.csv'
    pc = np.zeros((len(x), 4))
    pc[:,0] = x
    pc[:,1] = y
    pc[:,2] = z
    pc[:,3] = np.ones((len(x)))
    head = "X,Y,Z,1"
    np.savetxt(pcfile, pc, delimiter=',',fmt='%.10f', header=head)
    pvpythonCMD = '/opt/paraview/ParaView-5.9.0-RC2-MPI-Linux-Python3.8-64bit/bin/pvpython'
    os.environ["pvpythonCMD"] = pvpythonCMD
    tools.createVTKOutput(pcfile, 'points', 'toroidalChord')

#this plot shows HF along the perimeter
distX = np.diff(x_uOrdered)
distY = np.diff(y_vOrdered)
s = np.sqrt(distX**2+distY**2)
s = np.insert(s,0,0)
s = np.cumsum(s)

fig = go.Figure(data=go.Scatter(x=s-s[loc0], y=q_gyro, name="Only Gyro", line=dict(color='royalblue', width=4, dash='solid'),
                         mode='lines', marker_symbol='circle', marker_size=6))
fig.add_trace(go.Scatter(x=s-s[loc0], y=q_opt, name="Only Optical", line=dict(color='rgb(17,119,51)', width=4, dash='solid'),
                         mode='lines', marker_symbol='cross', marker_size=6))

fig.update_layout(
#title="Temperature Probe Time Evolution",
xaxis_title="Distance Along PFC [mm]",
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
    x=0.1
))



fig.show()
