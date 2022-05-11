#plots HF from csv file along a slice, ie the s direction

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
#for gunn / komm plots using Cube
pt = np.array([384.421, 396.983, -1603.5])
#for NSTXU castellation
#pt = np.array([476.692,274.396,-1603.31])

rPt = np.sqrt(pt[0]**2 + pt[1]**2)

#0 point for plots - corner of PFC where s=0.  get it from mesh pt
#toroidal
if mode == 'toroidal':
    #gunn / komm plots gyroConvergence2
    pt0 = np.array([380.606,400.667,-1603.5])
    #for nstxu castellation
    #pt0 = np.array([469.565,287.452,-1603.62])
    #tileEnd = 29.0
    #tileTop = -1.0
    #tileX0 = 0.0
    #tileX1 = 28.3

#poloidal
else:
    #gunn / komm plots gyroConvergence2
    pt0 = np.array([385.815,399.036,-1603.52])
    #for nstxu castellation
    #pt0 = np.array([489.315,280.878,-1603.3])
    #tileX0 = 0.0
    #tileX1 = 29.8

#max distance from the slice that we include points from on either side
threshDist = 0.5 #mm

#use this to flip plot around (right=>left starting point)
flip = True
#use this to plot all the points in a plane as projected
pointsInPlane = False

#use this to create a VTK object with points used in plot
vtk = False

#min and max x axis values for plot
sMin = -2
sMax = 12

#use this to add a box in plot where PFC is
addBox = False
#use this to add an arrow to show where PFC is
addArrow = False

#use this to write an EPS file
writeEPS = False
#epsFile = '/home/tom/phd/dissertation/diss/figures/sPolCubeHF.eps'

#location of HF csv files (here we have three)
#file1 = '/home/tom/HEAT/data/nstx_204118/001004/Cube/HF_optical.csv'
#file2 = '/home/tom/HEAT/data/nstx_204118/001004/Cube/HF_gyro.csv'
#file3 = '/home/tom/HEAT/data/nstx_204118/001004/Cube/HF_allSources.csv'
file1 = '/home/tom/results/gyroConvergence2/diamagnetic/10eV/5gP5vP5vS/HF_optical.csv'
file2 = '/home/tom/results/gyroConvergence2/diamagnetic/10eV/5gP5vP5vS/HF_gyro.csv'
file3 = '/home/tom/results/gyroConvergence2/diamagnetic/10eV/5gP5vP5vS/HF_allSources.csv'
#file1 = '/home/tom/results/gyroConvergence2/10eV/3gP3vP3vS/HF_optical.csv'
#file2 = '/home/tom/results/gyroConvergence2/10eV/3gP3vP3vS/HF_gyro.csv'
#file3 = '/home/tom/results/gyroConvergence2/10eV/3gP3vP3vS/HF_allSources.csv'
#file = '/home/tom/HEAT/data/nstx_204118/001004/narrowSlice/HF_gyro.csv'
#file1 = '/home/tom/HEAT/data/nstx_204118/000200/castellationTop/HF_optical.csv'
#file2 = '/home/tom/HEAT/data/nstx_204118/000200/castellationTop/HF_gyro.csv'
#file3 = '/home/tom/HEAT/data/nstx_204118/001004/Cube/HF_allSources.csv'
#file1 = '/home/tom/results/castellationTopDiamagnetic/newPointClouds/HF_optical.csv'
#file2 = '/home/tom/results/castellationTopDiamagnetic/newPointClouds/HF_gyro.csv'
#file3 = '/home/tom/results/castellationTopDiamagnetic/newPointClouds/HF_allSources.csv'
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

print(w)

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
    pcfile = '/home/tom/source/test/'+mode+'.csv'
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

#X axis title
if mode=='poloidal':
    xAxisTitle = 'S<sub>pol</sub> [mm]'
else:
    xAxisTitle = 'S<sub>tor</sub> [mm]'

fig = go.Figure()
if addBox == True:
#Shade Boxes
    fig.add_vrect(
    x0=-5, x1=-1.0,
    fillcolor="Gray", opacity=0.5,
    layer="below", line_width=0,
    ),
    fig.add_vrect(
    x0=29.8, x1=35,
    fillcolor="Gray", opacity=0.5,
    layer="below", line_width=0,
    ),
    #shade annotation
    fig.add_annotation(
        x=32.0,
        y=10.0,
        xref="x",
        yref="y",
        text="Optically Shadowed",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=20,
            color="white"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="gray",
        opacity=0.8,
        textangle=90
        )

#    fig.add_shape(type="rect",
#        xref="x", yref="y",
#        x0=0, y0=0,
#        x1=tileEnd, y1=tileTop,
#        line=dict(
#            color="Black",
#            width=2,
#        ),
#        fillcolor="fuchsia",
#        opacity=0.5,
#        )

#    fig.add_shape(type="rect",
#        xref="x", yref="y",
#        x0=-5, y0=0,
#        x1=-0.62, y1=-3,
#        line=dict(
#            color="Black",
#            width=2,
#        ),
#        fillcolor="Gray",
#        opacity=0.5,
#        )
#
#    fig.add_shape(type="rect",
#        xref="x", yref="y",
#        x0=29.62, y0=-3,
#        x1=35, y1=0,
#        line=dict(
#            color="Black",
#            width=2,
#        ),
#        fillcolor="Gray",
#        opacity=0.5,
#        )

#this calculates the integrals under the curves
gyroInt = np.trapz(q_gyro, s/1000.0)
optInt = np.trapz(q_opt, s/1000.0)

print("Gyro Integral:")
print(gyroInt)
print("Optical Integral:")
print(optInt)

fig.add_trace(go.Scatter(x=s-s[loc0], y=q_gyro, name="Ion Gyro", line=dict(color='royalblue', width=4, dash='solid'),
                         mode='lines', marker_symbol='circle', marker_size=6))
fig.add_trace(go.Scatter(x=s-s[loc0], y=q_opt, name="Ion Optical", line=dict(color='rgb(17,119,51)', width=4, dash='solid'),
                         mode='lines', marker_symbol='cross', marker_size=6))


if addArrow == True:
    #Arrow
    fig.add_annotation(
        x=tileX0,  # arrows' head
        y=-1,  # arrows' head
        ax=tileX1,  # arrows' tail
        ay=-1,  # arrows' tail
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',  # if you want only the arrow
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=4,
        arrowcolor='fuchsia'
    )
    fig.add_annotation(
        x=tileX1,  # arrows' head
        y=-1,  # arrows' head
        ax=tileX0,  # arrows' tail
        ay=-1,  # arrows' tail
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',  # if you want only the arrow
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=4,
        arrowcolor='fuchsia'
    )
#Arrow Text
fig.add_annotation(
        x=15,
        y=-0.6,
        xref="x",
        yref="y",
        text="PFC Surface",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=20,
            color="white"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="fuchsia",
        opacity=0.8,
        )

fig.update_layout(
#title="Temperature Probe Time Evolution",
xaxis_title=xAxisTitle,
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

fig.update_xaxes(range=[sMin,sMax])




fig.update_layout(legend=dict(
    yanchor="middle",
    y=0.9,
    xanchor="right",
    x=0.80
))



fig.show()

if writeEPS==True:
    fig.write_image(epsFile)
