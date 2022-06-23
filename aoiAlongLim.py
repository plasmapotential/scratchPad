#aoiAlongLim.py
#Description:   calculate B field angle of incidence (AOI) along the rlim,zlim
#               contour from gfile
#Date:          20220621
#engineer:      T Looby
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shutil

#you need equilParams_class to run this script
#if you dont have it you can run in the HEAT docker container and point EFITpath
#to /root/source
EFITpath = '/home/tom/source'
sys.path.append(EFITpath)
import EFIT.equilParams_class as EP
from scipy.interpolate import interp1d

#divertor points
points = [[1.41,	-1.137],
          [1.287,	-1.224],
          [1.285,	-1.235],
          [1.45,	-1.184],
          [1.448,	-1.184],
          [1.58,	-1.303],
          [1.578,	-1.303],
          [1.82,	-1.6],
          [1.82,	-1.6],
          [1.73,	-1.41],
          [1.73,	-1.41],
          [1.68,	-1.295]]

SPts = [
        1.18856079,
        1.33959099,
        1.40659099,
        1.57929304,
        1.58129304,
        1.7717923,
        1.7737923,
        2.15690194,
        2.15690194,
        2.53690194,
        2.5369019,
        2.6623013,
        ]


#Resolution in S direction
numS=500
#geqdsk file
gIn = '/home/tom/HEATruns/SPARC/scenarios_FreeGS/sparc/geqdsk_flattop_negPsi'
#gIn = '/home/tom/HEATruns/SPARC/scenarios_FreeGS/sparc/geqdsk_kappa165lsn_negPsi'
#gIn = '/home/tom/HEATruns/SPARC/scenarios_FreeGS/sparc/geqdsk_q1run_negPsi'
#gIn = '/home/tom/HEATruns/SPARC/scenarios_FreeGS/sparc/geqdsk_8T_negPsi'

#copy file to tmp location with new name so that EP class can read it
gRenamed = '/home/tom/HEAT/data/tmpDir/g000001.00001'
shutil.copyfile(gIn, gRenamed)

#Minimum S.  S=0 is midplane [m]
minS=1.1
#S greater than actual s maximum defaults to s[-1] [m]
maxS=2.7
#tile index for plotting single tile green overlay
Tidx = 5

#masks
sectionMask = True
interpolateMask = True
plotMaskContour=False
plotMaskAOI=False
plotMaskSingleT = False
plotMaskAllT = True


# Calculate distance along curve/wall (also called S):
def distance(rawdata):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(rawdata,axis=0)**2,axis=1)))
    distance = np.insert(distance, 0, 0)
    return distance

def normals(rawdata):
    N = len(rawdata) - 1
    norms = np.zeros((N,3))
    for i in range(N):
        RZvec = rawdata[i+1] - rawdata[i]
        vec1 = np.array([[RZvec[0], 0.0, RZvec[1]]])
        vec2 = np.array([[0.0, 1.0, 0.0]])
        n = np.cross(vec1, vec2)
        norms[i,:] = n / np.linalg.norm(n,axis=1)
    return norms


def centers(rz):
    centers = np.zeros((len(rz)-1, 2))
    dR = np.diff(rz[:,0])
    dZ = np.diff(rz[:,1])
    centers[:,0] = rz[:-1,0] + dR/2.0
    centers[:,1] = rz[:-1,1] + dZ/2.0
    return centers

#load gfile
ep = EP.equilParams(gRenamed)
data, idx = np.unique(ep.g['wall'], axis=0, return_index=True)
rawdata = data[np.argsort(idx)]
#close the contour
rawdata = np.vstack([rawdata, rawdata[0]])

# Call distance function to return distance along S
dist = distance(rawdata)

#get center points for entire contour
ctrs = centers(rawdata)
distCtrs = distance(ctrs)

# normal at each surface for entire contour
norms = normals(rawdata)
norms2D = np.delete(norms, 1, axis=1)


#interpolate if flag is true
if interpolateMask is True:
    # Resolution we need given the inputs
    resolution = (dist[-1] - dist[0])/float(numS)
    print("Resolution: {:f} m".format(resolution))
    # Calculate how many total grid points we need based upon resolution and
    # total distance around curve/wall
    numpoints = int((dist[-1] - dist[0])/resolution)
    # Spline Interpolation (linear) - Make higher resolution wall.
    interpolator = interp1d(dist, rawdata, kind='slinear', axis=0)
    alpha = np.linspace(dist[0], dist[-1], numpoints)
    #add in raw data points so that we preserve the corners
    alpha = np.sort(np.hstack([alpha, dist]))
    interpolated_points = interpolator(alpha)
    arr, idx = np.unique(interpolated_points, axis=0, return_index=True)
    interpolated_points = arr[np.argsort(idx)]
    newdist = distance(interpolated_points)

    #get center points
    newCtrs = centers(interpolated_points)
    ctrsAndPts = np.empty((len(interpolated_points) + len(newCtrs), 2))
    ctrsAndPts[0::2] = interpolated_points
    ctrsAndPts[1::2] = newCtrs

    newDistCtrs = distance(ctrsAndPts)[1::2]

else:
    interpolated_points = rawdata
    #get center points
    newCtrs = centers(interpolated_points)
    newdist = dist
    ctrsAndPts = np.empty((len(interpolated_points) + len(newCtrs), 2))
    ctrsAndPts[0::2] = interpolated_points
    ctrsAndPts[1::2] = newCtrs
    newDistCtrs = distance(ctrsAndPts)[1::2]

# normal at each surface for entire contour
newNorms = normals(interpolated_points)
newNorms2D = np.delete(newNorms, 1, axis=1)

if sectionMask == True:
    idxDist = np.where(np.logical_and(newdist >= minS, newdist <= maxS))[0]
    idxDistCtrs = np.where(np.logical_and(newDistCtrs >= minS, newDistCtrs <= maxS))[0]
else:
    idxDist = np.arange(len(newdist))
    idxDistCtrs = np.arange(len(newDistCtrs))

R = newCtrs[:,0]
Z = newCtrs[:,1]

Brz = np.zeros((len(R), 2))

#B field from MHD Equilibrium
Brz[:,0] = ep.BRFunc.ev(R,Z)
Brz[:,1] = ep.BZFunc.ev(R,Z)
Bt = ep.BtFunc.ev(R,Z)
B = np.sqrt(Brz[:,0]**2 + Brz[:,1]**2 + Bt**2)
Brz[:,0] /= B
Brz[:,1] /= B

#test cases with predefined B field
#Brz[:,0] = 0.0
#Brz[:,1] = -1.0
#Brz[:,0] = 1.82 - 1.578
#Brz[:,1] = (-1.6) - (-1.303)
#Bnorm = np.linalg.norm(Brz[0], axis=0)
#Brz /= Bnorm


bdotn = np.multiply(Brz, newNorms2D).sum(1)
AOI = np.degrees(np.arcsin(bdotn))

#print S at each target start/end point
ptIdxs = []
print("R          Z          S")
for i,pt in enumerate(points):
    test = np.logical_and(pt[0]==interpolated_points[:,0], pt[1]==interpolated_points[:,1])
    iloc = np.where(test==True)[0][0]
    ptIdxs.append(iloc)
    line = "{:0.8f} {:0.8f} {:0.8f}".format(interpolated_points[iloc,0], interpolated_points[iloc,1], newdist[iloc])
    print(line)

ptIdxs = np.array(ptIdxs)


#print all R,Z,S within section
#print("R          Z          S")
#for i,S in enumerate(dist[idxDist]):
#    line = "{:0.8f} {:0.8f} {:0.8f}".format(rawdata[i,0], rawdata[i,1], S)
#    print(line)


#Now print max AOI along each of the sections defined above
starts = points[0::2]
ends = points[1::2]
startIdxs = ptIdxs[0::2]
endIdxs = ptIdxs[1::2]

minAOI = []
maxAOI = []
avgAOI = []
NPFCs = int(len(points) / 2.0)
for i in range(NPFCs):
    AOIidx = AOI[startIdxs[i]:endIdxs[i]]
    minAOI.append(np.min(np.abs(AOIidx)))
    maxAOI.append(np.max(np.abs(AOIidx)))
    avgAOI.append(np.average(np.abs(AOIidx)))

minAOI = np.array(minAOI)
maxAOI = np.array(maxAOI)
avgAOI = np.array(avgAOI)

outputMat = np.vstack([minAOI,maxAOI,avgAOI]).T
print(outputMat)


#plot the PFC RZ contour with Region of Interest overlaid
if plotMaskContour is True:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rawdata[:,0], y=rawdata[:,1], name="Entire Contour"))
    fig.add_trace(go.Scatter(x=interpolated_points[idxDist,0],
                             y=interpolated_points[idxDist,1],
                             name="Region of Interest", mode='lines+markers'))
    #fig.add_trace(go.Scatter(x=ctrs[:,0], y=ctrs[:,1]))
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
        ))
    fig.update_xaxes(title="R [m]")
    fig.update_yaxes(title="Z [m]")
    fig.show()

#generic AOI plot.  No PFC tile overlays
if plotMaskAOI is True:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=newDistCtrs[idxDistCtrs], y=AOI[idxDistCtrs], mode='lines+markers'))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Distance Along Contour [m]")
    fig.update_yaxes(title="Angle of Incidence [degrees]")
    fig.show()

#plot for a single tile (Tidx above).  Set Smin Smax for tile first
if plotMaskSingleT == True:
    startIdxs = SPts[0::2]
    endIdxs = SPts[1::2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=newDistCtrs[idxDistCtrs], y=AOI[idxDistCtrs], mode='lines+markers'))
    fig.add_vrect(x0=startIdxs[Tidx], x1=endIdxs[Tidx],
              annotation_text="Tile of Interest", annotation_position="top left",
              fillcolor="green", opacity=0.25, line_width=0)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Distance Along Contour [m]")
    fig.update_yaxes(title="Angle of Incidence [degrees]")
    fig.show()

#plot for all tiles.  set Smin and Smax to entire divertor first
if plotMaskAllT == True:
    startIdxs = SPts[0::2]
    endIdxs = SPts[1::2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=newDistCtrs[idxDistCtrs], y=AOI[idxDistCtrs], mode='lines+markers'))
    for Tidx in range(6):
        fig.add_vrect(x0=startIdxs[Tidx], x1=endIdxs[Tidx],
                annotation_text="T{:d}".format(Tidx+1), annotation_position="top left",
                fillcolor=px.colors.qualitative.Vivid[Tidx], opacity=0.25, line_width=0)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Distance Along Contour [m]")
    fig.update_yaxes(title="Angle of Incidence [degrees]")
    fig.update_xaxes(range=[minS-0.03, maxS+0.03])
    fig.update_yaxes(range=[-8, 8])
    fig.show()
