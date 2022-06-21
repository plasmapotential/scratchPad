#aoiAlongLim.py
#Description:   calculate B field angle of incidence (AOI) along the rlim,zlim
#               contour from gfile
#Date:          20220621
#engineer:      T Looby
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

#you need equilParams_class to run this script
EFITpath = '/home/tom/source'
sys.path.append(EFITpath)
import EFIT.equilParams_class as EP
from scipy.interpolate import interp1d

#Resolution in S direction
numS=1000
#geqdsk file
gIn = '/home/tom/HEATruns/SPARC/scenarios_FreeGS/sparc/g000001.00010_geqdsk_flattop_negPsi'
#Minimum S.  S=0 is midplane [m]
minS=3
#S greater than actual s maximum defaults to s[-1] [m]
maxS=5.6
#masks
sectionMask = True
interpolateMask = True
plotMaskContour=True
plotMaskAOI=True


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
ep = EP.equilParams(gIn)
data, idx = np.unique(ep.g['wall'], axis=0, return_index=True)
rawdata = data[np.argsort(idx)]
#close the contour
rawdata = np.vstack([rawdata, rawdata[0]])

# Call distance function to return distance along S
dist = distance(rawdata)
if sectionMask == True:
    idxDist = np.where(np.logical_and(dist >= minS, dist <= maxS))[0]
else:
    idxDist = np.arange(len(dist))

# normal at each surface
norms = normals(rawdata[idxDist])
norms2D = np.delete(norms, 1, axis=1)

#get center points
ctrs = centers(rawdata[idxDist])
distCtrs = distance(ctrs)


if interpolateMask is True:
    # Resolution we need given the inputs
    resolution = (maxS - minS)/float(numS)
    # Calculate how many total grid points we need based upon resolution and
    # total distance around curve/wall
    numpoints = int((dist[idxDist][-1] - dist[idxDist][0])/resolution)
    # Spline Interpolation (linear) - Make higher resolution wall.
    interpolator = interp1d(dist, rawdata, kind='slinear', axis=0)
    alpha = np.linspace(dist[idxDist][0], dist[idxDist][-1], numpoints)
    interpolated_points = interpolator(alpha)
    newdist = distance(interpolated_points)
    #get center points
    newCtrs = centers(interpolated_points)

    # Spline Interpolation (linear) - Make higher resolution wall.
    interpolator2 = interp1d(distCtrs, norms2D, kind='slinear', axis=0)
    alpha2 = np.linspace(distCtrs[0], distCtrs[-1], numpoints-1)
    newNorms2D = interpolator2(alpha2)


else:
    interpolated_points = rawdata[idxDist]
    #get center points
    newCtrs = centers(interpolated_points)
    newdist = dist[idxDist]
    newNorms2D = norms2D

R = newCtrs[:,0]
Z = newCtrs[:,1]

Brz = np.zeros((len(R), 2))
Brz[:,0] = ep.BRFunc.ev(R,Z)
Brz[:,1] = ep.BZFunc.ev(R,Z)
Bp = np.sqrt(Brz[:,0]**2 + Brz[:,1]**2)
Brz[:,0] /= Bp
Brz[:,1] /= Bp

bdotn = np.multiply(Brz, newNorms2D).sum(1)
AOI = np.degrees(np.arcsin(np.abs(bdotn)))

if plotMaskContour is True:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rawdata[:,0], y=rawdata[:,1]))
    fig.add_trace(go.Scatter(x=rawdata[idxDist,0], y=rawdata[idxDist,1]))
    #fig.add_trace(go.Scatter(x=ctrs[:,0], y=ctrs[:,1]))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
    fig.show()


if plotMaskAOI is True:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=newdist, y=AOI))
    fig.update_layout(showlegend=False)
    fig.show()
