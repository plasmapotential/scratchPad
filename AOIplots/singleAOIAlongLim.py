#aoiAlongLim.py
#Description:   calculate B field angle of incidence (AOI) along the rlim,zlim
#               contour from gfile
#Date:          20220621
#engineer:      T Looby
import sys
import numpy as np
import scipy.interpolate as scinter
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

#v2y
#points = [
#            [1.385100, -1.115400],
#            [1.312500, -1.202500],
#            [1.320000, -1.210000],
#            [1.415700, -1.210000],
#            [1.415700, -1.205000],
#            [1.592200, -1.321400],
#            [1.587200, -1.321400],
#            [1.720000, -1.510000],
#            [1.720000, -1.575000],
#            [1.695000, -1.380000],
#            [1.695000, -1.380000],
#            [1.658800, -1.218900],
#        ]
#
#SPts = [
#    1.164718,
#    1.278107,
#    1.308107,
#    1.403807,
#    1.408807,
#    1.649849,
#    1.654849,
#    1.885513,
#    1.950513,
#    2.410513,
#    2.410513,
#    2.575630,
#]

#v3b
points = [
            [1385.1, -1115.4],
            [1294.9, -1223.6],
            [1320,   -1210],
            [1440.7, -1209],
            [1440.7, -1210],
            [1509.3, -1209],
            [1570.8, -1296.4],
            [1570,   -1297],
            [1720,   -1510],
            [1720,   -1575],
            [1840,   -1575],
            [1840,   -1380],
            [1695,   -1380],
            [1658.5, -1217.7]
        ]

SPts = [
        1164.1530442496035,
        1305.0192218792954,
        1333.5669012946173,
        1454.2710437256044,
        1455.2710437256044,
        1523.8783319681809,
        1630.7474575377298,
        1631.7474575377298,
        1892.2642516221035,
        1957.2642516221035,
        2077.2642516221035,
        2272.2642516221035,
        2417.2642516221035,
        2583.617911034814
]



#Resolution in S direction
numS=1000
#geqdsk file
#gIn = '/home/tom/HEATruns/SPARC/scenarios_FreeGS/sparc/geqdsk_flattop_negPsi'
#gIn = '/home/tom/HEATruns/SPARC/scenarios_FreeGS/sparc/geqdsk_kappa165lsn_negPsi'
#gIn = '/home/tom/HEATruns/SPARC/scenarios_FreeGS/sparc/geqdsk_q1run_negPsi'
#gIn = '/home/tom/HEATruns/SPARC/scenarios_FreeGS/sparc/geqdsk_8T_negPsi'
gIn = '/home/tom/work/CFS/GEQDSKs/MEQ_20230501/sparc_1350.EQDSK'

#copy file to tmp location with new name so that EP class can read it
gRenamed = '/home/tom/HEAT/data/tmpDir/g000001.00001'
shutil.copyfile(gIn, gRenamed)

#Minimum S.  S=0 is midplane [m]
minS=1.1
#S greater than actual s maximum defaults to s[-1] [m]
maxS=2.7
#tile index for plotting single tile green overlay
Tidx = 3

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
psiN = ep.psiFunc.ev(R,Z)
Bp = np.sqrt(Brz[:,0]**2 + Brz[:,1]**2)
Bt = ep.BtFunc.ev(R,Z)
B = np.sqrt(Brz[:,0]**2 + Brz[:,1]**2 + Bt**2)
Brz[:,0] /= B
Brz[:,1] /= B

#test cases with predefined B field
#Brz[:,0] = 0.0
#Brz[:,1] = -1.0
#Brz[:,0] = 1.82 - 1.578
#Brz[:,1] = (-1.6) - (-1.303)
#Bt = ep.BtFunc.ev(R,Z)
#B = np.sqrt(Brz[:,0]**2 + Brz[:,1]**2 + Bt**2)
#Brz[:,0] /= B
#Brz[:,1] /= B

bdotn = np.multiply(Brz, newNorms2D).sum(1)
AOI = np.degrees(np.arcsin(bdotn))

#calculate lambda_q at SPs
lq = 0.0003 #in meters at OMP
psiaxis = ep.g['psiAxis']
psiedge = ep.g['psiSep']
deltaPsi = np.abs(psiedge - psiaxis)
s_hat = psiN - 1.0
#map R coordinates up to the midplane via flux mapping
R_omp = np.linspace(ep.g['RmAxis'], ep.g['R1'] + ep.g['Xdim'], 100)
Z_omp = np.zeros(len(R_omp))
psi_omp = ep.psiFunc.ev(R_omp,Z_omp)
f = scinter.UnivariateSpline(psi_omp, R_omp, s = 0, ext = 'const')
R_mapped = f(psiN)
Z_mapped = np.zeros(len(R_mapped))
Bp_mapped = ep.BpFunc.ev(R_mapped,Z_mapped)
# Gradient
gradPsi = Bp_mapped*R_mapped
xfm = gradPsi / deltaPsi
# Decay width at target mapped to flux coordinates
lq_hat = lq * xfm

#calculate strike points
p0 = psiN[0]
S0 = newDistCtrs[0]
lq0 = lq_hat[0]
SPs = []
SPidxs = []
lqs = []
Nlqs = 1.0
Slqs = []
for i,p in enumerate(psiN):
    if p0 < 1.0:
        if p > 1.0:
            frac = (1.0 - p0) / (p - p0)
            S = (newDistCtrs[i] - S0)*frac + S0
            SPs.append(S)
            SPidxs.append(i)
            lqSP = (lq_hat[i] - lq0)*frac + lq0
            lqs.append(lqSP)
            Slq = (newDistCtrs[i] - S0) / (p - p0) * lqSP * Nlqs
            Slqs.append(Slq)

    elif p0 > 1.0:
        if p < 1.0:
            frac = (p0 - 1.0) / (p0 - p)
            S = (newDistCtrs[i] - S0)*frac + S0
            SPs.append(S)
            SPidxs.append(i)
            lqSP = lq0 - (lq0 - lq_hat[i])*frac
            lqs.append(lqSP)
            Slq = (newDistCtrs[i] - S0) / (p0 - p) * lqSP * Nlqs
            Slqs.append(Slq)

    else:
        print("Right on SP?...seems unlikely...")

    p0 = p
    S0 = newDistCtrs[i]
    lq0 = lq_hat[i]


#option 2, calculate fx analytically using fx function (replaces Slqs with lqAway)
Bpr = ep.BRFunc.ev(R[SPidxs],Z[SPidxs])
Bpz = ep.BZFunc.ev(R[SPidxs],Z[SPidxs])
# Get R and Z vectors at the midplane
R_omp_sol = ep.g['lcfs'][:,0].max()
# Evaluate B at outboard midplane
Bp_omp = ep.BpFunc.ev(R_omp_sol,0.0)
theta = np.zeros((Brz[SPidxs].shape))
theta[:,0] = Bpr / Bp[SPidxs]
theta[:,1] = Bpz / Bp[SPidxs]
thetadotn =  np.multiply(theta, newNorms2D[SPidxs]).sum(1)
#flux expansion at each target location
fx = (Bp_omp * R_omp_sol) / (Bp[SPidxs] * R[SPidxs]) * (1.0 / np.abs(thetadotn))
lqAway = Nlqs*fx*lq


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

    for i,S in enumerate(SPs):
        fig.add_vline(x=S, line_width=4, line_dash="dash")
        fig.add_vline(x=S+Slqs[i], line_width=4, line_dash="dot")
        fig.add_vline(x=S-Slqs[i], line_width=4, line_dash="dot")

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
    for i,S in enumerate(SPs):
        fig.add_vline(x=S, line_width=4, line_dash="dash")
        fig.add_vline(x=S+Slqs[i], line_width=4, line_dash="dot")
        fig.add_vline(x=S-Slqs[i], line_width=4, line_dash="dot")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Distance Along Contour [m]")
    fig.update_yaxes(title="Angle of Incidence [degrees]")
    fig.update_xaxes(range=[minS-0.03, maxS+0.03])
    fig.update_yaxes(range=[-8, 8])
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

    for i,S in enumerate(SPs):
        fig.add_vline(x=S, line_width=4, line_dash="dash")
        fig.add_vline(x=S+lqAway[i], line_width=4, line_dash="dot")
        fig.add_vline(x=S-lqAway[i], line_width=4, line_dash="dot")

    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Distance Along Contour [m]")
    fig.update_yaxes(title="Angle of Incidence [degrees]")
    fig.update_xaxes(range=[minS-0.03, maxS+0.03])
    fig.update_yaxes(range=[-8, 8])
    fig.show()

print("test")
print(Slqs)
print(lqAway)
print("test")
