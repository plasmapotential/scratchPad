#aoiAlongLim.py
#Description:   calculate B field angle of incidence (AOI) along the rlim,zlim
#               contour from gfile
#Date:          20220621
#engineer:      T Looby
import sys
import os
import numpy as np
import scipy.interpolate as scinter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shutil

#you need equilParams_class to run this script
#if you dont have it you can run in the HEAT docker container and point EFITpath
#to /root/source
EFITpath = '/home/tlooby/source'
sys.path.append(EFITpath)
import EFIT.equilParams_class as EP
from scipy.interpolate import interp1d

#divertor points
#v2a
#points = [[1.41,	-1.137],
#          [1.287,	-1.224],
#          [1.285,	-1.235],
#          [1.45,	-1.184],
#          [1.448,	-1.184],
#          [1.58,	-1.303],
#          [1.578,	-1.303],
#          [1.82,	-1.6],
#          [1.82,	-1.6],
#          [1.73,	-1.41],
#          [1.73,	-1.41],
#          [1.68,	-1.295]]
#
#SPts = [
#        1.18856079,
#        1.33959099,
#        1.40659099,
#        1.57929304,
#        1.58129304,
#        1.7717923,
#        1.7737923,
#        2.15690194,
#        2.15690194,
#        2.53690194,
#        2.5369019,
#        2.6623013,
#        ]

##v2y
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

#v3b points for overlays
points = ([
            [1.38510000,-1.11540000],
            [1.29490000,-1.22360000],
            [1.32000000,-1.21000000],
            [1.44070000,-1.20900000],
            [1.44070000,-1.21000000],
            #[1.50930000,-1.20900000],
            [1.57080000,-1.29640000],
            [1.57000000,-1.29700000],
            [1.72000000,-1.51000000],
            [1.72000000,-1.51000000],
            #[1.72000000,-1.57500000],
            #[1.84000000,-1.57500000],
            #[1.84000000,-1.38000000],
            [1.69500000,-1.38000000],
            [1.69500000,-1.38000000],
            [1.65850000,-1.21770000]
        ])

SPts = ([
        1.16415304,
        1.30501922,
        1.33356690,
        1.450642,
        1.45888,
        #1.520267,
        1.627867,
        1.632236,
        1.890172,
        1.894041,
        2.415766,
        2.419635,
        2.580192,
        ])
#tile lower bounds S for overlay rectangles
tileLos = np.array(
    [
        1.16415304,
        1.33356690,
        1.45527104,
        1.63174746,
        1.89226425,
        2.41726425,
    ]
)
#tile upper bounds S for overlay rectangles
tileHis = np.array(
    [
        1.30501922,
        1.45427104,
        1.63074746,
        1.89226425,
        2.41726425,
        2.58361791,
    ]
)



#Resolution in S direction
numS=1000
#Minimum S.  S=0 is midplane [m]
#minS = 0.0
#S greater than actual s maximum defaults to s[-1] [m]
#end of T4:
#maxS=1.89
#end of divertor:

#for entire lower divertor
minS=1.1
maxS= 2.6

#for inner divertor
#minS=1.1
#maxS= 1.9

#for T1:
#minS = 1.164
#maxS = 1.304

#for T2:
#minS = 1.335
#maxS = 1.451

#for T3:
#minS = 1.455
#maxS = 1.63

#for T4:
#minS = 1.632
#maxS = 1.893

#for T5B:
#minS = 1.955
#maxS = 2.076

#for T6:
minS = 2.416
maxS = 2.584

#tile index for plotting single tile green overlay
Tidx = 4

#masks
#field mode (total vs poloidal)
fieldMode = 'pol'
#only plot section of RZ contour of wall
sectionMask = True
#interpolate the wall points to get higher resolution AOI
interpolateMask = True
#plot the wall contour in an R,Z plot
plotMaskContour = False
#overlay strike points and lq widths
plotSP = True
#plot AOI over section of RZ wall contour with no PFC tile overlays
plotMaskAOI = False
#only plot a single tile (requires changing S min/max)
plotMaskSingleT = False
#plot AOI over section with PFC overlays
plotMaskAllT = False
#plot mins and maxes for all tiles for all timesteps
minMaxMask = True
#plot mins and maxes at the strike points for all timesteps
AOIatSP = False

#output CSV file
minMaxCSV = '/home/tlooby/projects/EQ_devon/output/minMax.csv'

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

#geqdsk files
#gPath = '/home/tom/work/CFS/GEQDSKs/sweep7_v2y/'
#gPath = '/home/tom/HEATruns/SPARC/sweep7_T5/originalGEQDSKs/'
gPath = '/home/tlooby/projects/EQ_devon/tmp/'
gNames = [f.name for f in os.scandir(gPath)]
gNames.sort()

fig = go.Figure()
#matrix for postProcessing
AOIarray = []
AOI_SParray = []
for gIdx,g in enumerate(gNames):
    #copy file to tmp location with new name so that EP class can read it
    gRenamed = '/home/tlooby/projects/EQ_devon/output/g000001.00001'
    shutil.copyfile(gPath+g, gRenamed)
    #load gfile
    ep = EP.equilParams(gRenamed)
    data, idx = np.unique(ep.g['wall'], axis=0, return_index=True)
    rawdata = data[np.argsort(idx)]


    #close the contour (if necessary)
    if np.all(rawdata[-1] != rawdata[0]):
        rawdata = np.vstack([rawdata, rawdata[0]])

    # Call distance function to return distance along S
    dist = distance(rawdata)

    #get center points for entire contour
    ctrs = centers(rawdata)
    distCtrs = distance(ctrs)

    # normal at each surface for entire contour
    norms = normals(rawdata)
    norms2D = np.delete(norms, 1, axis=1)

    #interpolate RZ wall contour if flag is true
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
        interpolated_points = np.round(arr[np.argsort(idx)], 8)
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

    #only use user specified section of the RZ wall contour
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
    #for poloidal field
    if fieldMode == 'pol':
        print("Running in poloidal field mode!")
        Brz[:,0] /= Bp
        Brz[:,1] /= Bp
    #for toroidal field
    else:
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

    #calculate angle of incidence
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

    try:
        f = scinter.UnivariateSpline(psi_omp, R_omp, s = 0, ext = 'const')
    except:
        print("ERROR!  psi_omp wasnt monotonic.  trying to truncate...")
        f = scinter.UnivariateSpline(psi_omp[5:], R_omp[5:], s = 0, ext = 'const')

    R_mapped = f(psiN)
    Z_mapped = np.zeros(len(R_mapped))
    Bp_mapped = ep.BpFunc.ev(R_mapped,Z_mapped)
    # Gradient and coordinate transformation between xyz and flux
    gradPsi = Bp_mapped*R_mapped
    xfm = gradPsi / deltaPsi
    # Decay width at target mapped to flux coordinates
    lq_hat = lq * xfm

    #calculate heat flux width and flux expansion
    #option 1: calculates flux expansion (fx) via flux mapping
    p0 = psiN[0]
    S0 = newDistCtrs[0]
    lq0 = lq_hat[0]
    SPs = []
    SPidxs = []
    lqs = []
    Nlqs = 1.0
    Slqs = []
    R_SP = []
    Z_SP = []
    for i,p in enumerate(psiN):
        if p0 < 1.0:
            if p > 1.0:
                frac = (1.0 - p0) / (p - p0)
                #S of strike point
                S = (newDistCtrs[i] - S0)*frac + S0
                SPs.append(S)
                SPidxs.append(i)
                #R,Z if strike point
                R_SP.append((R[i]-R[i-1])*frac + R[i-1])
                Z_SP.append((Z[i]-Z[i-1])*frac + Z[i-1])
                #local lambda_q at SP
                lqSP = (lq_hat[i] - lq0)*frac + lq0
                lqs.append(lqSP)
                #calculate distance in S of Nlq*lambda_q from SP
                Slq = (newDistCtrs[i] - S0) / (p - p0) * lqSP * Nlqs
                Slqs.append(Slq)

        elif p0 > 1.0:
            if p < 1.0:
                frac = (p0 - 1.0) / (p0 - p)
                #S of strike point
                S = (newDistCtrs[i] - S0)*frac + S0
                SPs.append(S)
                SPidxs.append(i)
                #R,Z if strike point
                R_SP.append((R[i]-R[i-1])*frac + R[i-1])
                Z_SP.append((Z[i]-Z[i-1])*frac + Z[i-1])
                #local lambda_q at SP
                lqSP = lq0 - (lq0 - lq_hat[i])*frac
                lqs.append(lqSP)
                #calculate distance in S of Nlq*lambda_q from SP
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
#    print("R          Z          S")
    for i,pt in enumerate(points):
        test0 = np.array(pt[0]==interpolated_points[:,0])
        test1 = np.array(pt[1]==interpolated_points[:,1])
        test = np.logical_and(test0,test1)
        #test = pt in interpolated_points
        iloc = np.where(test==True)[0][0]
        ptIdxs.append(iloc)
        line = "{:0.8f} {:0.8f} {:0.8f}".format(interpolated_points[iloc,0], interpolated_points[iloc,1], newdist[iloc])
#        print(line)

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
#    print(outputMat)
    AOIarray.append(outputMat)


    #plot the PFC RZ contour with Region of Interest overlaid
    if plotMaskContour is True:
        rawdata = np.vstack([rawdata, rawdata[0]])
        print("Plotting RZ contour")
        colorRZ = px.colors.qualitative.Plotly[gIdx]
        colorSec = px.colors.qualitative.Prism[gIdx]
        colorSP = px.colors.qualitative.Dark2[gIdx]
        #RZ contour
        fig.add_trace(go.Scatter(x=rawdata[:,0], y=rawdata[:,1], name="Entire Contour",
                                 line=dict(color=colorRZ)))
        #overlay section
        fig.add_trace(go.Scatter(x=interpolated_points[idxDist,0],
                             y=interpolated_points[idxDist,1],
                             opacity=0.6,
                             name="Region of Interest", mode='lines+markers',
                             line=dict(color=colorSec)) )

        #overlay strike points
        if plotSP == True:
            for j,S in enumerate(SPs):
                if S > minS and S < maxS:
                    idx = (np.abs(newdist - S)).argmin()
                    n='Strike Point {:d}'.format(gIdx)
                    if j==0: show=True
                    else: show=False
                    fig.add_trace(go.Scatter(x=[interpolated_points[idx,0]],
                                    y=[interpolated_points[idx,1]],
                                     legendgroup=n,
                                     name=n,
                                     showlegend=show,
                                     mode='markers',
                                     marker_symbol='x',
                                     marker=dict(color=colorSP,size=15)
                                    ))

        fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
        # set showlegend property by name of trace
        for trace in fig['data']:
            if(trace['name'] == 'SP'): trace['showlegend'] = False
        fig.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
                ))
        fig.update_xaxes(title="R [m]")
        fig.update_yaxes(title="Z [m]")
        if gIdx == len(gNames)-1:
            fig.show()

    #generic AOI plot.  No PFC tile overlays
    if plotMaskAOI is True:
        fig.add_trace(go.Scatter(x=newDistCtrs[idxDistCtrs], y=AOI[idxDistCtrs], mode='lines+markers', name=gNames[gIdx].split('.')[0]))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="Distance Along Contour [m]")
        if fieldMode=='pol':
            fig.update_yaxes(title="Poloidal Angle of Incidence [degrees]")
        else:
            fig.update_yaxes(title="Angle of Incidence [degrees]")

        #overlay SP and +/- Nlqs away
        if plotSP == True:
            for i,S in enumerate(SPs):
                if S < minS or S > maxS:
                    pass
                else:
                    #vertical lines at SP and +/- lq
                    #fig.add_vline(x=S, line_width=4, line_dash="dash")
                    #fig.add_vline(x=S+lqAway[i], line_width=4, line_dash="dot")
                    #fig.add_vline(x=S-lqAway[i], line_width=4, line_dash="dot")
                    #x and o at SP and +/- lq
                    interpSP = interp1d(newDistCtrs, AOI, kind='slinear', axis=0)
                    fig.add_trace(go.Scatter(x=[S], y=[interpSP(S)], mode="markers", marker_symbol='x', marker=dict(color='black', size=15), name="SP"))
                    fig.add_trace(go.Scatter(x=[S+lqAway[i]], y=[interpSP(S+lqAway[i])], mode="markers", marker_symbol='diamond', marker=dict(color='black', size=15), name="SP"))
                    fig.add_trace(go.Scatter(x=[S-lqAway[i]], y=[interpSP(S-lqAway[i])], mode="markers", marker_symbol='diamond', marker=dict(color='black', size=15), name="SP"))


        for trace in fig['data']:
            if(trace['name'] == 'SP'): trace['showlegend'] = False

        fig.update_layout(showlegend=True)
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.
            ))


        fig.show()

    #plot for a single tile (Tidx above).  Set Smin Smax for tile first
    if plotMaskSingleT == True:
        startIdxs = SPts[0::2]
        endIdxs = SPts[1::2]
        fig.add_trace(go.Scatter(x=newDistCtrs[idxDistCtrs], y=AOI[idxDistCtrs], mode='lines+markers'))
        fig.add_vrect(x0=startIdxs[Tidx], x1=endIdxs[Tidx],
                annotation_text="Tile of Interest", annotation_position="top left",
                fillcolor="green", opacity=0.25, line_width=0)
        #overlay SP and +/- Nlqs away
        if plotSP == True:
            for i,S in enumerate(SPs):
                fig.add_vline(x=S, line_width=4, line_dash="dash")
                fig.add_vline(x=S+Slqs[i], line_width=4, line_dash="dot")
                fig.add_vline(x=S-Slqs[i], line_width=4, line_dash="dot")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="Distance Along Contour [m]")
        if fieldMode=='pol':
            fig.update_yaxes(title="Poloidal Angle of Incidence [degrees]")
        else:
            fig.update_yaxes(title="Angle of Incidence [degrees]")
        fig.update_xaxes(range=[minS-0.03, maxS+0.03])
        fig.update_yaxes(range=[-8, 8])
        fig.show()

    #plot for all tiles.  set Smin and Smax to entire divertor first
    if plotMaskAllT == True:
        startIdxs = SPts[0::2]
        endIdxs = SPts[1::2]
        fig.add_trace(go.Scatter(x=newDistCtrs[idxDistCtrs], y=AOI[idxDistCtrs], mode='lines+markers', name=gNames[gIdx].split('.')[0]))
        #overlay colored boxes for each tile
        if gIdx == 0:
            for Tidx in range(6):
            #    fig.add_vrect(x0=startIdxs[Tidx], x1=endIdxs[Tidx],
            #        annotation_text="T{:d}".format(Tidx+1), annotation_position="top left",
            #        fillcolor=px.colors.qualitative.Vivid[Tidx], opacity=0.25, line_width=0)
                fig.add_vrect(x0=tileLos[Tidx], x1=tileHis[Tidx],
                    annotation_text="T{:d}".format(Tidx+1), annotation_position="top left",
                    fillcolor=px.colors.qualitative.Vivid[Tidx], opacity=0.25, line_width=0)

        #overlay SP and +/- Nlqs away
        if plotSP == True:
            for i,S in enumerate(SPs):
                fig.add_vline(x=S, line_width=4, line_dash="dash")
                fig.add_vline(x=S+lqAway[i], line_width=4, line_dash="dot")
                fig.add_vline(x=S-lqAway[i], line_width=4, line_dash="dot")

        fig.update_layout(showlegend=True)
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
            ))
        fig.update_xaxes(title="Distance Along Contour [m]")
        if fieldMode=='pol':
            fig.update_yaxes(title="Poloidal Angle of Incidence [degrees]")
        else:
            fig.update_yaxes(title="Angle of Incidence [degrees]")
        fig.update_xaxes(range=[minS-0.03, maxS+0.03])
        fig.update_yaxes(range=[-8, 8])
        fig.show()

    #print("Nlqs*lq via flux mapping:")
    #print(Slqs)
    #print("Nlqs*lq via fx equation:")
    #print(lqAway)


    if AOIatSP == True:
        test = np.logical_and(np.array(SPs)>minS,np.array(SPs)<maxS )
        use = np.where(test==True)[0]
        R_SP = np.array(R_SP)[use]
        Z_SP = np.array(Z_SP)[use]
        S_SP = np.array(SPs)[use]

        #B field from MHD Equilibrium
        Brz_SP = np.zeros((len(use),2))
        Brz_SP[:,0] = ep.BRFunc.ev(R_SP,Z_SP)
        Brz_SP[:,1] = ep.BZFunc.ev(R_SP,Z_SP)
        Bp = np.sqrt(Brz_SP[:,0]**2 + Brz_SP[:,1]**2)
        Bt = ep.BtFunc.ev(R_SP,Z_SP)
        B = np.sqrt(Brz_SP[:,0]**2 + Brz_SP[:,1]**2 + Bt**2)
        Brz_SP[:,0] /= B
        Brz_SP[:,1] /= B

        #approximate norms by nearest point in normal vectors
        norms_SP = newNorms2D[SPidxs][use]

        #calculate angle of incidence
        bdotn_SP = np.multiply(Brz_SP, norms_SP).sum(1)
        AOI_SP = np.degrees(np.arcsin(bdotn_SP))
        #print("timestep: {:d}".format(gIdx))
        #print(R_SP)
        #print(Z_SP)
        #print(np.array(SPs)[use])
        #print(AOI_SP)
        for k in range(len(use)):
            row = [gIdx, R_SP[k], Z_SP[k], S_SP[k], AOI_SP[k]]
            AOI_SParray.append(row)
        print(np.array(AOI_SParray))

if AOIatSP == True:
    print("=== Postprocessing AOIs ===")
    AOI_SParray = np.array(AOI_SParray)
    #after all timesteps have been processed, create an additional plot of min,max AOI
    #for each tile at all timesteps
    xLabels = ['T1', 'T2', 'T4',]
    x = np.arange(3)
    symbols = ['x', 'star', 'diamond', 'bowtie', 'hourglass', 'circle-x', 'hexagram', 'square', 'cross', 'triangle-up', 'triangle-down', 'pentagon' ]
    fig1 = go.Figure()
    fig2 = go.Figure()
    tile4_AOIs = []
    for i,row in enumerate(AOI_SParray):
        gIdx = int(row[0])
        if i%2 == 0:
            fig1.add_trace(go.Scatter(x=[0], y=[np.abs(row[4])], mode='markers', marker_symbol=symbols[gIdx], marker=dict(color='#2CA02C',size=15), name='T1 @ {:d}'.format(gIdx)))
            tile4_AOIs.append(row[3])
        else:
            fig1.add_trace(go.Scatter(x=[1], y=[np.abs(row[4])], mode='markers', marker_symbol=symbols[gIdx], marker=dict(color='#1F77B4',size=15), name='T4 @ {:d}'.format(gIdx)))


    fig1.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
        ),
        xaxis = dict(tickmode = 'array', tickvals = x, ticktext=xLabels),
        font=dict(
            size=20,
        )
        )
    fig1.update_xaxes(title="Tile Number", range=[-0.5, 1.5])
    fig1.update_yaxes(title="Angle of Incidence [degrees]")
    fig1.show()

    #change this idx to plot aoi as a function of poloidal angle on different tiles
    tileIdx = 0
    fig2.add_trace(go.Scatter(x=AOI_SParray[tileIdx::2, 3], y=np.abs(AOI_SParray[tileIdx::2, 4]), mode='lines+markers', marker_symbol=symbols[gIdx], marker=dict(color='#1F77B4',size=15), name='AOI Across T4'))
    fig2.update_xaxes(title="Poloidal Distance From Inner Midplane [m]")
    fig2.update_yaxes(title="Angle of Incidence [degrees]")
    fig2.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
        ),
        font=dict(
            size=20,
        )
        )
    fig2.show()


    print("Table of AOIs at the Strike Points")
    print(AOI_SParray)





if minMaxMask == True:
    print("=== Postprocessing AOIs ===")
    #after all timesteps have been processed, create an additional plot of min,max AOI
    #for each tile at all timesteps
    xLabels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
    x = np.arange(6)
    symbols = ['x', 'star', 'diamond', 'bowtie', 'hourglass', 'circle-x', 'hexagram' ]
    fig1 = go.Figure()
    minMaxArr = np.zeros((len(gNames), len(x)*2))
    from plotly.validators.scatter.marker import SymbolValidator
    symbols = SymbolValidator().values

    for i,arr in enumerate(AOIarray):
        #fig1.add_trace(go.Scatter(x=x, y=arr[:,0], mode='markers', marker_symbol=symbols[i], marker=dict(color='blue',size=15), name='Minimum gIdx {:d}'.format(i)))
        fig1.add_trace(go.Scatter(x=x, y=arr[:,1], mode='markers', marker_symbol=symbols[i], marker=dict(color='red',size=15), name='Maximum gIdx {:d}'.format(i)))
        minMaxArr[i,0:len(x)] = arr[:,0]
        minMaxArr[i,len(x):] = arr[:,1]

    fig1.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
        ),
        xaxis = dict(tickmode = 'array', tickvals = x, ticktext=xLabels),
        font=dict(
            size=20,
        )
        )
    fig1.update_xaxes(title="Tile Number")
    fig1.update_yaxes(title="Angle of Incidence [degrees]")
    # set showlegend property by name of trace
    for trace in fig1['data']: 
        trace['showlegend'] = False
    fig.update_layout(showlegend=False)
    fig1.show()

    #save csv with min/max data
    head = "T1_min, T2_min, T3_min, T4_min, T5_min, T6_min, T1_max, T2_max, T3_max, T4_max, T5_max, T6_max"
    np.savetxt(minMaxCSV, minMaxArr, delimiter=',',fmt='%.10f', header=head)

    reducedMinMaxArr = np.zeros((len(x), 2))
    for i in range(len(x)):
        reducedMinMaxArr[i,0] = np.min(minMaxArr[:,i])
        reducedMinMaxArr[i,1] = np.max(minMaxArr[:,len(x)+i])
    print(reducedMinMaxArr)



