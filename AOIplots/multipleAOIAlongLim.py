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
numS=1000
#Minimum S.  S=0 is midplane [m]
minS=1.1
#S greater than actual s maximum defaults to s[-1] [m]
maxS=2.15
#tile index for plotting single tile green overlay
Tidx = 3

#masks
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
minMaxMask = False
#plot mins and maxes at the strike points for all timesteps
AOIatSP = True

#output CSV file
minMaxCSV = '/home/tom/HEATruns/SPARC/RZ2AOI/minMax.csv'

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
gPath = '/home/tom/HEATruns/SPARC/RZ2AOI/sparc/useThese/'
gNames = [f.name for f in os.scandir(gPath)]
gNames.sort()

fig = go.Figure()
#matrix for postProcessing
AOIarray = []
AOI_SParray = []
for gIdx,g in enumerate(gNames):
    #copy file to tmp location with new name so that EP class can read it
    gRenamed = '/home/tom/HEAT/data/tmpDir/g000001.00001'
    shutil.copyfile(gPath+g, gRenamed)
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
    f = scinter.UnivariateSpline(psi_omp, R_omp, s = 0, ext = 'const')
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
        test = np.logical_and(pt[0]==interpolated_points[:,0], pt[1]==interpolated_points[:,1])
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
        fig.add_trace(go.Scatter(x=newDistCtrs[idxDistCtrs], y=AOI[idxDistCtrs], mode='lines+markers'))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="Distance Along Contour [m]")
        fig.update_yaxes(title="Angle of Incidence [degrees]")
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
        fig.update_yaxes(title="Angle of Incidence [degrees]")
        fig.update_xaxes(range=[minS-0.03, maxS+0.03])
        fig.update_yaxes(range=[-8, 8])
        fig.show()

    #plot for all tiles.  set Smin and Smax to entire divertor first
    if plotMaskAllT == True:
        startIdxs = SPts[0::2]
        endIdxs = SPts[1::2]
        fig.add_trace(go.Scatter(x=newDistCtrs[idxDistCtrs], y=AOI[idxDistCtrs], mode='lines+markers', name='gIdx: {:d}'.format(gIdx)))
        #overlay colored boxes for each tile
        if gIdx == 0:
            for Tidx in range(6):
                fig.add_vrect(x0=startIdxs[Tidx], x1=endIdxs[Tidx],
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
    xLabels = ['T1', 'T4',]
    x = np.arange(2)
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

    tileIdx = 1
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
    for i,arr in enumerate(AOIarray):
        fig1.add_trace(go.Scatter(x=x, y=arr[:,0], mode='markers', marker_symbol=symbols[i], marker=dict(color='blue',size=15), name='Minimum gIdx {:d}'.format(i)))
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
    fig1.show()

    #save csv with min/max data
    head = "T1_min, T2_min, T3_min, T4_min, T5_min, T6_min, T1_max, T2_max, T3_max, T4_max, T5_max, T6_max"
    np.savetxt(minMaxCSV, minMaxArr, delimiter=',',fmt='%.10f', header=head)

    reducedMinMaxArr = np.zeros((len(x), 2))
    for i in range(len(x)):
        reducedMinMaxArr[i,0] = np.min(minMaxArr[:,i])
        reducedMinMaxArr[i,1] = np.max(minMaxArr[:,len(x)+i])
    print(reducedMinMaxArr)
