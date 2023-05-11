import sys
import numpy as np
EFITPath = '/home/tom/source'
HEATPath = '/home/tom/source/HEAT/github/source'
sys.path.append(EFITPath)
sys.path.append(HEATPath)
import MHDClass
import matplotlib.pyplot as plt

#gFilePath = '/home/tom/HEATtest/accelerationTests/g204118.00100'
#gFilePath = '/home/tom/HEATtest/chrisST40/v3Div/g000001.00001'
#gFilePath = '/home/tom/HEATtest/chrisST40/v3Div/g000001.00001_psiDiv2pi'
#gFilePath = '/home/tom/HEATtest/NSTXU/limiter/g204118.00113'
#gFilePath = '/home/tom/HEATtest/NSTXU/test/g204118.00100'
#gFilePath = '/home/tom/Downloads/g1210923001.01000'
gFilePath = '/home/tom/work/CFS/GEQDSKs/TSCruns/TSC-V2h01/TSC-V2h01/corrected_v2y_Ip_Bt_psi_Fpol/withTimesteps/g000001.00400'
gFilePath = '/home/tom/work/CFS/GEQDSKs/TSCruns/TSC-V2h01/TSC-V2h01/corrected_v2y_Ip_Bt_psi_Fpol/interpolated/g000001.00400'
MHD = MHDClass.setupForTerminalUse(gFile=gFilePath)
ep = MHD.ep


def nstxu_wall(oldwall=False):
    """
    returns simplified wall.  Uses two different wall versions
    """
    if oldwall:
        R = np.array([0.1851, 0.1851, 0.2794, 0.2794, 0.2979, 0.5712,
                    1.0433, 1.3192, 1.3358,
                    1.4851, 1.4791, 1.5174, 1.5313, 1.5464, 1.5608,
                    1.567, 1.5657, 1.5543, 1.5341, 1.5181, 1.4818,
                    1.4851, 1.3358, 1.3192, 1.0433,
                    0.5712, 0.2979, 0.2794, 0.2794, 0.1851, 0.1851])
        Z = np.array([0.0, 1.0081, 1.1714, 1.578, 1.6034, 1.6034,
                    1.43, 1.0397, 0.9976,
                    0.545, 0.4995, 0.306, 0.2355, 0.1586, 0.0801,
                    0.0, -0.0177, -0.1123, -0.221, -0.3026, -0.486,
                    -0.545, -0.9976, -1.0397, -1.43,
                    -1.6034, -1.6034, -1.578, -1.1714, -1.0081, 0])
    else:
      R = np.array([ 0.3147568,  0.3147568,  0.4441952,  0.4441952,  0.443484 ,
           0.443484 ,  0.6000496,  0.7672832,  0.8499856,  1.203452,  1.3192,  1.3358,  1.4851,  1.489 ,
           1.5638,  1.57  ,  1.5737,  1.575 ,  1.5737,  1.57  ,  1.5638,
           1.489 ,  1.4851,  1.3358,  1.3192,  1.203452 ,  0.8499856,  0.7672832,  0.6000496,  0.443484 ,
           0.443484 ,  0.4441952,  0.4441952,  0.3147568,  0.3147568 ])
      Z = np.array([ 0.       ,  1.0499344,  1.2899136,  1.5104872,  1.5104872,
            1.6028416,  1.6028416,  1.5367   ,  1.5367   ,  1.397508,  1.0397,  0.9976,  0.545 ,  0.49  ,
            0.1141,  0.0764,  0.0383,  0.    , -0.0383, -0.0764, -0.1141,
            -0.49  , -0.545 , -0.9976, -1.0397, -1.397508 , -1.5367   , -1.5367   , -1.6028416, -1.6028416,
            -1.5104872, -1.5104872, -1.2899136, -1.0499344,  0.])
    return R,Z




r = ep.g['R']
z = ep.g['Z']

aspectRatio =  np.abs(max(r)-min(r)) /  np.abs(max(z)-min(z))

#use = np.where(r > 0.25)[0]
use = np.arange(len(r))
psiRZ = ep.g['psiRZn']
R,Z = np.meshgrid(r[use], z)

Bt = ep.BtFunc.ev(R,Z)
Bp = ep.BpFunc.ev(R,Z)
Br = ep.B_R
Bz = ep.B_Z
B = np.sqrt(Bt**2 + Bp**2)

#dont waste contour space on xfmr coil field if its way higher than Bt0
BMax = np.max(B)
if ep.g['Bt0'] < 0:
    BMin = np.max(Bt)
else:
    BMin = np.min(Bt)

rbdry = ep.g['lcfs'][:,0]
zbdry = ep.g['lcfs'][:,1]

#rlim, zlim = nstxu_wall(oldwall=False) #FOR NSTXU
rlim = ep.g['wall'][:,0]
zlim = ep.g['wall'][:,1]

height=1000
aspect = (z.max()-z.min()) / (r.max()-r.min())
width = (1.0/aspect)*height


import plotly
import plotly.graph_objects as go
import plotly.express as px
#B mag data
#fig = go.Figure(data =
#    go.Contour(
#        z=B,
#        x=r[use], # horizontal axis
#        y=z, # vertical axis
#        colorscale='cividis',
#        contours_coloring='heatmap',
#        name='B',
#        showscale=False,
#        ncontours=200,
#        contours=dict(
#            start=BMin,
#            end=BMax,
#            ),
#    ))

#Bp
#fig = go.Figure(data =
#    go.Contour(
#        z=Bp,
#        x=r[use], # horizontal axis
#        y=z, # vertical axis
#        colorscale='cividis',
#        contours_coloring='heatmap',
#        name='Bp',
#        showscale=False,
#        ncontours=200,
#    ))

#psiRZ
fig = go.Figure(data =
    go.Contour(
        z=psiRZ,
        x=r[use], # horizontal axis
        y=z, # vertical axis
        colorscale='cividis',
        contours_coloring='heatmap',
        name='psi',
        showscale=False,
        ncontours=100,
    ))


#Wall in green
fig.add_trace(
    go.Scatter(
        x=rlim,
        y=zlim,
        mode="markers+lines",
        name="Wall",
        line=dict(
            color="#19fa1d"
                )
        )
        )

#add in a PFC surface line
#fig.add_trace(
#    go.Scatter(
#        x=[0.3946, 0.6783],
#        y=[-0.8146, -0.8911],
#        mode="markers+lines",
#        name="PFC",
#        line=dict(
#            color="#fc0317"
#                )
#        )
#        )


#white lines around separatrix
levelsAtLCFS = np.linspace(0.95,1.05,15)
CS = plt.contourf(R,Z,psiRZ,levelsAtLCFS,cmap=plt.cm.cividis)
for i in range(len(levelsAtLCFS)):
    levelsCS = plt.contour(R,Z,psiRZ,levels=[levelsAtLCFS[i]])
    for j in range(len(levelsCS.allsegs[0])):
        rCS = levelsCS.allsegs[0][j][:,0]
        zCS = levelsCS.allsegs[0][j][:,1]
        fig.add_trace(
            go.Scatter(
                x=rCS,
                y=zCS,
                mode="lines",
                line=dict(
                    color="white",
                    width=1,
                    dash='dot',
                        )
            )
        )

#Seperatrix in red.  Sometimes this fails if psi is negative
#so we try and except.
#if try fails, just plot using rbdry,zbdry from gfile
logFile = False
try:
    CS = plt.contourf(R,Z,psiRZ,levels,cmap=plt.cm.cividis)
    lcfsCS = plt.contour(CS, levels = [1.0])
    for i in range(len(lcfsCS.allsegs[0])):
        rlcfs = lcfsCS.allsegs[0][i][:,0]
        zlcfs = lcfsCS.allsegs[0][i][:,1]
        fig.add_trace(
            go.Scatter(
                x=rlcfs,
                y=zlcfs,
                mode="lines",
                name="LCFS",
                line=dict(
                    color="red",
                    width=4,
                        )
                )
                )
except:
    print("Could not create contour plot.  Psi levels must be increasing.")
    print("Try flipping psi sign and replotting.")
    print("plotting rbdry, zbdry from gfile (not contour)")
    if logFile is True:
        log.info("Could not create contour plot.  Psi levels must be increasing.")
        log.info("Try flipping psi sign and replotting.")
        log.info("plotting rbdry, zbdry from gfile (not contour)")
    fig.add_trace(
        go.Scatter(
            x=rbdry,
            y=zbdry,
            mode="lines",
            name="LCFS",
            line=dict(
                color="red",
                width=4,
                    )
            )
            )



###version with subplots
#fig = plotly.subplots.make_subplots(rows=1, cols=1)
##psiRZ
#fig.add_trace(
#    go.Contour(
#        z=psiRZ,
#        x=r[use], # horizontal axis
#        y=z, # vertical axis
#        colorscale='cividis',
#        contours_coloring='heatmap',
#        name='psi',
#        showscale=False,
#        ncontours=100,
#    ),
#    row=1,
#    col=1
#    )
print(r.min())
print(r.max())
print(z.min())
print(z.max())
print(aspect)
print(height)
print(width)
fig.update_layout(
    title="204118@1004ms",
    xaxis_title="R [m]",
    yaxis_title="Z [m]",
    xaxis_range=[r.min(),r.max()],
    yaxis=dict(scaleanchor="x", scaleratio=1),
    #autosize=True,
    autosize=False,
    width=width*1.1,
    height=height,
    #paper_bgcolor='rgba(0,0,0,0)',
    #plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    font=dict(
#            family="Courier New",
        size=26,
        #color="#dcdce3"
    ),
    margin=dict(
        l=100,
        r=100,
        b=100,
        t=100,
        pad=4
    ),
    )
fig.show()
pdfFile = '/home/tom/phd/dissertation/diss/figures/pdf/204118_00113_EQ.pdf'
#pdfFile = '/home/tom/phd/dissertation/diss/figures/pdf/test.pdf'
#fig.write_image(pdfFile)
