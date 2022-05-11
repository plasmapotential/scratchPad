import numpy as np
import os
import sys

import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

#default source code location (EFIT class should be here)
EFITPath = '/home/tom/source'
#append EFIT to python path
sys.path.append(EFITPath)
import EFIT.equilParams_class as EP


gfile = '/home/tom/NSTX/g204118.01004'
#gfile = '/home/tom/HEATtest/chrisST40/tomCase/g000001.00001'
ep = EP.equilParams(gfile)

r = ep.g['R']
z = ep.g['Z']
R,Z = np.meshgrid(r, z)

#Bp = ep.BpFunc.ev(R,Z)
#Bt = ep.BtFunc.ev(R,Z)
Bp = ep.Bp_2D
Bt = ep.Bt_2D


rbdry = ep.g['lcfs'][:,0]
zbdry = ep.g['lcfs'][:,1]

rlim = ep.g['wall'][:,0]
zlim = ep.g['wall'][:,1]



fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05, shared_yaxes=True,
                    subplot_titles=("Bt [T]", "Bp [T]"))

#dont waste contour space on xfmr coil field if its way higher than Bt0
BtMax = 3*ep.g['Bt0']
if ep.g['Bt0'] < 0:
    BtMin = np.max(Bt)
else:
    BtMin = np.min(Bt)

#psi data
fig.add_trace(
    go.Contour(
        z=Bt,
        x=r, # horizontal axis
        y=z, # vertical axis
        #colorscale='cividis',
        contours_coloring='heatmap',
        name='Bt',
        showscale=False,
        ncontours=30,
        contours=dict(
            start=BtMin,
            end=BtMax,
        ),
    ),
    row=1,
    col=1
)
#Wall in green
fig.add_trace(
    go.Scatter(
        x=rlim,
        y=zlim,
        mode="markers+lines",
        name="Wall",
        line=dict(
            color="#19fa1d"
                ),
        ),
    row=1,
    col=1
    )

fig.add_trace(
    go.Contour(
        z=Bp,
        x=r, # horizontal axis
        y=z, # vertical axis
        colorscale='viridis',
        contours_coloring='heatmap',
        name='Bp',
        showscale=False,
        ncontours=40,
    ),
    row=1,
    col=2
)

#Wall in green
fig.add_trace(
    go.Scatter(
        x=rlim,
        y=zlim,
        mode="markers+lines",
        name="Wall",
        line=dict(
            color="#19fa1d"
                ),
        ),
    row=1,
    col=2
    )

fig.update_layout(showlegend=False)

fig.show()
