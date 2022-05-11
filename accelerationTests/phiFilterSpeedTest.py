#plots comparing no toroidal angle filter vs toroidal angle filter

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

x = np.array([
15408310320,
25663266720,
52168175280,
84388894200,
])

N_ROI = np.array([
22628,
37688,
76612,
123930,
])

noAccROI = [
402.056393,
755.292171,
1442.013701,
2814.136635,
]
wAccROI = [
60.408144,
97.398902,
218.635687,
486.340592,
]


N_Int = np.array([
18402,
30912,
86790,
208018,
])



noAccInt = [
108.536348,
172.101073,
519.762276,
1358.904294,
]

wAccInt = [
41.543077,
46.50396,
69.647498,
120.715957,
]



fig = go.Figure()
#fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Changing ROI Resolution", "Changing Intersect Resolution"))

#Changing ROI Resolution
fig.add_trace(go.Scatter(x=N_ROI, y=noAccROI, name="Without Filter", line=dict(color='royalblue', width=4, dash='solid'),
                         mode='lines+markers', marker_symbol='circle', marker_size=16, showlegend=True),
#              row=1,
#              col=1,
#              secondary_y=False
            )
fig.add_trace(go.Scatter(x=N_ROI, y=wAccROI, name="With Filter", line=dict(color='red', width=4, dash='dot'),
                         mode='lines+markers', marker_symbol='square', marker_size=16,showlegend=True),
#           row=1,
#           col=1,
#             secondary_y=False
            )

#Changing Intersect Resolution
#fig.add_trace(go.Scatter(x=N_Int, y=noAccInt, name="Without Filter", line=dict(color='royalblue', width=4, dash='solid'),
#                         mode='lines+markers', marker_symbol='circle', marker_size=16),
##              row=2,
##              col=1,
##              secondary_y=False
#            )
#fig.add_trace(go.Scatter(x=N_Int, y=wAccInt, name="With Filter", line=dict(color='red', width=4, dash='dot'),
#                         mode='lines+markers', marker_symbol='square', marker_size=16),
##           row=2,
##           col=1,
##             secondary_y=False
#            )


fig.update_layout(
#title="Changing ROI Mesh Resolution",
#xaxis_title="",
#yaxis_title="% Error (log scale)",
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
#fig.update_xaxes(title_text="",row=1,col=1)
fig.update_xaxes(title_text="Number of ROI Mesh Elements")

fig.update_yaxes(title_text="Time [s]")
#fig.update_yaxes(title_text="# Faces", secondary_y=True)


fig.update_layout(legend=dict(
    yanchor="top",
    y=0.9,
    xanchor="left",
    x=0.1
    ),
    margin=dict(
        l=5,
        r=5,
        b=5,
        t=5,
        pad=2
    ),
#    xaxis = dict(
#        showexponent = 'all',
#        exponentformat = 'e'
#    ),
    font=dict(
        size=20,
    )
)

#fig.update_annotations(font_size=20)
fig.show()
epsFile = '/home/tom/phd/dissertation/diss/figures/phiFilterSpeedROI.eps'
#epsFile = '/home/tom/phd/dissertation/diss/figures/phiFilterSpeedInt.eps'
#fig.write_image(epsFile)
