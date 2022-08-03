#compareDists.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from plotly.subplots import make_subplots


#for comparing vPhases
N_vPhase=5
vPhases1 = np.degrees(np.linspace(0.0,np.pi/2,N_vPhase+2)[1:-1])
vPhases2 = np.degrees(np.arccos(np.linspace(1.0,-1.0,N_vPhase+2)[1:-1])/2.0)
vPhases3 = np.degrees(np.arcsin(np.sqrt(np.linspace(0.0,1.0,N_vPhase+2))))[1:-1]

b = np.linspace(0,np.pi/2,10000).T
#b = np.arcsin(np.linspace(0,1,1000)).T
pdf = lambda x: np.cos(x)*np.sin(x)*2
#pdf = lambda x: np.cos(x)
b_pdf = pdf(b)

#generate the CDF
b_cdf = np.cumsum(b_pdf[1:])*np.diff(b)
b_cdf = np.insert(b_cdf, 0, 0)

#create bspline interpolators for the cdf and cdf inverse
inverseCDF = interp1d(b_cdf, b, kind='linear')
forwardCDF = interp1d(b, b_cdf, kind='linear')
#CDF location of vSlices and bin boundaries
cdfBounds = np.linspace(0,b_cdf[-1],N_vPhase+1)
#CDF location of velocity bin bounds omitting 0 and 1
#old method does not make vSlices truly bin centers
#cdfBounds = np.linspace(0,1,self.N_vSlice+1)[1:-1]
#new method spaces bins uniformly, then makes vSlices center of these bins in CDF space
cdfSlices = np.diff(cdfBounds)/2.0 + cdfBounds[:-1]

#vSlices are Maxwellian distribution sample locations (@ bin centers)
vPhases= inverseCDF(cdfSlices)
vBounds = inverseCDF(cdfBounds)
vPhases4 = np.degrees(vPhases)


print(vPhases1)
print(vPhases3)
print(np.degrees(vPhases4))
print(np.degrees(np.arcsin(np.sqrt(np.linspace(0.0,1.0,N_vPhase+2)[1:-1]))))
print("---")
print(cdfSlices)

fracs = []
for i in range(len(cdfBounds)-1):
    fracs.append(cdfBounds[i+1] - cdfBounds[i])

print(fracs)
print(sum(fracs))

print(np.degrees(np.arcsin(np.linspace(0,1,5+2))))




fig = go.Figure()
for i in range(len(vPhases1)):
    fig.add_trace(go.Scatterpolar(
            r = np.array([0.0,1.0]),
            theta = np.array([vPhases1[i],vPhases1[i]]),
            mode = 'lines+markers',
            marker={'symbol':"square", 'size':16},
            name = '{:0.1f}'.format(vPhases1[i]),
            text=['{:0.1f}'.format(vPhases1[i])],
            textposition="bottom center",
            line_color= px.colors.qualitative.Dark2[0],
            line={'width':5},
            legendgroup='Original',
            legendgrouptitle_text='Original',
            showlegend=True,
            ))
    fig.add_trace(go.Scatterpolar(
            r = np.array([0.0,1.0]),
            theta = np.array([vPhases4[i],vPhases4[i]]),
            mode = 'lines+markers',
            marker={'symbol':"circle", 'size':16},
            name = '{:0.1f}'.format(vPhases4[i]),
            text=['{:0.1f}'.format(vPhases4[i])],
            textposition="bottom center",
            line_color= px.colors.qualitative.Dark2[2],
            line={'width':7, 'dash':'dash'},
            legendgroup='New CDF Method',
            legendgrouptitle_text='New CDF Method',
            showlegend=True,
            ))
fig.add_annotation(x=0.0, y=0.5,
            text="V||",
            font=dict(size=16),
            showarrow=False,
            )
fig.update_annotations(font_size=16)
fig.update_layout(
    title={'text': "Velocity Phase Angles",'y':0.94,'x':0.5,'xanchor': 'center','yanchor': 'top'},
    showlegend = True,
    polar = dict(
        sector = [0,90],
        radialaxis=dict(title=dict(text="V\u22A5",font=dict(size=16))),
        #angularaxis=dict(title=dict(text='$V_{||}$',font=dict(size=24)))
        ),
        )
fig.show()
