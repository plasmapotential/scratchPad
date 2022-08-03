#compareDists.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from plotly.subplots import make_subplots

#compare velocity distribution functions and sampling
#reviewer #2 method
#deuterium
ionMassAMU = 2.0
#electrons
electronMassAMU = 1.0 / 1836
T0 = 100 #eV
kg2eV = 5.609e35 #1kg = 5.609e35 eV/c^2
eV2K = 1.160e4 #1ev=1.160e4 K
AMU = 931.494e6 #ev/c^2
kB = 8.617e-5 #ev/K
e = 1.602e-19 # C
c = 299792458 #m/s
m = ionMassAMU * AMU
Z=1 #assuming isotopes of hydrogen here


N_vSlice=5
vSlices = np.ones((N_vSlice))*np.nan
energySlices = np.zeros((N_vSlice))
energyIntegrals = np.zeros((N_vSlice))
energyFracs = np.zeros((N_vSlice))
vBounds = np.zeros((N_vSlice+1))
#get average velocity for each temperature point
vThermal = np.sqrt(2.0*T0/(m/c**2))
#set upper bound of v*f(v) (note that this cuts off high energy particles)
vMax = 5 * vThermal
#get 100 points to initialize functional form of f(v) (note this is a 2D matrix cause vMax is 2D)
v = np.linspace(0,vMax,10000).T

#generate the (here maxwellian) velocity vector PDF
#pdf = lambda x: (self.mass_eV/self.c**2) / (self.T0[i]) * np.exp(-(self.mass_eV/self.c**2 * x**2) / (2*self.T0[i]) )
pdf = lambda x: ( (m/c**2) / (2 * np.pi * T0) )**(3.0/2.0) * np.exp(-(m/c**2 * x**2) / (2*T0) )
#speed pdf (integrate over solid angle)
v_pdf = 4*np.pi * v**2 * pdf(v)
#generate the CDF
v_cdf = np.cumsum(v_pdf[1:])*np.diff(v)
v_cdf = np.insert(v_cdf, 0, 0)
#create bspline interpolators for the cdf and cdf inverse
inverseCDF = interp1d(v_cdf, v, kind='linear')
forwardCDF = interp1d(v, v_cdf, kind='linear')
#CDF location of vSlices and bin boundaries
cdfBounds = np.linspace(0,v_cdf[-1],N_vSlice+1)
#CDF location of velocity bin bounds omitting 0 and 1
#old method does not make vSlices truly bin centers
#cdfBounds = np.linspace(0,1,self.N_vSlice+1)[1:-1]
#new method spaces bins uniformly, then makes vSlices center of these bins in CDF space
cdfSlices = np.diff(cdfBounds)/2.0 + cdfBounds[:-1]
#vSlices are Maxwellian distribution sample locations (@ bin centers)
vSlices = inverseCDF(cdfSlices)
vSlices_pdf = 4*np.pi * vSlices**2 * pdf(vSlices)
vBounds = inverseCDF(cdfBounds)
vBounds = vBounds

f_E = lambda x: 2 * np.sqrt(x / np.pi) * (T0)**(-3.0/2.0) * np.exp(-x / T0)
##energy slices that correspond to velocity slices
#energySlices = f_E(0.5 * (m/c**2) * vSlices**2)
##energy integrals
#for j in range(N_vSlice):
#    Elo = 0.5 * (m/c**2) * vBounds[j]**2
#    Ehi = 0.5 * (m/c**2) * vBounds[j+1]**2
#    energyIntegrals[j] = integrate.quad(f_E, Elo, Ehi)[0]
#energyTotal = energyIntegrals.sum()

f_E2 = lambda x: x**2 * f_E(x)
#energy slices that correspond to velocity slices
energySlices = f_E2(0.5 * (m/c**2) * vSlices**2)
#energy integrals
for j in range(N_vSlice):
    Elo = 0.5 * (m/c**2) * vBounds[j]**2
    Ehi = 0.5 * (m/c**2) * vBounds[j+1]**2
    energyIntegrals[j] = integrate.quad(f_E2, Elo, Ehi)[0]
energyTotal = energyIntegrals.sum()



print(energyIntegrals / energyTotal)
print(np.sum(energyIntegrals / energyTotal))

#energy fractions
for j in range(N_vSlice):
    energyFracs[j] = energyIntegrals[j] / energyTotal


fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, x_title="Velocity [m/s]",)
#PDF Traces
fig.add_trace(go.Scatter(x=v, y=v_pdf,
                    mode='lines',
                    line={'width':6},
                    name='Maxwellian PDF',
                    showlegend=True),
                    row=1, col=1)
fig.add_trace(go.Scatter(x=vSlices, y=vSlices_pdf,
                    mode='markers',
                    marker={'symbol':"circle", 'size':16, "color":"black"},
                    name='PDF Slices'),
                    row=1, col=1)
#CDF Traces
fig.add_trace(go.Scatter(x=v, y=v_cdf,
                    mode='lines',
                    line_dash="dashdot",
                    line={'width':6, 'color':'purple'},
                    name='Maxwellian CDF',
                    showlegend=True),
                    row=2, col=1)
fig.add_trace(go.Scatter(x=vSlices, y=cdfSlices,
                    mode='markers',
                    marker={'symbol':"cross", 'size':16, "color":"darkgreen"},
                    name='CDF Slices'),
                    row=2, col=1)
    #Bin boundaries (vertical for PDF, horizontal for CDF)
for i in range(len(cdfBounds)):
    #fig.add_vline(dict(x=vBounds[i], line_width=3, line_dash="dash", line_color="green"), row=2, col=1)
    fig.add_shape(dict(type="line", x0=vBounds[i], x1=vBounds[i], y0=0, y1=max(v_pdf)*1.05, line_color="firebrick", line_width=3, line_dash="dot"), row=1, col=1)
    fig.add_shape(dict(type="line", x0=0, x1=vMax, y0=cdfBounds[i], y1=cdfBounds[i], line_color="firebrick", line_width=3, line_dash="dot"), row=2, col=1)
fig.update_layout(
    title="Velocity PDF vs. CDF",
#        yaxis= dict(showticklabels=True),
#        yaxis_range=[0,1.05*max(v_pdf)],
#        xaxis_range=[0,vMax],
    font=dict(size=18),
    )
fig.update_yaxes(range=[-1e-6,max(v_pdf)*1.05], showticklabels=False, row=1,col=1)
fig.update_yaxes(range=[-0.05,1.05], showticklabels=True, row=2,col=1)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.98,
    xanchor="right",
    x=0.95,
    font=dict(size=14),
    ),
    )
fig.show()
