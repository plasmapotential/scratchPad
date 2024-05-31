#plots the pdf and cdf for the maxwellian velocity distribution function

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.interpolate import interp1d
#constants
ionMassAMU = 2.515
T_eV = 1000.0 #eV

kg2eV = 5.609e35 #1kg = 5.609e35 eV/c^2
eV2K = 1.160e4 #1ev=1.160e4 K
AMU = 931.494e6 #ev/c^2
kB = 8.617e-5 #ev/K
e = 1.602e-19 # C
c = 299792458 #m/s
mass_eV = ionMassAMU * AMU
Z=1 #assuming isotopes of hydrogen here

vThermal = np.sqrt(2.0*T_eV/(mass_eV/c**2))
print("Thermal Velocity: {:f} [m/s]".format(vThermal))
#vThermal = np.sqrt(T_eV/(mass_eV/c**2))
LPar = 4.649 #m


#toroidal rotation velocity projected along B
v_rot_b = 0.0

#maximum V for plotting and integrals
vMax = 3 * vThermal + v_rot_b

#number of velocity samples
N_vS = 5


def gaussian1D(B, x, x0=0.0):
    """
    returns a 1D gaussian
    """
    g = np.sqrt(B/np.pi) * np.exp(-B*(x-x0)**2)
    return g

#1D velocity
beta = (mass_eV/c**2) / (2.0*T_eV)
v = np.linspace(0.0, vMax, 10000).T
v_pdf = gaussian1D(beta, v, x0=v_rot_b)
#3D speed (Stangeby 2.12 or Chen 7.18)
#v_pdf = 4*np.pi * (B/np.pi)**(3.0/2.0) * v**2 * np.exp(-B*v**2)
#generate the CDF
v_cdf = np.cumsum(v_pdf[1:])*np.diff(v)
v_cdf = np.insert(v_cdf, 0, 0)
#create bspline interpolators for the cdf and cdf inverse
inverseCDF = interp1d(v_cdf, v, kind='linear')
forwardCDF = interp1d(v, v_cdf, kind='linear')
#CDF location of vSlices and bin boundaries
cdfBounds = np.linspace(0,v_cdf[-1],N_vS+1)
#space bins uniformly, then makes vSlices center of these bins in CDF space
cdfSlices = np.diff(cdfBounds)/2.0 + cdfBounds[:-1]
#vSlices are Maxwellian distribution sample locations (@ bin centers)
vSlices = inverseCDF(cdfSlices)
vBounds = inverseCDF(cdfBounds)

vSlices_pdf = gaussian1D(beta, vSlices, x0=v_rot_b)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,)
fig1 = go.Figure()
fig2 = go.Figure()

fig1.add_trace(go.Scatter(x=v, y=v_pdf,
                    mode='lines',
                    line={'width':6},
                    name='Maxwellian PDF'),
                )

fig1.add_trace(go.Scatter(x=vSlices, y=vSlices_pdf,
                    mode='markers',
                    marker={'symbol':"square", 'size':16},
                    name='Slices'),
                )


fig2.add_trace(go.Scatter(x=v, y=v_cdf,
                    mode='lines',
                    line={'width':6, 'color':'black'},
                    name='Maxwellian CDF'),
                )
fig2.add_trace(go.Scatter(x=vSlices, y=cdfSlices,
                    mode='markers',
                    marker={'symbol':"square", 'size':16},
                    name='Slices'),
                )
for i in range(len(cdfBounds)):
    fig2.add_vline(x=vBounds[i], line_width=3, line_dash="dash", line_color="green")

#if you want both in single figure
for i in range(len(fig1.data)):
    fig.add_trace(fig1.data[i], row=1, col=1)
for i in range(len(fig2.data)):
    fig.add_trace(fig2.data[i], row=2, col=1)
for i in range(len(cdfBounds)):
    fig.add_vline(x=vBounds[i], line_width=3, line_dash="dash", line_color="green", row=2, col=1)


fig.update_layout(
    #title="Velocity Distributions (vSlices)",
    yaxis= dict(showticklabels=False),
    yaxis_range=[0,max(v_pdf)*1.05],
    xaxis_range=[0,vMax],
    #xaxis_title="Velocity [m/s]",
    font=dict(size=18),
    )
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.95,
    xanchor="right",
    x=0.95
    ),
    )

fig.update_xaxes(title_text="Velocity [m/s]", row=2, col=1)

fig.show()