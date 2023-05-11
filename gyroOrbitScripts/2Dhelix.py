import plotly.graph_objects as go
import numpy as np


B = 2.524

#unit conversions
kg2eV = 5.609e35 #1kg = 5.609e35 eV/c^2
eV2K = 1.160e4 #1ev=1.160e4 K
AMU = 931.494e6 #ev/c^2
kB = 8.617e-5 #ev/K
e = 1.602e-19 # C
c = 299792458 #m/s
diamag = -1 #diamagnetism = -1 for ions, 1 for electrons
mass_eV = 1.0 * AMU
Z=1 #assuming isotopes of hydrogen here
omega = e * B / (mass_eV / kg2eV)
fGyro = np.abs(omega)/(2*np.pi)
TGyro = 1.0/fGyro


vP = np.array([30.0, 45.0, 60.0])
fig = go.Figure()


v = 150000*1e3 #[m/s]
theta = np.radians(5.0) #[deg]
gP = np.radians(0)
x0 = 2.0
z0 = 0.5

for phase in vP:
    vPhase = np.radians(phase)
    vPerp = v * np.cos(vPhase)
    vPar = v * np.sin(vPhase)
    t = np.linspace(0,3.0/vPar, 1000)
    r = vPerp / omega
    #calculate helix
    sPerp = r * np.cos(omega*t + gP)
    sPar = vPar * t
    trajB = np.vstack([sPar, sPerp]).T
    #transform
    xfm = np.array([[-np.cos(theta), -np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    trajXZ = np.matmul(trajB, xfm)
    fig.add_trace(go.Scatter(x=trajXZ[:,0]+x0, y=trajXZ[:,1]+z0, name='{:0.1f} degrees'.format(phase)))

#block
xB = np.array([-1.0, 0.0, 0.0, 0.5, 0.5, 2.0])
zB = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0])

fig.add_trace(go.Scatter(x=xB, y=zB, name="Geometry"))
fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
fig.update_layout(xaxis_title="x[mm]", yaxis_title="z[mm]", title=r'v = $ \vec{v} = \vec{v_{||}} + \vec{v_{\perp}} = 150km/s$',
                 font=dict(size=18),
                 titlefont=dict(size=24),
                 )
fig.update_xaxes(range=[-1.0, 2.0])
fig.update_yaxes(range=[-1.0, 1.0])
fig.show()

