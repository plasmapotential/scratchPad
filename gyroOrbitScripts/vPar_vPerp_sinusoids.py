import numpy as np
import plotly.graph_objects as go

ionMassAMU = 2.0
T_eV = 100 #eV
B = 2 #T

kg2eV = 5.609e35 #1kg = 5.609e35 eV/c^2
eV2K = 1.160e4 #1ev=1.160e4 K
AMU = 931.494e6 #ev/c^2
kB = 8.617e-5 #ev/K
e = 1.602e-19 # C
c = 299792458 #m/s
mass_eV = ionMassAMU * AMU
Z=1 #assuming isotopes of hydrogen here

#thermal velocity for v
vT= np.sqrt(3.0*T_eV/(mass_eV/c**2))
v = np.array([vT*0.5, vT, vT*2.0])
cases = ['1/2 vThermal', 'vThermal', '2 vThermal']

omegaGyro = Z * e * B / (mass_eV / kg2eV)
fGyro = omegaGyro/(2*np.pi)

vPhase = np.arctan(1.0/np.sqrt(2.0))
vPerp = v * np.cos(vPhase)
vPar = v * np.sin(vPhase)
rGyro = vPerp / omegaGyro
LGyro = vPar / omegaGyro

print(vPhase)
print(rGyro)
print(LGyro)
print(vT)

deltat = 1/fGyro
t = np.linspace(0,deltat, 100)
fig = go.Figure()
for i,r in enumerate(rGyro):
    x = vPar[i]*t
    y = r * np.sin(omegaGyro*t)
    fig.add_trace(go.Scatter(x=x, y=y, name=cases[i]))



fig.update_layout(
#title="Temperature Probe Time Evolution",
xaxis_title="Parallel [m]",
yaxis_title="x [m]",
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




fig.update_layout(legend=dict(
    yanchor="middle",
    y=0.9,
    xanchor="right",
    x=0.9
))


fig.show()




def maxwellian(x):
    """
    returns a maxwellian distribution with x as independent variable.  x
    must either be scalar or len(T0)
    Uses mass, T0, c, from class
    """
    pdf = (mass_eV/c**2) / (T0) * np.exp(-(mass_eV/c**2 * x**2) / (2*T0) )
    return pdf
