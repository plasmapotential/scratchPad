#distribution calculator
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import trapz
from scipy import interpolate

#unit conversions
kg2eV = 5.609e35 #1kg = 5.609e35 eV
eV2K = 1.160e4 #1ev=1.160e4 K

#constants
m_eV = 931.49e6 #eV/c^2
#m_kg = m / kg2eV
kB = 8.617e-5 #ev/K
B = 5 #tesla
e = 1.602e-19 # C
T0_eV = 15000 # eV
#T0_K = T0_eV * eV2K
c = 3e8 #m/s

v_perp0 = np.sqrt(T0_eV/(m_eV/c**2))
omega = e * B / (m_eV / kg2eV)
freq = omega / (2*np.pi)
r0 = v_perp0 / omega
r1 = 6.46e-3 * np.sqrt(T0_eV / 1000) / B # from Freidberg pg. 146 EQ 8.20

v_perp, dv = np.linspace(0,4*v_perp0,10000, retstep=True)

f_v = v_perp * (m_eV/c**2) / (T0_eV) * np.exp(-(m_eV/c**2 * v_perp**2) / (2*T0_eV) )
p_v = (m_eV/c**2) / (2*np.pi*T0_eV) * np.exp(-(m_eV/c**2 * v_perp**2) / (2*T0_eV) )

r_gyro = v_perp / omega

maxR =3*r0
x,y = np.meshgrid(np.linspace(-maxR,maxR,100), np.linspace(-maxR,maxR,100))
r2D = np.sqrt(x**2+y**2)
f = interpolate.interp1d(r_gyro, f_v, bounds_error=False, fill_value=0)

fig = go.Figure(data =
    go.Heatmap(
        z=f(r2D),
        x=np.linspace(-maxR,maxR,100),
        y=np.linspace(-maxR,maxR,100),
        showscale=False,
        connectgaps=True,
        zsmooth='best'
    ))


fig.update_layout(
    title="f(r) around guiding center (gc) for T=15keV, B=5T",
    xaxis_title="x - x_gc [m]",
    yaxis_title="y - y_gc [m]",
    font=dict(
        family="Arial",
        size=30,
        color="Black"
        ),
    )

fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='limegreen', width=4, dash='solid'),
                         mode='markers', marker_symbol='circle', marker_size=14))

fig.show()
