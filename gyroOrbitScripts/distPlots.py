#distribution calculator
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import trapz


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

fig = go.Figure()
#velocity
#fig.add_trace(go.Scatter(x=v_perp, y=f_r, name="TEST", line=dict(color='royalblue', width=4, dash='solid'),
#                         mode='lines', marker_symbol='circle', marker_size=14))
#fig.add_vline(x=v_perp0, line=dict(color='green', dash='dot', width=6))
#radius
fig.add_trace(go.Scatter(x=r_gyro, y=f_v, name="TEST", line=dict(color='royalblue', width=4, dash='solid'),
                         mode='lines', marker_symbol='circle', marker_size=14))
fig.add_vline(x=r0, line=dict(color='green', dash='dot', width=6))
fig.add_vline(x=r1, line=dict(color='magenta', dash='dot', width=6))
fig.add_vline(x=r1/2, line=dict(color='magenta', dash='dot', width=6))

fig.update_layout(
    title="f(r) for T=15keV",
    xaxis_title="Gyroradius [m]",
    yaxis_title="",
    font=dict(
        family="Arial",
        size=30,
        color="Black"
        ),
    )


#Velocity stuff
#print(np.average(f_v))
#print(np.sum(f_v)*dv)
#print(trapz(f_v,v_perp))
#print(v_perp0)
#radius stuff
print(r0)
print(r1)
fig.show()
