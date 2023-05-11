#solves the HEAT equation in 1D using impulse response / greens functions
#created via chatGPT4 (bugs fixed by tom)

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


import plotly.graph_objects as go

import numpy as np
from scipy.integrate import simps

# Green's function for a 1D semi-infinite solid
def G(x, t, x0, t0, alpha):
    if t <= t0:
        return 0
    return (1 / (2 * np.sqrt(np.pi * alpha * (t - t0)))) * np.exp(-(x - x0)**2 / (4 * alpha * (t - t0)))

# Parameters
L = 0.000001  # Length of the domain (for visualization purposes)
tMax = 1e-9
Nx = 100  # Number of spatial points
Nt = 100  # Number of time points
alpha = 40e-6  # Thermal diffusivity [m^2/s]

# Spatial and time grids
x = np.linspace(0, L, Nx)
t = np.linspace(0, tMax, Nt)

# User-defined boundary condition q(0, t) as a NumPy array
q = np.zeros(Nt)
q[1] = 1e6  # Example: a step function with amplitude 100 for the first half of the time range

# Calculate the temperature distribution T(x, t) using the convolution integral
T = np.zeros((Nx, Nt))

for i, xi in enumerate(x):
    for j, tj in enumerate(t):
        integrand = np.array([G(xi, tj, 0, t_k, alpha) * q_k for t_k, q_k in zip(t, q)])
        T[i, j] = simps(integrand, t)


# Visualize the results using a heatmap
#import plotly.express as px
#fig = px.imshow(T, x=x, y=t, labels=dict(x="Position (x)", y="Time (t)", color="Temperature (T)"), aspect="auto")
import plotly.graph_objects as go

fig = go.Figure(go.Contour(x=x, y=t, z=T.T, colorscale='Viridis', 
                           colorbar=dict(title="Temperature (T)")))
fig.update_layout(title="Temperature distribution",
                  xaxis_title="Position (x)",
                  yaxis_title="Time (t)")
fig.show()