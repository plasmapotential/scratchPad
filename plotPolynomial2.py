#plots a polynomial
import numpy as np
import plotly.graph_objects as go

topPt = np.array([0.0, 6.0])
btmPt = np.array([67.5, 0.0])
x = np.linspace(topPt[0],btmPt[0],1000)

#number of polys
N = 5

c0 = 6.0
c2Max = -c0 / btmPt[0]**2
c4Max = -c0 / btmPt[0]**4
c2 = np.linspace(c2Max, 0, N)

def c2Fromc4(c0, y, x, c4):
    return (y-c0-c4*x**4)/x**2

def c4Fromc2(c0, y, x, c2):
    return (y-c0-c2*x**2)/x**4


#use this to plot all solutions (some not zero intersecting)
c4 = np.linspace(c4Max, 0, N)
#use this to only plot zero intersection solutions
c4 = c4Fromc2(c0, 0.0, 67.5, c2)

coeffs = np.zeros((N, 5))
coeffs[:,0] = c0
coeffs[:,2] = c2
coeffs[:,4] = c4



domain = None
window = None

fig = go.Figure()
for c in coeffs:
    p = np.polynomial.Polynomial(c, domain=domain, window=window)
    fig.add_trace(go.Scatter(x=x, y=p(x)))
#fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
fig.show()
