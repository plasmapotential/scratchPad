import numpy as np
import plotly.graph_objects as go

topPt = np.array([0.0, 6.0])
btmPt = np.array([67.5, 0.0])

coeffs = [
            [6.0, 0.0, -0.001, 0.0, -6.95e-8, 0.0, 0.0],
            [6.0, 0.0, -0.0001, 0.0,-2.671e-7, 0.0, 0.0],
            [6.0, 0.0, -0.00132, 0.0, -1e-10, 0.0, 0.0],
            [6.0, 0.0, -0.00086, 0.0, -1e-7, 0.0, 0.0],           
        ]



x = np.linspace(topPt[0],btmPt[0],1000)
xGrid = np.tile(x, (1000, 1))


c0 = topPt[1]

c2Max = -c0 / btmPt[0]**2
c4Max = -c0 / btmPt[0]**4
c2 = np.linspace(c2Max, 0, 100)
c4 = np.linspace(c4Max, 0, 100)

print(c2Max)
print(c4Max)

X,Y = np.meshgrid(c2,c4)

def poly(c0, c2, c4, x):
    return c0 + c2*x**2 + c4*x**4

def c2Fromc4(c0, y, x, c4):
    return (y-c0-c4*x**4)/x**2

def c4Fromc2(c0, y, x, c2):
    return (y-c0-c2*x**2)/x**4

z = poly(c0, X, Y, 67.5)

use = np.where(np.abs(z)>1e-1)
#z[use] = np.nan


fig = go.Figure()

fig.add_trace(go.Contour(x=c2, y=c4, z=z))

fig.add_trace(go.Scatter(x=c2, y=c4Fromc2(6, 0, 67.5, c2)))
#fig.add_trace(go.Scatter(x=c2Fromc4(6, 0, 67.5, c2), y=c4))


#fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)

fig.show()

