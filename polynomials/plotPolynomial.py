#plots a polynomial
import numpy as np
import plotly.graph_objects as go

topPt = np.array([0.0, 6.0])
btmPt = np.array([5.0, 0.0])
#btmPt = np.array([67.5, 0.0])

coeffs = [
            [6.0, 0.0, -0.001, 0.0, -6.95e-8, 0.0, 0.0],
            [6.0, 0.0, -0.0001, 0.0,-2.671e-7, 0.0, 0.0],
            [6.0, 0.0, -0.00132, 0.0, -1e-10, 0.0, 0.0],
            [6.0, 0.0, -0.00086, 0.0, -1e-7, 0.0, 0.0],   
       
        ]

coeffs = [[10.0, 0.0, -0.2, 0.0]]

x = np.linspace(topPt[0],btmPt[0],100)

#domain = [topPt[0],btmPt[0]]
#window = [btmPt[1],topPt[1]]

domain = None
window = None

fig = go.Figure()
for c in coeffs:
    p = np.polynomial.Polynomial(c, domain=domain, window=window)
    fig.add_trace(go.Scatter(x=x, y=p(x)))
#fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
fig.show()

p = np.polynomial.Polynomial(coeffs[0])
print(p(topPt[0]))
print(p(btmPt[0]))
