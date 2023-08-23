import numpy as np
import plotly.graph_objects as go

topPt = np.array([0.0, 6.0])
btmPt = np.array([67.5, 0.0])

coeffs = [
            [6.0, 0.0, -0.001, 0.0, -6.95e-8, 0.0, 0.0],
            [6.0, 0.0, -0.0001, 0.0,-2.671e-7, 0.0, 0.0],
            [6.0, 0.0, -0.000001, 0.0,-2.888e-7, 0.0, 0.0],
            #[6.0, 0.0, -0.1, 0.0, -6.95e-6, 0.0, 0.0],
            #[2.43, 0.0, -0.443139, 0.0, -0.132090,  0.0, 0.0],
        ]

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
