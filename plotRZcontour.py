import numpy as np
import plotly.graph_objects as go

#topPt = np.array([1.84, 1.04])
topPt = np.array([2.03, 0.86])
btmPt = np.array([2.43, 0.0])

coeffs = [
            [2.43, 0.0, -0.525412, 0.0, -0.0334444, 0.0, 0.0],
            [2.43, 0.0, -0.443139, 0.0, -0.132090,  0.0, 0.0],
            [2.43, 0.0, -0.374472, 0.0, -0.203461,  0.0, 0.0],
            [2.43, 0.0, -0.313361, 0.0, -0.262355,  0.0, 0.0],
            [2.43, 0.0, -0.247534, 0.0, -0.325026,  0.0, 0.0]
        ]


y = np.linspace(btmPt[1],topPt[1],100)

domain = [btmPt[1],topPt[1]]
window = [btmPt[1],topPt[1]]

fig = go.Figure()
for c in coeffs:
    p = np.polynomial.Polynomial(c, domain=domain, window=window)
    fig.add_trace(go.Scatter(x=p(y), y=y))
fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
fig.show()

p = np.polynomial.Polynomial(coeffs[0])
print(p(topPt[1]))
print(p(btmPt[1]))
