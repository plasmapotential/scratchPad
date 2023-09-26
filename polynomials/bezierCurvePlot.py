import numpy as np
from scipy.special import comb

import plotly.graph_objects as go

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals





points = np.array([ [0,6], [10, 6.0], [40, 6.0], [67.5, 6], [67.5, 5],[67.5,0] ])
nPoints = len(points)
xpoints = [p[0] for p in points]
ypoints = [p[1] for p in points]

xvals, yvals = bezier_curve(points, nTimes=1000)

fig = go.Figure()
fig.add_trace(go.Scatter(x=xvals, y=yvals))
fig.add_trace(go.Scatter(x=xpoints, y=ypoints, mode='markers'))
fig.show()

