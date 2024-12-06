#makes a fake trace as a function of time.  heaviside functions
import numpy as np
import scipy
import plotly.graph_objects as go

sMin = -20.0
sMax = 40.0
sHat = np.linspace(sMin, sMax, 100)


trace1 = np.zeros((len(sHat)))
trace2 = np.zeros((len(sHat)))




def eichProfile(sHat, S, lq, q0, fx):
    q = q0/2.0 * np.exp((S/(2*lq))**2 - sHat/(lq*fx)) * scipy.special.erfc(S/(2*lq) - sHat/(S*fx))
    return q



lq = 0.3 
S = 0.15
fx = 10
q0 = 1.0
q1 = eichProfile(sHat, S, lq, q0, fx)

lq = 0.6 
S = 0.6
fx = 10
q0 = 1.0
q2 = eichProfile(sHat, S, lq, q0, fx)



# Plot using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=sHat, y=q1, name='Prediction'))
fig.add_trace(go.Scatter(x=sHat, y=q2, name='Experiment'))


fig.update_layout(xaxis_title='[mm]', yaxis_title='Heat Flux', font=dict(family="Arial",size=20,))
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
fig.show()