import numpy as np
import plotly.graph_objects as go


#heat flux width
lq = 2.0 #[mm]

x = np.linspace(-lq, 2*lq, 1000)

def heatFluxProfile(x):
    y =  np.exp(-x/lq)
    zeroLocs = np.where(x<0.0)[0]
    y[zeroLocs] = 0.0
    return y

y = heatFluxProfile(x)

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, line={'color':"blue", 'width':10}))
fig.add_vline(x=0.0, line={'color':"red", 'width':4})
fig.add_vline(x=lq, line={'color':"black", 'dash':'dash', 'width':6})

fig.update_layout(yaxis_range=[-0.1,1.5])

fig.update_layout(
    #title="Plot Title",
    xaxis_title="Distance from LCFS [mm]",
    yaxis_title="Normalized Heat Flux",
    font=dict(
        family="Calibri",
        size=26,
    )
)


fig.show()