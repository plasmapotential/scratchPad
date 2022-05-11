#====Length of Intersection Trace vs Error and Simulation Time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


df = pd.read_csv (r'/home/tom/HEAT/data/nstx_204118/001004/psiPointCloud_all.csv')
Y = np.zeros((len(df['psi'])))
fig = go.Figure()
#fig.add_trace(go.Scatter(x=df['psi'], y=Y, mode='markers', marker_size=1))
#fig.update_xaxes(showgrid=False)
#fig.update_yaxes(showgrid=False,
#                 zeroline=True, zerolinecolor='black', zerolinewidth=3,
#                 showticklabels=False)


fig.add_trace(go.Histogram(
                    x=df['psi'],
                    #marginal="box",
                    nbinsx=100,
                    opacity=0.7,
                    )
                )

fig.add_vrect(x0=0.98, x1=1.0, fillcolor="green", opacity=0.3)
fig.add_vline(x=0.99, line=dict(color='red', width=10, dash='dash'))

fig.update_layout(
    title="Counts of psi on IBDH",
    xaxis_title="psiN",
    yaxis_title="# of Points (Count)",
    font=dict(
        family="Arial",
        size=24,
        color="Black"
        ),
    )

fig.show()
