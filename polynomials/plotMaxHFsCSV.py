#plots maximum heat fluxes from csv file
#where each row is a separate aoi
#Engineer: T Looby
#Date:     20230927
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator


f = '/home/tlooby/projects/dummy/hfMax.csv'
data = np.genfromtxt(f, delimiter=',')
print(data)
fig = go.Figure()
N = data.shape[1] - 1
color = px.colors.qualitative.Plotly

symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down']
for i in range(data.shape[0]):
    print(data[i,1:])
    fig.add_trace(go.Scatter(x=np.arange(N), y=data[i,1:], 
                             mode = 'lines+markers',
                             name='{:0.1f}deg'.format(data[i,0]),
                             marker_symbol=symbols[i],
                             marker_line_color=color[0], 
                             marker_color="lightskyblue",
                             marker_line_width=2, marker_size=15,
                             )
                )

fig.update_xaxes(title='Polynomial #')
fig.update_yaxes(title='Max HF Fraction of q0')

fig.show()