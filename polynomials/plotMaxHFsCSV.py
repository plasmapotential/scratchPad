#plots maximum heat fluxes from csv file
#where each row is a separate aoi
#Engineer: T Looby
#Date:     20230927
import plotly.graph_objects as go
import numpy as np

f = '/home/tlooby/projects/dummy/hfMax.csv'
data = np.genfromtxt(f, delimiter=',')
fig = go.Figure()
N = data.shape[1] - 2
print(data.shape[1])
for i in range(data.shape[0]):
    fig.add_trace(go.Scatter(x=np.arange(N)+1, y=data[i,1:], name='{:0.2f}'.format(data[i,0])))

fig.update_xaxes(title='Polynomial #')
fig.update_yaxes(title='Max HF Fraction of q0')

fig.show()