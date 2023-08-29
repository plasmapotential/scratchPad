#plots q as a function of R, psi, psiN

import numpy as np
import plotly.graph_objects as go
import pandas as pd


#root = '/home/tlooby/HEAT/data/sparc_001658_MEQscan_T5B_5MW_600um_bdotn_8.7MA/'
root = '/home/tlooby/results/boundaryOps/qProfiles/'

#data4 = np.genfromtxt(root+'data4.csv', delimiter=' ')
#data8 = np.genfromtxt(root+'data8.csv', delimiter=' ')

data4 = pd.read_csv(root+'data4_lqVaries.csv', delimiter=' ', header=0)
data8 = pd.read_csv(root+'data8_lqVaries.csv', delimiter=' ', header=0)

d4 = data4.values
d8 = data8.values


print(d4[0])
fig = go.Figure()
maxIdx = np.argmax(d8[:,3])
delta = d4[maxIdx, 2] - d8[maxIdx, 2]
print(delta)

print(data4.columns)


symbols = ['x', 'star', 'diamond', 'asterisk', 'bowtie', 'hourglass', 'circle-x', 'hexagram' ]

#R
fig.add_trace(go.Scatter(x=data8['R_omp'], y=data8['q'] ,name= '8.7MA',
                         mode='lines+markers', marker_size=10, marker_symbol=symbols[0], 
                         marker=dict(maxdisplayed=200)))
fig.add_trace(go.Scatter(x=data4['R_omp'], y=data4['q'] ,name= '4.35MA',
                         mode='lines+markers', marker_size=10, marker_symbol=symbols[1], 
                         marker=dict(maxdisplayed=200)))
xTitle = 'R_omp [m]'
yTitle = 'q_norm'

#psiN
#fig.add_trace(go.Scatter(x=data8['psiN'], y=data8['q'] ,name= '8.7MA',
#                         mode='lines+markers', marker_size=10, marker_symbol=symbols[0], 
#                         marker=dict(maxdisplayed=200)))
#fig.add_trace(go.Scatter(x=data4['psiN'], y=data4['q'] ,name= '4.35MA',
#                         mode='lines+markers', marker_size=10, marker_symbol=symbols[1], 
#                         marker=dict(maxdisplayed=200)))
#xTitle = 'psi_norm'
#yTitle = 'q_norm'
#psi
#fig.add_trace(go.Scatter(x=data8['psi'], y=data8['q'] ,name= '8.7MA',
#                         mode='lines+markers', marker_size=10, marker_symbol=symbols[0], 
#                         marker=dict(maxdisplayed=200)))
#fig.add_trace(go.Scatter(x=data4['psi'], y=data4['q'] ,name= '4.35MA',
#                         mode='lines+markers', marker_size=10, marker_symbol=symbols[1], 
#                         marker=dict(maxdisplayed=200)))
#xTitle = 'psi [Wb/rad]'
#yTitle = 'q_norm'

fig.update_layout(
    #title="Plot Title",
    xaxis_title=xTitle,
    yaxis_title=yTitle,
    font=dict(
        family="Courier New, monospace",
        size=18,
    )
)
fig.show()