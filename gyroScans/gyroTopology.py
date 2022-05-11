#plots the topology of gyro space and colorbar of IQR
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


prefix = '/home/tom/results/gyroTopology/'

origCases = [
            '1gP_1vP_1vS',
            '2gP_2vP_2vS',
            '3gP_1vP_1vS',
            '3gP_1vP_3vS',
            '3gP_3vP_1vS',
            '3gP_3vP_3vS',
            '5gP_1vP_1vS',
            '5gP_1vP_5vS',
            '5gP_5vP_1vS',
            '5gP_5vP_5vS',
            '9gP_1vP_1vS',
            '9gP_1vP_9vS',
            '9gP_9vP_1vS',
            '9gP_9vP_9vS'
        ]
static_vS_cases = [
    '1gP_1vP_3vS',
    '1gP_2vP_3vS',
    '1gP_3vP_3vS',
    '2gP_1vP_3vS',
    '2gP_2vP_3vS',
    '2gP_3vP_3vS',
    '3gP_1vP_3vS',
    '3gP_2vP_3vS',
    '3gP_3vP_3vS',
]
static_gP_cases = [
    '3gP_1vP_1vS',
    '3gP_1vP_2vS',
    '3gP_1vP_3vS',
    '3gP_2vP_1vS',
    '3gP_2vP_2vS',
    '3gP_2vP_3vS',
    '3gP_3vP_1vS',
    '3gP_3vP_2vS',
    '3gP_3vP_3vS',
]
axes_static_gP = ['vP','vS']
axes_static_vS = ['gP','vP']

#cases = static_gP_cases
#axisLabels = axes_static_gP
cases = static_vS_cases
axisLabels = axes_static_vS


gP = []
vP = []
vS = []

for case in cases:
    tmp = case.split('_')
    gP.append(tmp[0].split('g')[0])
    vP.append(tmp[1].split('v')[0])
    vS.append(tmp[2].split('v')[0])
gP = np.asarray(gP)
vP = np.asarray(vP)
vS = np.asarray(vS)

manifold = np.vstack([gP,vP,vS]).T
df = pd.DataFrame(manifold, columns=['xdata','ydata','zdata'])

#get all qGyro data from cases
qGyroData = []
for case in cases:
    file = prefix+case+'/HF_gyro_all.csv'
    data = pd.read_csv(file)
    qGyroData.append(data.iloc[:,3].values)

qGyroData = np.asarray(qGyroData)
#get 999 as ground truth
truthFile = prefix+'9gP_9vP_9vS/HF_gyro_all.csv'
dfTruth = pd.read_csv(truthFile)
qTruth = dfTruth.iloc[:,3].values

##surface plot
#truthDiff = []
#for j in range(len(cases)):
#    dTruth = qGyroData[j] - qTruth
#    truthDiff.append(dTruth)
#truthDiff = np.asarray(truthDiff)
#df2 = pd.DataFrame(truthDiff.T, columns=cases)
#quartiles = df2.quantile([0.25, 0.75]).values.T
#IQR = quartiles[:,1] - quartiles[:,0]
#
#z = IQR.reshape(3,3).T
#print(z)
#print(IQR)
#print(cases)
#fig = go.Figure(go.Surface(z=z, x=[1,2,3], y=[1,2,3], colorscale="Electric", opacity=0.7,
#                          contours = {"z": {"show": True, "start": 0.0, "end": 10, "size": 0.1}},
#                          )
#                )
#fig.update_layout(scene = dict(
#                    xaxis=dict(title=dict(font=dict(size=28), text=axisLabels[0])),
#                    yaxis=dict(title=dict(font=dict(size=28), text=axisLabels[1])),
#                    zaxis=dict(title=dict(font=dict(size=28), text="IQR")),
#                    aspectratio = dict(x=1,y=1,z=0.5)
#                    ),
#                    font=dict(size=16),
#
#                    )

#contour plot
truthDiff = []
for j in range(len(cases)):
    dTruth = qGyroData[j] - qTruth
    truthDiff.append(dTruth)
truthDiff = np.asarray(truthDiff)
df2 = pd.DataFrame(truthDiff.T, columns=cases)
quartiles = df2.quantile([0.25, 0.75]).values.T
IQR = quartiles[:,1] - quartiles[:,0]

z = IQR.reshape(3,3).T
print(z)
print(IQR)
print(cases)

fig = go.Figure(data =
    go.Contour(z=z, x=[1,2,3], y=[1,2,3], colorscale="Electric",
                contours=dict(
                    coloring ='heatmap',
                    showlabels = True, # show labels on contours
                    start=min(IQR),
                    end=max(IQR),
                    size=0.1,
                    labelfont = dict( # label font properties
                        size = 18,
                        color = 'white',
                        )
                    ),
        colorbar=dict(
            title='IQR', # title here
            titleside='top',
            titlefont=dict(size=28)
                )
                ))


fig.update_layout(
    xaxis_title=axisLabels[0],
    yaxis_title=axisLabels[1],
    font=dict(size=28)
    )


fig.show()
