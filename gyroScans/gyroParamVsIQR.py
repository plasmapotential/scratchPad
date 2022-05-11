#plots gyro vP, vS, gP vs IQR
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
vS_cases = [
    '3gP_3vP_1vS',
    '3gP_3vP_2vS',
    '3gP_3vP_3vS',
    '3gP_3vP_4vS',
    '3gP_3vP_5vS',
]
vP_cases = [
    '3gP_1vP_3vS',
    '3gP_2vP_3vS',
    '3gP_3vP_3vS',
    '3gP_4vP_3vS',
    '3gP_5vP_3vS',
]
gP_cases = [
    '1gP_3vP_3vS',
    '2gP_3vP_3vS',
    '3gP_3vP_3vS',
    '4gP_3vP_3vS',
    '5gP_3vP_3vS',
]
all_cases = [
    '1gP_1vP_1vS',
    '2gP_2vP_2vS',
    '3gP_3vP_3vS',
    '4gP_4vP_4vS',
    '5gP_5vP_5vS',
]

axes_gP = ['gP','IQR']
axes_vS = ['vS','IQR']
axes_vP = ['vP','IQR']

#get 999 as ground truth
truthFile = prefix+'9gP_9vP_9vS/HF_gyro_all.csv'
dfTruth = pd.read_csv(truthFile)
qTruth = dfTruth.iloc[:,3].values

#scatter plot
fig = go.Figure()
x = [1,2,3,4,5]

#gyroPhase
truthDiff = []
for j,case in enumerate(gP_cases):
    file = prefix+case+'/HF_gyro_all.csv'
    data = pd.read_csv(file)
    dTruth = data.iloc[:,3].values - qTruth
    truthDiff.append(dTruth)
df2 = pd.DataFrame(np.asarray(truthDiff).T)
quartiles = df2.quantile([0.25, 0.75]).values
IQR = quartiles[1] - quartiles[0]
fig.add_trace(go.Scatter(x=x, y=IQR, name='gP: (N,3,3)'))
print("gP")
print(IQR)

#vPhase
truthDiff = []
for j,case in enumerate(vP_cases):
    file = prefix+case+'/HF_gyro_all.csv'
    data = pd.read_csv(file)
    dTruth = data.iloc[:,3].values - qTruth
    truthDiff.append(dTruth)
df2 = pd.DataFrame(np.asarray(truthDiff).T)
quartiles = df2.quantile([0.25, 0.75]).values
IQR = quartiles[1] - quartiles[0]
fig.add_trace(go.Scatter(x=x, y=IQR, name='vP: (3,N,3)'))
print("vP")
print(IQR)
#
#
#vSlice
truthDiff = []
for j,case in enumerate(vS_cases):
    file = prefix+case+'/HF_gyro_all.csv'
    data = pd.read_csv(file)
    dTruth = data.iloc[:,3].values - qTruth
    truthDiff.append(dTruth)
df2 = pd.DataFrame(np.asarray(truthDiff).T)
quartiles = df2.quantile([0.25, 0.75]).values
IQR = quartiles[1] - quartiles[0]
fig.add_trace(go.Scatter(x=x, y=IQR, name='vS: (3,3,N)'))
print("vS")
print(IQR)

#full
truthDiff = []
for j,case in enumerate(all_cases):
    file = prefix+case+'/HF_gyro_all.csv'
    data = pd.read_csv(file)
    dTruth = data.iloc[:,3].values - qTruth
    truthDiff.append(dTruth)
df2 = pd.DataFrame(np.asarray(truthDiff).T)
quartiles = df2.quantile([0.25, 0.75]).values
IQR = quartiles[1] - quartiles[0]
fig.add_trace(go.Scatter(x=x, y=IQR, name='all: (N,N,N)'))
print("NNN")
print(IQR)


#fig.add_annotation(
#        x=1.25,
#        y=8.5,
#        xref="x",
#        yref="y",
#        text="gyroPhase",
#        showarrow=False,
#        font=dict(
#            family="Courier New, monospace",
#            size=20,
#            color="#ffffff"
#            ),
#        align="center",
#        arrowhead=2,
#        arrowsize=1,
#        arrowwidth=2,
#        arrowcolor="#636363",
#        ax=20,
#        ay=-30,
#        bordercolor="#c7c7c7",
#        borderwidth=2,
#        borderpad=4,
#        bgcolor="#636EFA",
#        opacity=0.8
#        )
#
#
#fig.add_annotation(
#        x=1.25,
#        y=7.5,
#        xref="x",
#        yref="y",
#        text="vPhase",
#        showarrow=False,
#        font=dict(
#            family="Courier New, monospace",
#            size=20,
#            color="#ffffff"
#            ),
#        align="center",
#        arrowhead=2,
#        arrowsize=1,
#        arrowwidth=2,
#        arrowcolor="#636363",
#        ax=20,
#        ay=-30,
#        bordercolor="#c7c7c7",
#        borderwidth=2,
#        borderpad=4,
#        bgcolor="#EF553B",
#        opacity=0.8
#        )
#
#fig.add_annotation(
#        x=1.25,
#        y=5.75,
#        xref="x",
#        yref="y",
#        text="vPhase",
#        showarrow=False,
#        font=dict(
#            family="Courier New, monospace",
#            size=20,
#            color="#ffffff"
#            ),
#        align="center",
#        arrowhead=2,
#        arrowsize=1,
#        arrowwidth=2,
#        arrowcolor="#636363",
#        ax=20,
#        ay=-30,
#        bordercolor="#c7c7c7",
#        borderwidth=2,
#        borderpad=4,
#        bgcolor="#00CC96",
#        opacity=0.8
#        )



fig.update_layout(
    xaxis_title="N",
    yaxis_title="IQR",
    font=dict(size=28),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
        ),
    )



fig.show()
