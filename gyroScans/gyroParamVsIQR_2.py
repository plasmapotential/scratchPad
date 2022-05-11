#plots gyro vP, vS, gP vs IQR
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


#for original gyroConvergence with plane
#prefix = '/home/tom/results/gyroTopology/'
#for gyroConvergence2 (not plane, cube)
prefix = '/home/tom/results/gyroConvergence2/diamagnetic/10eV/'

#use this to write an EPS file
writeEPS = False
epsFile = '/home/tom/phd/dissertation/diss/figures/IQRplot.eps'


vS_cases = [
    '5gP_1vP_1vS',
    '5gP_1vP_2vS',
    '5gP_1vP_3vS',
    '5gP_1vP_4vS',
    '5gP_1vP_5vS',
]
vP_cases = [
    '5gP_1vP_1vS',
    '5gP_2vP_1vS',
    '5gP_3vP_1vS',
    '5gP_4vP_1vS',
    '5gP_5vP_1vS',
]

vPvS_cases = [
    '5gP_1vP_1vS',
    '5gP_2vP_2vS',
    '5gP_3vP_3vS',
    '5gP_4vP_4vS',
    '5gP_5vP_5vS',
]

#for plane cases
#all_cases = [
#    '1gP_1vP_1vS',
#    '2gP_2vP_2vS',
#    '3gP_3vP_3vS',
#    '4gP_4vP_4vS',
#    '5gP_5vP_5vS',
#]

#for gyroConvergence2 cases (with the cube, not plane)
all_cases = [
    '1gP1vP1vS',
    '2gP2vP2vS',
    '3gP3vP3vS',
    '4gP4vP4vS',
    '5gP5vP5vS',
]



axes_gP = ['gP','IQR']
axes_vS = ['vS','IQR']
axes_vP = ['vP','IQR']

#get 999 as ground truth
#truthFile = prefix+'9gP_9vP_9vS/HF_gyro_all.csv'

#gyroConvergence2
#get 555 as ground truth
truthFile = prefix+'5gP5vP5vS/HF_gyro.csv'
dfTruth = pd.read_csv(truthFile)
qTruth = dfTruth.iloc[:,3].values

#scatter plot
fig = go.Figure()
x = [1,2,3,4,5]

##vPhase
#truthDiff = []
#for j,case in enumerate(vP_cases):
#    file = prefix+case+'/HF_gyro_all.csv'
#    data = pd.read_csv(file)
#    dTruth = data.iloc[:,3].values - qTruth
#    truthDiff.append(dTruth)
#df2 = pd.DataFrame(np.asarray(truthDiff).T)
#quartiles = df2.quantile([0.25, 0.75]).values
#IQR = quartiles[1] - quartiles[0]
#fig.add_trace(go.Scatter(x=x, y=IQR, name='vP: (5,N,1)', marker_symbol='circle', marker_size=15, line=dict(width=3)))
#print("vP")
#print(IQR)
##
##
##vSlice
#truthDiff = []
#for j,case in enumerate(vS_cases):
#    file = prefix+case+'/HF_gyro_all.csv'
#    data = pd.read_csv(file)
#    dTruth = data.iloc[:,3].values - qTruth
#    truthDiff.append(dTruth)
#df2 = pd.DataFrame(np.asarray(truthDiff).T)
#quartiles = df2.quantile([0.25, 0.75]).values
#IQR = quartiles[1] - quartiles[0]
#fig.add_trace(go.Scatter(x=x, y=IQR, name='vS: (5,1,N)', marker_symbol='square', marker_size=15, line=dict(width=3)))
#print("vS")
#print(IQR)

#vPvS
#truthDiff = []
#for j,case in enumerate(vPvS_cases):
#    file = prefix+case+'/HF_gyro_all.csv'
#    data = pd.read_csv(file)
#    dTruth = data.iloc[:,3].values - qTruth
#    truthDiff.append(dTruth)
#df2 = pd.DataFrame(np.asarray(truthDiff).T)
#quartiles = df2.quantile([0.25, 0.75]).values
#IQR = quartiles[1] - quartiles[0]
#fig.add_trace(go.Scatter(x=x, y=IQR, name='vPvS: (5,N,N)', marker_symbol='cross',marker_size=15, line=dict(width=3)))
#print("5NN")
#print(IQR)

##subtract heat flux then calculate IQR
##all
#truthDiff = []
#for j,case in enumerate(all_cases):
#    #plane cases
#    #file = prefix+case+'/HF_gyro_all.csv'
#    #cube cases (gyroConvergence2)
#    file = prefix+case+'/HF_gyro.csv'
#    data = pd.read_csv(file)
#    dTruth = data.iloc[:,3].values - qTruth
#    truthDiff.append(dTruth)
#df2 = pd.DataFrame(np.asarray(truthDiff).T)
#quartiles = df2.quantile([0.25, 0.75]).values
#IQR = quartiles[1] - quartiles[0]

#===first calculate IQR, then subtract from ground truth
dfTruth = pd.DataFrame(np.array(qTruth))
quarts = dfTruth.quantile([0.25, 0.75]).values
IQRtruth = np.abs(quarts[1] - quarts[0])
IQR = []
for j,case in enumerate(all_cases):
    #plane cases
    #file = prefix+case+'/HF_gyro_all.csv'
    #cube cases (gyroConvergence2)
    file = prefix+case+'/HF_gyro.csv'
    data = pd.read_csv(file)
#    df2 = pd.DataFrame(data.iloc[:,3].values.sort())
    df2 = pd.DataFrame(data.iloc[:,3].values)
    quarts = df2.quantile([0.25, 0.75]).values
    IQRtemp = np.abs(quarts[1] - quarts[0])
    IQR.append(IQRtemp)

IQR = np.array(IQR)
IQR = np.abs(IQRtruth - IQR[:,0])



fig.add_trace(go.Scatter(x=x, y=IQR, name='all: (N,N,N)', marker_symbol='triangle-up', marker_size=15, line=dict(width=3)))
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
    yaxis_title="|IQR(N,N,N) - IQR(5,5,5)|",
    xaxis={'tickvals':x},
    font=dict(size=20),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
        ),

    )



fig.show()
if writeEPS==True:
    fig.write_image(epsFile)
