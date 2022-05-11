import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats


prefix = '/home/tom/results/gyroMELscans/tinyPlane/'

cases = [
            '3mm_3gP_3vP_3vS',
            '2mm_3gP_3vP_3vS',
            '1mm_3gP_3vP_3vS',
#            '0.5mm_3gP_3vP_3vS',
        ]

#get all qGyro data from cases
qGyroData = []
for case in cases:
    file = prefix+case+'/HF_gyro_all.csv'
    data = pd.read_csv(file)
    qGyroData.append(data.iloc[:,3].values)

qGyroData = np.asarray(qGyroData)


#get 999 as ground truth
#truthFile = prefix+'9gP_9vP_9vS/HF_gyro_all.csv'
#dfTruth = pd.read_csv(truthFile)
#qTruth = dfTruth.iloc[:,3].values
#truthDiff = []
#for j in range(len(cases)):
#    dTruth = qGyroData[j] - qTruth
#    truthDiff.append(dTruth)
#truthDiff = np.asarray(truthDiff)
#df = pd.DataFrame(truthDiff.T, columns=cases[:-1])

#using graph_objects
nBins = 500
fig = go.Figure()
colors1 = [px.colors.qualitative.D3[0], px.colors.qualitative.D3[1], px.colors.qualitative.D3[2]]
colors2 = [px.colors.qualitative.Plotly[5], px.colors.qualitative.Plotly[9], px.colors.qualitative.Plotly[2]]
for j in range(len(cases)):
    #histogram
    fig = fig.add_trace(go.Histogram(x=qGyroData[j], opacity=0.7, name=cases[j], histnorm='percent', nbinsx=nBins, marker=dict(color=colors1[j])))
    #box plots
    #fig.add_trace(go.Box(x=qGyroData[j], opacity=0.7, name=cases[j]))
    #fig.add_trace(go.Box(x=truthDiff[j], opacity=0.7, name=cases[j]))
    #histogram outline only
    count, index = np.histogram(qGyroData[j], bins=nBins)
    countPercent = count / np.sum(count)*100.0
    #fig.add_traces(go.Scatter(x=index, y = countPercent, line=dict(width = 1, shape='hvh')))
    #density plot
    N = len(count)
    N_window = 10
    runMean = np.convolve(countPercent, np.ones(N_window)/N_window )
    fig.add_traces(go.Scatter(x=index, y = runMean, name="Smoothed "+cases[j], marker_color=colors2[j],))

fig.update_layout(barmode='overlay')
fig.show()
