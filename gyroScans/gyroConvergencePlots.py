import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px

prefix = '/home/tom/results/gyroConvergence/'

cases = [
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
labels = [
            '111=>222',
            '222=>311',
            '311=>313',
            '313=>331',
            '331=>333',
            '333=>511',
            '511=>515',
            '515=>551',
            '551=>555',
            '555=>911',
            '911=>919',
            '919=>991',
            '991=>999',
        ]

#get all qGyro data from cases
qGyroData = []
for case in cases:
    file = prefix+case+'/HF_gyro_all.csv'
    data = pd.read_csv(file)
    qGyroData.append(data.iloc[:,3].values)

qGyroData = np.asarray(qGyroData)

qOptical = pd.read_csv(prefix+'HF_optical_all.csv').iloc[:,3].values

#now calculate error quantities from this data
difference = []
avgDiff = []
maxDiff = []

for j in range(len(cases)-1):
    dQ = np.abs(qGyroData[j+1] - qGyroData[j])
    difference.append(dQ)
    maxDiff.append(np.max(dQ))
    avgDiff.append(np.average(dQ))

difference = np.asarray(difference)
avgDiff = np.asarray(avgDiff)
maxDiff = np.asarray(maxDiff)

optDiff = []
optMax = []
optAvg = []
for j in range(len(cases)):
    dOpt = np.abs(qGyroData[j] - qOptical)
    optDiff.append(dOpt)
    optMax.append(np.max(dOpt))
    optAvg.append(np.average(dOpt))

optDiff = np.asarray(optDiff)
optMax = np.asarray(optMax)
optAvg = np.asarray(optAvg)


x = np.arange(len(cases))

#===Relative Differences
#avgDiff
fig = go.Figure(data =
    go.Scatter(
        x=labels,
        y=avgDiff,
        mode="markers+lines",
        name="Average Difference",
        #line=dict(
        #    color="#fc0317"
        #        )
        )
    )
#max difference
fig.add_trace(
    go.Scatter(
        x=labels,
        y=maxDiff,
        mode="markers+lines",
        name="Maximum Difference",
        #line=dict(
        #    color="#fc0317"
        #        )
        )
        )

##===Differences from Optical
##avgDiff
#fig = go.Figure(data =
#    go.Scatter(
#        x=cases,
#        y=optAvg,
#        mode="markers+lines",
#        name="Average Difference",
#        #line=dict(
#        #    color="#fc0317"
#        #        )
#        )
#    )
##max difference
#fig.add_trace(
#    go.Scatter(
#        x=cases,
#        y=optMax,
#        mode="markers+lines",
#        name="Maximum Difference",
#        #line=dict(
#        #    color="#fc0317"
#        #        )
#        )
#        )




fig.update_layout(
    title="",
    xaxis_title="Case",
    yaxis_title="Difference [MW/m2]",
    autosize=True,
    showlegend=True,
    font=dict(
#            family="Courier New",
        size=26,
        ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
        )
    )

fig.show()
