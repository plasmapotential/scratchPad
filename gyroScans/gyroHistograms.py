import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


prefix = '/home/tom/results/gyroConvergence2/diamagnetic/10eV/'

#use this to write an EPS file
writeEPS = False
#epsFile = '/home/tom/phd/dissertation/diss/figures/conv2Hist.eps'
epsFile = '/home/tom/work/ORNL_postdoc/manuscripts/2022_NF_NSTXUgyro/NF_rev1/figures/conv2Hist.eps'

#for the old convergence test with plane (ie not a cube shape)
#cases = [
#            '1gP_1vP_1vS',
#            '2gP_2vP_2vS',
##            '3gP_1vP_1vS',
##            '3gP_1vP_3vS',
##            '3gP_3vP_1vS',
#            '3gP_3vP_3vS',
##            '5gP_1vP_1vS',
##            '5gP_1vP_5vS',
##            '5gP_5vP_1vS',
#            '5gP_5vP_5vS',
##            '9gP_1vP_1vS',
##            '9gP_1vP_9vS',
##            '9gP_9vP_1vS',
#            '9gP_9vP_9vS'
#        ]
#get 121212 as ground truth
#truthFile = prefix+'12gP_12vP_12vS/HF_gyro_all.csv'

#for the case with the cube and diamagnetism turned on
cases = [
            '1gP1vP1vS',
            '2gP2vP2vS',
            '3gP3vP3vS',
            '4gP4vP4vS',
        ]
#get 555 as ground truth
truthFile = prefix+'5gP5vP5vS/HF_gyro.csv'


#labels = [
#            '111=>222',
#            '222=>311',
#            '311=>313',
#            '313=>331',
#            '331=>333',
#            '333=>511',
#            '511=>515',
#            '515=>551',
#            '551=>555',
#            '555=>911',
#            '911=>919',
#            '919=>991',
#            '991=>999',
#        ]

labels1 = {
	'1gP1vP1vS':'(1,1,1)',
	'2gP2vP2vS':'(2,2,2)',
	'3gP3vP3vS':'(3,3,3)',
	'4gP4vP4vS':'(4,4,4)',
	}
labels2 = [
	'(1,1,1)',
	'(2,2,2)',
	'(3,3,3)',
	'(4,4,4)',
	]

#get all qGyro data from cases
qGyroData = []
for case in cases:
    #for old plane convergence tests (not cube)
    #file = prefix+case+'/HF_gyro_all.csv'
    file = prefix+case+'/HF_gyro.csv'
    data = pd.read_csv(file)
    qGyroData.append(data.iloc[:,3].values)

qGyroData = np.asarray(qGyroData)

dfTruth = pd.read_csv(truthFile)
qTruth = dfTruth.iloc[:,3].values

#Histogram using graph_objects
#fig = go.Figure()
#truthDiff = []
#for j in range(len(cases)-1):
#    dTruth = qGyroData[j] - qTruth
#    fig = fig.add_trace(go.Histogram(x=dTruth, opacity=0.7, name=cases[j]))
#fig.update_layout(barmode='stack')
#fig.show()

#Histogram using plotly_express
truthDiff = []
for j in range(len(cases)):
    dTruth = qGyroData[j] - qTruth
    truthDiff.append(dTruth)
truthDiff = np.asarray(truthDiff)
df = pd.DataFrame(truthDiff.T, columns=cases)
df.columns=labels2

#histogram with box plot overlay
#fig = px.histogram(df, marginal="box", barmode='overlay', range_x=[-75,75], opacity=0.6, log_y=True, color_discrete_sequence=px.colors.qualitative.G10)
#histogram with rug plot overlay
#fig = px.histogram(df, marginal="rug", barmode='overlay', range_x=[-75,75], opacity=0.6, log_y=True, color_discrete_sequence=px.colors.qualitative.G10)
fig = px.histogram(df, barmode='overlay', range_x=[-75,75], opacity=0.6, log_y=True, color_discrete_sequence=px.colors.qualitative.G10)

#histogram with no box plot
#fig = px.histogram(df, barmode='overlay', range_x=[-75,75], opacity=0.6, log_y=True, color_discrete_sequence=px.colors.qualitative.G10)
#plot curves only (no hist)
#fig = ff.create_distplot([df[c] for c in df.columns], df.columns, bin_size=0.25, show_hist=False, show_rug=True, colors=px.colors.qualitative.G10)

fig.update_layout(
    #title="Power Difference from 5gP_5vP_5vS",
    xaxis_title="Heat Flux Difference [MW/m2] from (5,5,5)",
    yaxis_title="Count",
    autosize=True,
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.7,
        xanchor="left",
        x=0.01,
        title="",
        ),
    font=dict(
    size=18,
    ),
    margin=dict(
        l=5,
        r=5,
        b=5,
        t=5,
        pad=2
    ),
)


fig.show()

if writeEPS==True:
    fig.write_image(epsFile)
