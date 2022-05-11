import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

gyroPhases = np.array([0.0, 120.0, 240.0])
vPhases = np.array([22.5, 45.0, 67.5])
vSlices = np.array([25635,56825, 98525])
gyroTicks = [0, 45, 135, 180, 225, 270, 315]

#====================== Gyro Phase
#fig = go.Figure()
#for i in range(len(gyroPhases)):
#    fig.add_trace(go.Scatterpolar(
#                r = np.array([0.0,1.0]),
#                theta = np.array([gyroPhases[i],gyroPhases[i]]),
#                mode = 'lines+markers',
#                marker={'symbol':"circle", 'size':20},
#                name = '{:0.1f}'.format(gyroPhases[i]),
#                text=['{:0.1f}'.format(gyroPhases[i])],
#                textposition="bottom center",
#                #line_color= px.colors.qualitative.Dark2[0],
#                line={'width':10},
#                ))
#
#fig.update_layout(
#     title={'text': "Gyro Phase Angles",'y':0.94,'x':0.5,'xanchor': 'center','yanchor': 'top'},
#     showlegend = False,
#     polar = dict(radialaxis=dict(visible=False), angularaxis=dict(tickvals=gyroTicks)),
#    )
#fig.show()

#====================== vPhase
#fig = go.Figure()
#for i in range(len(vPhases)):
#    fig.add_trace(go.Scatterpolar(
#                r = np.array([0.0,1.0]),
#                theta = np.array([vPhases[i],vPhases[i]]),
#                mode = 'lines+markers',
#                marker={'symbol':"square", 'size':20},
#                name = '{:0.1f}'.format(vPhases[i]),
#                text=['{:0.1f}'.format(vPhases[i])],
#                textposition="bottom center",
#                #line_color= px.colors.qualitative.Dark2[0],
#                line={'width':10},
#                ))
#
#
#
#fig.add_annotation(x=0.25, y=0.5,
#            text="V||",
#            font=dict(size=16),
#            showarrow=False,
#            )
#
#fig.update_annotations(font_size=16)
#
#fig.update_layout(
#    title={'text': "Velocity Phase Angles",'y':0.94,'x':0.5,'xanchor': 'center','yanchor': 'top'},
#    showlegend = False,
#    polar = dict(
#        sector = [0,90],
#        radialaxis=dict(title=dict(text="V\u22A5",font=dict(size=16))),
#        #angularaxis=dict(title=dict(text='$V_{||}$',font=dict(size=24)))
#        ),
#    )
#
#fig.show()

#====================== vSlice
vMax = 300000
v = np.linspace(0,vMax,100)
vSlice = np.array([50000, 100000])


mass_eV = 2.014*931.49*10e6
T0 = 10
c = 3000000000


#generate the (here maxwellian) PDF
pdf = lambda x: (mass_eV/c**2) / (T0) * np.exp(-(mass_eV/c**2 * x**2) / (2*T0) )
v_pdf = v * pdf(v)
vSlice_pdf = vSlice * pdf(vSlice)



fig = go.Figure()
fig.add_trace(go.Scatter(x=v, y=v_pdf,
                    mode='lines',
                    line={'width':6},
                    name='Maxwellian PDF'))

fig.add_trace(go.Scatter(x=vSlice, y=vSlice_pdf,
                    mode='markers',
                    marker={'symbol':"square", 'size':16},
                    name='Slices'))


fig.update_layout(
    title="Velocity Distribution",
    yaxis= dict(showticklabels=False),
    xaxis_title="Velocity",
    font=dict(size=18),

    )


fig.show()
#====================== SUBPLOTS
#fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}]*2],
#                    subplot_titles=("Gyro Phase Angles", "Velocity Phase Angles"))
#for i in range(len(gyroPhases)):
#    fig.add_trace(go.Scatterpolar(
#                r = np.array([0.0,1.0]),
#                theta = np.array([gyroPhases[i],gyroPhases[i]]),
#                mode = 'lines+markers',
#                marker={'symbol':"circle", 'size':20},
#                name = '{:0.1f}'.format(gyroPhases[i]),
#                text=['{:0.1f}'.format(gyroPhases[i])],
#                textposition="bottom center",
#                #line_color= px.colors.qualitative.Dark2[0],
#                line={'width':10},
#                ), 1, 1)
#
#for i in range(len(vPhases)):
#    fig.add_trace(go.Scatterpolar(
#                r = np.array([0.0,1.0]),
#                theta = np.array([vPhases[i],vPhases[i]]),
#                mode = 'lines+markers',
#                marker={'symbol':"square", 'size':20},
#                name = '{:0.1f}'.format(vPhases[i]),
#                text=['{:0.1f}'.format(vPhases[i])],
#                textposition="bottom center",
#                #line_color= px.colors.qualitative.Dark2[0],
#                line={'width':10},
#                ), 1, 2)
#
#
#
#fig.add_annotation(x=0.53, y=0.5,
#            text="$V_{||}$",
#            font=dict(size=24),
#            showarrow=False,
#            )
#
#fig.update_annotations(font_size=24)
#
#fig.update_layout(
#    showlegend = False,
#     polar = dict(radialaxis=dict(visible=False), angularaxis=dict(tickvals=gyroTicks)),
#     polar2 = dict(
#        sector = [0,90],
#        radialaxis=dict(title=dict(text='$V_\perp$',font=dict(size=24))),
#        #angularaxis=dict(title=dict(text='$V_{||}$',font=dict(size=24)))
#        )
#    )
#fig.show()
