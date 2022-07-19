import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
N=3
#vPhases = np.degrees(np.linspace(0.0,np.pi/2,N+2)[1:-1])
#vPhases = np.degrees(np.linspace(np.sin(0.0)**2,np.sin(np.pi/2)**2, N+2)[1:-1])
vPhases = np.degrees(np.linspace(np.cos(2*0.0),np.cos(np.pi), N+2)[1:-1])
fig = go.Figure()
for i in range(len(vPhases)):
    fig.add_trace(go.Scatterpolar(
            r = np.array([0.0,1.0]),
            theta = np.array([vPhases[i],vPhases[i]]),
            mode = 'lines+markers',
            marker={'symbol':"square", 'size':16},
            name = '{:0.1f}'.format(vPhases[i]),
            text=['{:0.1f}'.format(vPhases[i])],
            textposition="bottom center",
            #line_color= px.colors.qualitative.Dark2[0],
            line={'width':7},
            ))
fig.add_annotation(x=0.2, y=0.5,
            text="V||",
            font=dict(size=16),
            showarrow=False,
            )
fig.update_annotations(font_size=16)
fig.update_layout(
    title={'text': "Velocity Phase Angles",'y':0.94,'x':0.5,'xanchor': 'center','yanchor': 'top'},
    showlegend = False,
    polar = dict(
        sector = [0,90],
        radialaxis=dict(title=dict(text="V\u22A5",font=dict(size=16))),
        #angularaxis=dict(title=dict(text='$V_{||}$',font=dict(size=24)))
        ),
        )
fig.show()
