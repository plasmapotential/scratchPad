#coverPoly.py
#Description:   cover a polygon with tesselation of shapes, then color by group ID
#Engineer:      T Looby
#Date:          20220505 (cinco de mayo...)

import json
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
import plotly.graph_objects as go

#======= Create Mesh / Tesselation
#this is an example shape used for testing
polyX = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
polyY = np.array([0.0, 1.0, 2.0, 0.5, 0.0])

#we will tesselate with rectangles / squares here
def square(x, y, s):
    return Polygon([(x, y), (x+s, y), (x+s, y+s), (x, y+s)])

poly = MultiPoint(np.vstack([polyX,polyY]).T).convex_hull
polyCoords = np.array(poly.exterior.coords)

grid_size = 0.1
ibounds = np.array(poly.bounds)//grid_size
ibounds[2:4] += 1
xmin, ymin, xmax, ymax = ibounds*grid_size
xrg = np.arange(xmin, xmax, grid_size)
yrg = np.arange(ymin, ymax, grid_size)
mp = MultiPolygon([square(x, y, grid_size) for x in xrg for y in yrg])
solution = MultiPolygon(list(filter(poly.intersects, mp)))

#======= Create Figure and Group ID painting
fig = go.Figure(go.Scatter(x=polyCoords[:,0], y=polyCoords[:,1], fill="toself"))

for geom in solution.geoms:
    xs, ys = np.array(geom.exterior.xy)
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", line=dict(color="seagreen")))

#create df for table
IDs = np.arange(len(solution.geoms))
groups = np.zeros((len(IDs)))
data = np.vstack([IDs, groups]).T
df = pd.DataFrame(data, columns=['ID', 'Group'])

fig.update_layout(clickmode='event+select')
fig.update_layout(showlegend=False)
fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)

app = Dash(__name__)

#CSS stylesheets
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    },
    'bigApp': {
        'max-width': '100%',
        'display': 'flex',
        'flex-direction': 'row',
        'width': '100vw',
        'height': '80vh',
        'vertical-align': 'middle',
    },
    'graph': {
#        'display': 'flex',
        'width': '45%',
        'height': '90%'
    },
    'table': {
        'width': '45%',
        'height': '90%',
        'overflowY': 'scroll',

    },
    'button': {
        'width': '10%',
    }

}

#generate HTML5 application
app.layout = html.Div([

    #data storage object
    dcc.Store(id='colorData', storage_type='memory'),

    #graph Div
    html.Div([
        dcc.Graph(
            id='polyGraph',
            figure=fig,
            style=styles['graph']
            ),
        html.Div([
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i}
                    for i in df.columns],
            data=df.to_dict('records'),
            style_cell=dict(textAlign='left'),
            style_header=dict(backgroundColor="paleturquoise"),
            style_data=dict(backgroundColor="lavender")
            ),
            ],
            style=styles['table'],
            )

        ],
        style=styles['bigApp']
        ),
        html.Div([
            html.Label("Group ID:", style={'margin':'0 10px 0 10px'}),
            dcc.Input(id="grp", style=styles['button']),
        ]
        )

    ],
    )

@app.callback(
    [Output('polyGraph', 'figure'),
     Output('table', 'data'),
     Output('colorData', 'data')],
    Input('polyGraph', 'selectedData'),
    [State('table', 'data'),
     State('grp', 'value'),
     State('colorData', 'data')]
    )
def color_selected_data(selectedData, tableData, group, colorData):
    """
    colors selected mesh cells based upon group ID
    """
    #selected data is None on page load, dont fire callback
    if selectedData is not None:
        #user must input a group ID
        if group == None:
            print("You must enter a value for group!")
            raise PreventUpdate

        #get mesh elements in selection
        ids = []
        for i,pt in enumerate(selectedData['points']):
            ids.append(pt['curveNumber'])

        #initialize colorData dictionary
        if colorData is None:
            colorData = {}

        #loop thru IDs of selected, assigning color by group
        ids = np.array(np.unique(ids))
        for ID in ids:
            if fig.data[ID].line.color == None:
                pass
            else:
                fig.data[ID].line.color = '#9834eb'
                if group == None:
                    group = 0

                #also update the table
                tableData[ID]['Group'] = group

                if group in colorData:
                    fig.data[ID].line.color = colorData[group]
                else:
                    colorData[group] = px.colors.qualitative.Plotly[len(colorData)]
                    fig.data[ID].line.color = colorData[group]

    return fig, tableData, colorData


if __name__ == '__main__':
    app.run_server(debug=True)
