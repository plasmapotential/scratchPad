#crossSection2Mesh.py
#Description:   get toroidal cross section from CAD STP file, and create 2D mesh
#Engineer:      T Looby
#Date:          20220405
import json
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, LinearRing
import plotly.graph_objects as go
import os
import sys

#user inputs
rMax = 5000
zMax = 3000
phi = 90.0 #degrees

rootPath = '/home/tom/work/CFS/projects/reducedCAD/vacVes'

#testCase
#STPfile = '/home/tom/work/CFS/projects/reducedCAD/HEAT/testCAD.step'
#real CAD
STPfile = '/home/tom/work/CFS/projects/reducedCAD/vacVes/vacVes.step'

#outputs
STPout2D = '/home/tom/work/CFS/projects/reducedCAD/vacVes/vacVesout2D.step'
pTableOut = '/home/tom/work/CFS/projects/reducedCAD/vacVes/pTable.csv'

#HEAT path
HEATpath = '/home/tom/source/HEAT/github/source'
sys.path.append(HEATpath)

#load HEAT environment
import launchHEAT
launchHEAT.loadEnviron()

#load HEAT CAD module and STP file
import CADClass
CAD = CADClass.CAD(os.environ["rootDir"], os.environ["dataPath"])
CAD.STPfile = STPfile
print(CAD.STPfile)
CAD.permute_mask = False
print("For large CAD files, loading may take a few minutes...")
CAD.loadSTEP()

#get poloidal cross section at user specified toroidal angle
print("Number of part objects in CAD: {:d}".format(len(CAD.CADparts)))
slices = CAD.getPolCrossSection(rMax,zMax,phi)
print("Number of part objects in section: {:d}".format(len(slices)))

#for saving output
save2DSTEP = False
if save2DSTEP == True:
    CAD.saveSTEP(STPout2D, slices)

#save CSV of points
saveCSV = False
if saveCSV == True:
    xyz = np.array([ [v.X, v.Y, v.Z] for v in slices[0].Shape.Vertexes])
    R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
    Z = xyz[:,2]
    rz = np.vstack([R,Z]).T
    f = rootPath + '/slice{:03d}.csv'.format(i)
    head = 'R[mm], Z[mm]'
    np.savetxt(f, rz, delimiter=',', header=head)

#build an ordered list of vertices that comprise a contour
contourList = []
edgeList = []
for slice in slices:
    edgeList = CAD.getVertexesFromEdges(slice.Shape.Edges)
    contours = CAD.findContour(edgeList)
    contourList.append(contours)

#plot the contours
contourPlot = False
if contourPlot == True:
    fig = go.Figure()
    for slice in contourList:
        for c in slice:
            R = np.sqrt(c[:,0]**2+c[:,1]**2)
            Z = c[:,2]
            fig.add_trace(go.Scatter(x=R, y=Z, mode='lines+markers'))

    fig.show()

#create a mesh over the contours
createMesh = True
if createMesh == True:

    #we will tesselate with rectangles / squares here
    def square(x, y, s):
        return Polygon([(x, y), (x+s, y), (x+s, y+s), (x, y+s)])

    #which contour from the list are we meshing
    c = contourList[0]

    #get the outer boundary and any holes inside
    outerContour = c[1]
    holeContour = c[0]

    R_out = np.sqrt(outerContour[:,0]**2+outerContour[:,1]**2)
    Z_out = outerContour[:,2]

    R_hole = np.sqrt(holeContour[:,0]**2+holeContour[:,1]**2)
    Z_hole = holeContour[:,2]
    #create a ring of the hole
    holeRing = LinearRing(np.vstack([R_hole,Z_hole]).T)

    #use multipoint
    #poly = MultiPoint(np.vstack([R_out,Z_out]).T).convex_hull
    #use polygon
    poly = Polygon(np.vstack([R_out,Z_out]).T, [holeRing])
    polyCoords = np.array(poly.exterior.coords)

    grid_size = 200
    ibounds = np.array(poly.bounds)//grid_size
    ibounds[2:4] += 1
    xmin, ymin, xmax, ymax = ibounds*grid_size
    xrg = np.arange(xmin, xmax, grid_size)
    yrg = np.arange(ymin, ymax, grid_size)
    mp = MultiPolygon([square(x, y, grid_size) for x in xrg for y in yrg])
    solution = MultiPolygon(list(filter(poly.intersects, mp)))


    #now create parallelogram table
    pTable = np.zeros((len(solution.geoms), 7))
    for i,geom in enumerate(solution.geoms):
        #Rc
        pTable[i,0] = np.array(geom.centroid)[0] *1e-3 #to meters
        #Zc
        pTable[i,1] = np.array(geom.centroid)[1] *1e-3 #to meters
        #L
        pTable[i,2] = grid_size *1e-3 #to meters
        #w
        pTable[i,3] = grid_size *1e-3 #to meters
        #AC1
        pTable[i,4] = 0.0
        #AC2
        pTable[i,5] = 0.0

    #save parallelogram table
    print("Saving Parallelogram Table...")
    print(pTableOut)
    head = 'Rc[m], Zc[m], L[m], W[m], AC1[deg], AC2[deg], GroupID'
    np.savetxt(pTableOut, pTable, delimiter=',',fmt='%.10f', header=head)



dashApp = True
if dashApp == True:
    #plot the boundary
    fig = go.Figure(go.Scatter(x=polyCoords[:,0], y=polyCoords[:,1], fill="toself"))
    fig.add_trace(go.Scatter(x=R_hole, y=Z_hole))
    #plot the mesh
    for geom in solution.geoms:
        xs, ys = np.array(geom.exterior.xy)
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", line=dict(color="seagreen")))


#    #create df for table if you dont have file
#    IDs = np.arange(len(solution.geoms))
#    groups = np.zeros((len(IDs)))
#    data = np.vstack([IDs, groups]).T
#    df = pd.DataFrame(data, columns=['ID', 'Group'])
    df = pd.read_csv(pTableOut)
    df.columns = df.columns.str.strip()


    fig.update_layout(clickmode='event+select')
    fig.update_layout(showlegend=False)
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)

    #DASH server
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
            'height': '95vh',
            'vertical-align': 'middle',
            'justify-content': 'center',
        },
        'column': {
            'width': '45%',
            'height': '100%',
            'display': 'flex',
            'flex-direction': 'column',
            'justify-content': 'center',

        },
        'graph': {
    #        'display': 'flex',
            'width': '100%',
            'height': '90%',
            'justify-content': 'center',
        },
        'btnRow': {
            'height': '10%',
            'width': '100%',
            'justify-content': 'center',
        },
        'button': {
            'width': '10%',
            'justify-content': 'center',
        },
        'table': {
            'width': '45%',
            'height': '90%',
            'overflowY': 'scroll',

        },



    }

    #generate HTML5 application
    app.layout = html.Div([

        #data storage object
        dcc.Store(id='colorData', storage_type='memory'),

        #graph Div
        html.Div([
            html.Div([
                dcc.Graph(
                    id='polyGraph',
                    figure=fig,
                    style=styles['graph']
                ),
                html.Div([
                    html.Label("Group ID:", style={'margin':'0 10px 0 10px'}),
                    dcc.Input(id="grp", style=styles['button']),
                    ],
                    style=styles['btnRow']
                    ),
                ],
                style=styles['column']
                ),
            html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i}
                        for i in df.columns],
                data=df.to_dict('records'),
                export_format="csv",
                style_cell=dict(textAlign='left'),
                style_header=dict(backgroundColor="paleturquoise"),
                style_data=dict(backgroundColor="lavender")
                ),
                ],
                style=styles['table'],
                ),
            ],
            style=styles['bigApp']
            ),

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
                    try:
                        tableData[ID]['GroupID'] = group
                    except: #contour traces will not have tableData
                        print("Group ID "+str(ID)+" not found in table!")

                    if group in colorData:
                        fig.data[ID].line.color = colorData[group]
                    else:
                        colorData[group] = px.colors.qualitative.Plotly[len(colorData)]
                        fig.data[ID].line.color = colorData[group]

        return fig, tableData, colorData


if __name__ == '__main__':
    app.run_server(debug=True)

#generate faces from wires and edges
#createFace = False
#if createFace == True:
#    faces = []
#    for i,slice in enumerate(slices):
#        print('{:d}================='.format(i))
#        print(slice.Shape.SubShapes)
#        tmpObj = CAD.CADdoc.addObject("Part::Feature", "Obj{:d}".format(i))
##       try:
##           tmpObj.Shape = CAD.createWire(slice.Shape.SubShapes)
##       except:
##           continue
#        tmpObj.Shape = CAD.createWire(slice.Shape)
#        CAD.CADdoc.addObject("Part::Face", "Face{:d}".format(i)).Sources = (tmpObj,)
#        faces.append(CAD.CADdoc.Objects[-1])
#        CAD.CADdoc.recompute()
#
#    #save step with faces
#    CAD.saveSTEP(rootPath + 'testOutRealCAD2.stp', faces)
