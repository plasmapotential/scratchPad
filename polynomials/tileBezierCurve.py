#adjusts a tiles Bspline curves
import sys
import numpy as np
import plotly.graph_objects as go

FreeCADPath = '/lib/freecad-python3/lib'
sys.path.append(FreeCADPath)
import FreeCAD
import Import
import Part

length = 135.73661899718053
moveAmount = 10.0

names = [
        'AN_objects446',
        ]

STPfile = '/home/tlooby/results/bidirectionalSurfaces/olimTile.step'
stpOut = '/home/tlooby/results/bidirectionalSurfaces/olimNew.stp'
fcOut = '/home/tlooby/results/bidirectionalSurfaces/olimNew.FcStd'

CAD = Import.open(STPfile)
CADdoc = FreeCAD.ActiveDocument
CADobjs = CADdoc.Objects
CADparts = []
for obj in CADobjs:
    if type(obj) == Part.Feature:
        if obj.Label in names:
            CADparts.append(obj)


part = CADparts[0]
newPart = CADdoc.addObject("Part::Feature", "newPart")


eIdx = []
edges = []
for i,e in enumerate(part.Shape.Edges):
    if np.abs(e.Length - length) < 1e-6:
        eIdx.append(i)
        edges.append(e)




for j in eIdx:
    e = part.Shape.Edges[j]
    poles = np.array(e.Curve.getPoles())
    v0 = poles[-1] - poles[0]
    v1 = poles[1] - poles[0]
    v2 = np.cross(v0, v1)
    v3 = np.cross(v2, v0)
    v0N =v0 / np.linalg.norm(v0)
    v1N =v1 / np.linalg.norm(v1)
    v2N =v2 / np.linalg.norm(v2)
    v3N =v3 / np.linalg.norm(v3)
    mag = np.linalg.norm(v0)
    x = np.dot(poles, v0N)    
    y = np.dot(poles, v3N)
    xNew = x.copy()
    yNew = y.copy()
    xNew[1] = xNew[1] - moveAmount
    xNew[2] = xNew[2] + moveAmount
    dx = xNew - x
    #for plotting the control points
    #fig = go.Figure()
    #y-=min(y)
    #fig.add_trace(go.Scatter(x=x,y=y, name='old'))
    #fig.add_trace(go.Scatter(x=xNew,y=y, name='new'))
    #fig.show()
    #now put these new points back into the CAD model
    newPoles = np.zeros((np.array(poles).shape))
    for i,delta in enumerate(dx):
        newPoles[i,:] =  v0N * delta + np.array(poles[i])

    newP = [FreeCAD.Vector(v) for v in newPoles]

    old_curve = part.Shape.Edges[j].Curve
    degree = old_curve.Degree
    knots = old_curve.getKnots()
    multiplicities = old_curve.getMultiplicities()
    weights = old_curve.Weights if old_curve.isRational() else None
    new_curve = Part.BSplineCurve()
    new_curve.buildFromPolesMultsKnots(newP, multiplicities, knots, old_curve.isPeriodic(), degree, weights)
    new_edge = Part.Edge(new_curve)
    
    #print(new_edge.Curve.getPole(2))
    #print(part.Shape.Edges[j].Curve.getPole(2))    
    part.Shape.Edges[j] = new_edge

    affected_faces = [face for face in part.Shape.Faces if e.CenterOfMass in [fe.CenterOfMass for fe in face.Edges]]

    # 2.2. Reconstruct Those Faces
    new_faces = []
    for face in affected_faces:
        # Extract wires from the face
        wires = face.Wires

        # For each wire, check if it contains the old edge
        new_wires = []
        for wire in wires:
            comList = [fe.CenterOfMass for fe in wire.Edges]
            new = np.unique(np.where(e.CenterOfMass == np.array(comList))[0])
            old = np.unique(np.where(e.CenterOfMass != np.array(comList))[0])
            
            edges = [None]*len(wire.Edges)
            for i in range(len(edges)):
                if i in old:
                    edges[i] = wire.Edges[i]
                if i in new:
                    edges[i] = new_edge
                    #print('---')
                    #print(i)
                    #print(new_edge.Curve.getPole(2))
                    #print(edges[i].Curve.getPole(2))
                    #print('---')

#            if len(new) > 0:
#                edges[new[0]] = new_edge
            new_wire = Part.Wire(edges)
            new_wires.append(new_wire)

        # Create a new face using the new wires
        new_face = Part.Shape(new_wires)
        new_faces.append(new_face)

    # Combine new faces with unchanged faces
    all_faces = [face for face in part.Shape.Faces if face not in affected_faces] + new_faces

    # 2.3. Rebuild the Solid
    new_solid = Part.Solid(Part.Shell(all_faces))

    # Replace the old part shape with the new solid
    newPart.Shape = new_solid
    #print(newPart.Shape.Edges[j].Curve.getPole(2)) 
    #input()



#    v = FreeCAD.Vector(newPoles[1])
#    print(v) 
#    print(c.getPole(2))
#    old_curve.setPole(2, v)
#    print(c.getPole(2))
#    part.Shape.Edges[j].Curve = c
#    print(part.Shape.Edges[j].Curve.getPole(2))





#    print(v) 
#    print(part.Shape.Edges[j].Curve.getPole(2))
#    part.Shape.Edges[j].Curve.setPole(2,v) 
#    print(part.Shape.Edges[j].Curve.getPole(2))
#    input()
    


#print(poles)
#print(newPoles)
Import.export(newPart.Shape, stpOut)
#CADdoc.saveAs(fcOut)
