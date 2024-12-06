#Description:   file for testing obj mesh import / export functionality
#Date:          20241030
#engineer:      T Looby
import os, sys
import numpy as np
from pathlib import Path
FreeCADPath = '/usr/lib/freecad-python3/lib'
#FreeCADPath = '/opt/freecad/squashfs-root/usr/lib'
sys.path.append(FreeCADPath)
import FreeCAD
import Mesh


doc = FreeCAD.newDocument('testDoc')

meshFile = '/home/tlooby/Downloads/test.obj'

loaded_mesh = Mesh.Mesh(meshFile)


# Add the loaded mesh to the active document as a new object
mesh_object = doc.addObject("Mesh::Feature", "ImportedMesh")
mesh_object.Mesh = loaded_mesh
FreeCAD.ActiveDocument.recompute()