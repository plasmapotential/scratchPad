#creates a FEM mesh from stl using freecad api
import sys 

#FreeCADPath = '/usr/lib/freecad-python3/lib'
FreeCADPath = '/home/tlooby/opt/freecad/squashfs-root/usr/lib'
sys.path.append(FreeCADPath)
import FreeCAD
import Mesh
import Fem

# Import the dummy module so we dont have to import open3d
import dummy_open3d
sys.modules['open3d'] = dummy_open3d


HEATPath = '/home/tlooby/source/HEAT/github/source'
sys.path.append(HEATPath)
import CADClass




# Path to your STL file
path_to_stp = '/home/tlooby/projects/elmer/testElmer2/test.step'
#output directory
outDir = '/home/tlooby/projects/elmer/testElmer2/'
meshOut1 = outDir + 'test.unv'
meshOut2 = outDir + 'test.vtk'

CAD = CADClass.CAD()
CAD.STPfile = path_to_stp
CAD.permute_mask = False
CAD.loadSTEP()

obj = CAD.CADobjs[0]

#build FEM mesh using netgen algorithm
m = CAD.createFEMmeshNetgen(obj)

#build FEM mesh using gmsh algorithm
#CAD.createFEMmeshGmsh(obj)

#save mesh as unv file
print("Saving mesh file: "+meshOut1)
Fem.export([m], meshOut1)

#save mesh as vtk file
print("Saving mesh file: "+meshOut2)
Fem.export([m], meshOut2)







