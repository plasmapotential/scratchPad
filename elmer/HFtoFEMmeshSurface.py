#HFtoFEMmeshSurface.py
#Engineer: TL
#Date: 20230201
#imports a HF from HEAT csv and interpolates onto FEM mesh surface
#supports output for XYZ coordinates of each surface node, or the
#node id

import pandas as pd
import sys 
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import gmsh

#output directory
rootDir = '/home/tlooby/projects/elmer/testElmer3/'
meshFile = rootDir + 'part_auto.unv'
hfFile = rootDir + 'HF_optical.csv'
hfXYZOut = rootDir + 'xyzHF.dat'
hfNodeOut = rootDir + 'nodehf_auto.dat'

hf_data = pd.read_csv(hfFile)

# Extracting coordinates and heat flux from the CSV file
points = hf_data[['# X', 'Y', 'Z']].values
values = hf_data['$MW/m^2$'].values

gmsh.initialize()
gmsh.open(meshFile)  # Load your mesh file

node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_tags = np.array(node_tags)
node_coords = np.array(node_coords).reshape(-1, 3)
boundary_nodes = set()

# Get the entities of the dimension you are interested in
entities = gmsh.model.getEntities(2)  # 2 for surfaces

for entity in entities:
    dim, tag = entity

    # Get the nodes of this surface entity
    _, _, surface_node_tags = gmsh.model.mesh.getElements(dim, tag)
    for node_tag_array in surface_node_tags:
        for node_tag in node_tag_array:
            boundary_nodes.add(node_tag)  # Use add() for single elements


# Extract the coordinates of boundary nodes
boundaryPts = np.array([node_coords[np.where(node_tags == node_tag)[0][0]] for node_tag in boundary_nodes])

# Perform the interpolation
print("Interpolating...")
hfOnMesh = griddata(points, values, boundaryPts, method='nearest')
hfArray = np.hstack((boundaryPts, hfOnMesh[:,np.newaxis]))
nodes = np.array(list(boundary_nodes), dtype=int)
nodeArray = np.vstack([np.array(list(boundary_nodes), dtype=int), hfOnMesh*1e6]).T

#for saving X,Y,Z,HF
np.savetxt(hfXYZOut, hfArray, fmt="% .9E", delimiter=' ')
#for saving nodeId,HF
np.savetxt(hfNodeOut, nodeArray, fmt="% .9E", delimiter=',')

#fig = go.Figure()
#fig.add_trace(go.Scatter3d(x=ctrs[:,0], y=ctrs[:,1], z=ctrs[:,2], mode='markers'))
#fig.update_scenes(aspectmode='data')
#fig.show()
gmsh.finalize()



#threads
# https://www.elmerfem.org/forum/viewtopic.php?t=1451


sifFile = rootDir + 'bc.sif'
with open(sifFile, 'w') as f:
     
    for i in range(len(nodeArray)):
        f.write("\nBoundary Condition {:d}\n".format(i+1))
        f.write("Target Nodes(1) = {:d}\n".format(nodes[i]))
        #f.write("Name = \"Boundary Condition {:d}\" \n".format(i+1))
        f.write("Name = \"Heat Flux\"\n")
        f.write("Heat Flux = {:0.4f}\n".format(nodeArray[i,1]))
        f.write("End\n")
