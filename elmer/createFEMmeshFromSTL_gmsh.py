#creates a FEM mesh from stl using gmsh.  courtesy of chatgpt

import gmsh

# Path to your STL file
path_to_stl = '/home/tlooby/projects/elmer/testElmer2/test.stl'
#output directory
outDir = '/home/tlooby/projects/elmer/testElmer2/'


# Initialize Gmsh
gmsh.initialize()

# Set the model name and add a new model
gmsh.model.add("model_from_stl")

# Import the STL file
#surfaces, volumeTags = gmsh.model.occ.importShapes(path_to_stl)

# Merge the STL file
gmsh.merge(path_to_stl)


# Synchronize necessary before meshing
gmsh.model.occ.synchronize()


# Get surface entities
surfaces = gmsh.model.getEntities(2)
print(len(surfaces))
# If there's only one surface, try creating a volume directly from it
if len(surfaces) > 1:
    volume = gmsh.model.occ.addVolume([surfaces[0][1]])
    print(f"Volume created with tag: {volume}")
else:
    print("More than one surface entity detected. Expected only one for a simple STL file.")

# Synchronize the model after defining the volume
gmsh.model.occ.synchronize()

# Generate 3D mesh (volume mesh)
gmsh.model.mesh.generate(3)

# Save the mesh to msh format
#gmsh.write("test.msh")
# Optional: Save the mesh to other formats, e.g., .unv for use in Elmer
gmsh.write(outDir + "model_from_stl.unv")

# Finalize Gmsh
gmsh.finalize()
