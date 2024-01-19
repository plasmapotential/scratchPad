#creates a FEM mesh from stl using MeshPy.  courtesy of chatgpt

import meshpy.tet
import numpy as np
import os

def read_stl(filename):
    """Reads an STL file and returns the vertices and faces."""
    import numpy as np
    import stl
    from stl import mesh

    # Load mesh from file
    your_mesh = mesh.Mesh.from_file(filename)
    vertices = np.unique(np.vstack([your_mesh.v0, your_mesh.v1, your_mesh.v2]), axis=0)
    faces = []
    for f in range(len(your_mesh.v0)):
        face = []
        for v in (your_mesh.v0[f], your_mesh.v1[f], your_mesh.v2[f]):
            idx = np.where(np.all(vertices == v, axis=1))[0][0]
            face.append(idx)
        faces.append(face)
    
    return vertices, faces

def generate_tet_mesh(vertices, faces):
    """Generates a tetrahedral mesh using MeshPy from given vertices and faces."""
    
    # Define the facet markers
    facet_markers = np.zeros(len(faces), dtype=np.int)

    # Define the tetrahedral mesh info
    info = meshpy.tet.MeshInfo()
    info.set_points(vertices)
    info.set_facets(faces, facet_markers=facet_markers)

    # Generate the mesh
    mesh = meshpy.tet.build(info, max_volume=1e-2, min_ratio=1.1, attributes=True)

    return mesh

def save_mesh(mesh, filename):
    """Saves the mesh to a file in VTK format."""
    mesh.write_vtk(filename)

# Path to your STL file
stl_file = '/home/tlooby/projects/elmer/testElmer2/test.stl'

outDir = '/home/tlooby/projects/elmer/testElmer2/'

# Output filename for the tetrahedral mesh
output_file = outDir + "test.vtk"
print("Starting...")

# Read the STL file
vertices, faces = read_stl(stl_file)
print("Generating Tet mesh...")
# Generate the tetrahedral mesh
tet_mesh = generate_tet_mesh(vertices, faces)

# Save the tetrahedral mesh to a file
save_mesh(tet_mesh, output_file)

print(f"Mesh generated and saved to {output_file}")
