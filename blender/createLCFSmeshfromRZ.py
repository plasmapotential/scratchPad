#blender macros to create a mesh from LCFS contour


#import object
import bpy
import csv

# Adjust the file path to your CSV file
file_path = "path/to/your/contour.csv"

# Clear existing mesh data
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Create a new mesh
mesh = bpy.data.meshes.new("ContourMesh")
obj = bpy.data.objects.new("ContourObject", mesh)
bpy.context.collection.objects.link(obj)

verts = []
edges = []
faces = []

with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for i, row in enumerate(csv_reader):
        x, y, z = map(float, row)
        verts.append((x, y, z))

mesh.from_pydata(verts, edges, faces)
bpy.context.view_layer.objects.active = obj


#spin object
import bpy
import bmesh

# Define parameters
angle = 360
steps = 32
axis = (0, 0, 1)  # Z-axis
center = (0, 0, 0)

# Get the active mesh
obj = bpy.context.object
mesh = bmesh.from_edit_mesh(obj.data)

# Spin
bmesh.ops.spin(mesh,
               geom=mesh.verts[:] + mesh.edges[:] + mesh.faces[:],
               cent=center,
               axis=axis,
               dvec=(0, 0, 0),
               angle=angle * (3.14159 / 180.0),
               steps=steps)

bmesh.update_edit_mesh(obj.data)
