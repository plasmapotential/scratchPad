#converts and STL mesh to an FBX mesh for UE5
import bpy
import os

def stl_to_fbx(stl_path, fbx_path):
    print("Clearing Data...")
    # Clear all data from the current blend file to start fresh
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print("Importing Mesh...")
    # Import the STL
    bpy.ops.import_mesh.stl(filepath=stl_path)
    print("Exporting FBX")
    # Export the object as FBX
    bpy.ops.export_scene.fbx(filepath=fbx_path, use_selection=True)
    print("Complete.")

if __name__ == "__main__":
    input_stl = "/home/tlooby/projects/UE5/T4.stl"
    output_fbx = "/home/tlooby/projects/UE5/T4.fbx"
    stl_to_fbx(input_stl, output_fbx)
