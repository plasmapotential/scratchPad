#converts a vtk field line trace to an STL file
import vtk

# Function to read the VTK file
def read_vtk_file(filename):
    reader = vtk.vtkPolyDataReader()  # Use vtkPolyDataReader for .vtk files
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# Function to triangulate the data
def triangulate_polydata(data):
    # Ensure that all polygons are triangles
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(data)
    triangulator.Update()
    return triangulator.GetOutput()

# Function to write the STL file
def write_stl_file(data, output_filename):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(data)
    writer.Write()

def convert_lines_to_tubes(data, radius=2.0):
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputData(data)
    tube_filter.SetRadius(radius)
    tube_filter.SetNumberOfSides(12)  # Adjust for more detail if needed
    tube_filter.Update()
    return tube_filter.GetOutput()


# Main script
path = "/home/tlooby/Downloads/"
input_filename = path + "output_trace_truncated.vtk"   # Replace with your VTK file path
output_filename = path + "output_trace.stl"  # Output STL file path

# Read the input VTK file
data = read_vtk_file(input_filename)

# Triangulate the polydata to create a proper mesh
#triangulated_data = triangulate_polydata(data)

# Convert the lines to tubes
tube_data = convert_lines_to_tubes(data)

# Write the triangulated data to an STL file
write_stl_file(tube_data, output_filename)

print(f"STL file saved to {output_filename}")


