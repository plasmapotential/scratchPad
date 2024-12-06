#truncates a vtk field line trace so that it is shorter.
#saves truncated version as another vtk
import vtk

# Function to read the VTK file
def read_vtk_file(filename):
    reader = vtk.vtkPolyDataReader()  # Use vtkPolyDataReader for .vtk files
    reader.SetFileName(filename)
    reader.Update()
    
    data = reader.GetOutput()

    # Check if data is loaded properly
    if data is None or data.GetNumberOfPoints() == 0:
        raise ValueError(f"Error reading VTK file: {filename}")
    
    print(f"Number of points: {data.GetNumberOfPoints()}")
    print(f"Number of cells: {data.GetNumberOfCells()}")

    return data

# Function to write the truncated VTK file
def write_vtk_file(data, output_filename):
    writer = vtk.vtkPolyDataWriter()  # Use vtkPolyDataWriter for .vtk files
    writer.SetFileName(output_filename)
    writer.SetInputData(data)
    writer.Write()

# Function to truncate the field line trace
def truncate_trace(data, max_index):
    # Create a new vtkPoints and vtkCellArray for the truncated trace
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Truncate the points and lines
    for i in range(max_index + 1):
        points.InsertNextPoint(data.GetPoints().GetPoint(i))
    
    for cell_id in range(data.GetNumberOfCells()):
        cell = data.GetCell(cell_id)
        cell_points = cell.GetPointIds()
        
        # Only include cells that are within the truncated range
        if cell_points.GetId(0) <= max_index and cell_points.GetId(1) <= max_index:
            cells.InsertNextCell(cell)

    # Create a new PolyData object with the truncated points and lines
    truncated_data = vtk.vtkPolyData()
    truncated_data.SetPoints(points)
    truncated_data.SetLines(cells)

    return truncated_data


# Main script
path = "/home/tlooby/Downloads/"
input_filename = path + "Field_trace_pt002.vtk"  # Replace with your file path
output_filename = path + "output_trace_truncated.vtk"  # Output file path

# Read the input file
data = read_vtk_file(input_filename)

# Truncate the trace at index 2939
truncated_data = truncate_trace(data, 2939)

# Write the truncated data to a new file
write_vtk_file(truncated_data, output_filename)

print(f"Truncated trace saved to {output_filename}")