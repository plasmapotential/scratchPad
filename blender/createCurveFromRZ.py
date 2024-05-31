import bpy
import csv

# Adjust the file path to your CSV file
file_path = "/home/tlooby/projects/UE5/lcfs.csv"
# Create a new curve
curve_data = bpy.data.curves.new('MyCurve', type='CURVE')
curve_data.dimensions = '3D'
curve_data.resolution_u = 2

# Create a new object with the curve data
curve_object = bpy.data.objects.new('MyCurveObject', curve_data)
bpy.context.collection.objects.link(curve_object)

# Create a new spline in the curve
spline = curve_data.splines.new(type='POLY')

points=[]
with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for i, row in enumerate(csv_reader):
        if i>0:
            x, y, z = map(float, row)
            points.append((x, y, z))

spline.points.add(len(points) - 1)  # Add points to the spline
for i, point in enumerate(points):
    x, y, z = point
    spline.points[i].co = (x, y, z, 1)  # Add the point with w=1

# Update the scene
bpy.context.view_layer.update