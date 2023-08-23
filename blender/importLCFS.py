#script to import LCFS data to blender.
#paste this script into blender scripting window
import bpy
import os
import numpy as np

# Replace the path below with the path to your CSV file
f = "/home/tlooby/source/tomTest/blender/lcfs.csv"

# Read the CSV data
RZ = np.genfromtxt(f, comments="#", delimiter=',',autostrip=True)
RZ = [(x,y,z,1) for x,y,z in RZ]

# Create a new curve object
curve_data = bpy.data.curves.new('LCFS_Curve', type='CURVE')
curve_data.dimensions = '2D'
curve_data.resolution_u = 2

# Create a new spline within the curve object
polyline = curve_data.splines.new('POLY')
polyline.points.add(len(RZ) - 1)

# Assign the CSV data to the spline points
for i, point in enumerate(polyline.points):
    point.co = RZ[i]

# Create a new object using the curve data
curve_obj = bpy.data.objects.new('LCFS_Curve', curve_data)

# Link the curve object to the current collection
bpy.context.collection.objects.link(curve_obj)

# Set the new curve object as the active object
bpy.context.view_layer.objects.active = curve_obj
curve_obj.select_set(True)