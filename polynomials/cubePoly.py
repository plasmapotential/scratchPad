#creates a polynomial surface in freecad.  meant to be pasted into FreeCAD
#python terminal
import numpy as np


# Define the polynomial function
def polynomial(x, c):
    p = c[0] + c[1]*x + c[2]*x**2
    return p

doc = App.ActiveDocument


# Create a cube
cube = doc.addObject("Part::Box", "Cube")
cube.Length = 10
cube.Width = 10
cube.Height = 10

# Coefficients for the polynomial
c = np.array([10.0, 0.0, -0.05, 0.0])

# Generate a series of vertical rectangles based on the polynomial values
solids = []
for x in range(10):
    p1 = Base.Vector(x, 0, polynomial(x, c))
    p2 = Base.Vector(x+1, 0, polynomial(x+1, c))
    p3 = Base.Vector(x+1, 10, polynomial(x+1, c))
    p4 = Base.Vector(x, 10, polynomial(x, c))
    wire = Part.makePolygon([p1, p2, p3, p4, p1])
    face = Part.Face(wire)
    solid = face.extrude(Base.Vector(0, 0, -11.0))
    solids.append(solid)

# Union the solids together
final_solid = solids[0]
for s in solids[1:]:
    final_solid = final_solid.fuse(s)

#doc.addObject("Part::Feature", "poly").Shape = final_solid

# Get the common part between the cube and the final solid
result = cube.Shape.common(final_solid)

# Add the result to the document
doc.addObject("Part::Feature", "PolynomialCutOut").Shape = result

# Duplicate the result object
duplicate_result = result.copy()

# Rotate the duplicate by 180° about the Z-axis
rotate_center = Base.Vector(5, 5, 5)  # center of the cube
rotate_axis = Base.Vector(0, 0, 1)  # Z-axis
duplicate_result.rotate(rotate_center, rotate_axis, 180)

# Translate the rotated object to position it correctly
# Since the object is rotated, we need to move it by twice its width to align it on the YZ plane.
duplicate_result.translate(Base.Vector(-10, 0, 0))  # translate by cube's length

# Fuse the original and the rotated object
final_object = result.fuse(duplicate_result)

# Add the final object to the document
doc.addObject("Part::Feature", "FinalObject").Shape = final_object