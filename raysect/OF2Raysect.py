#converts paraview exported OF mesh to obj format for use by raysect

#f = 'test_1240.vtm'

import pyvista as pv
f = '/home/tlooby/results/boundaryOps/T4slice/test_1240.vtm'
grid = pv.read(f)
gridXYZ = grid[0].points
temperature = grid[0].point_data['T']
print(temperature)
print(gridXYZ)