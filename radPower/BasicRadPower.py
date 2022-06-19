"""
calculates radiated power from a point source to a 10mm spherical mesh,
then uses that sphere as a source mesh.
Power is then distributed to a target mesh
"""
import numpy as np
import os,sys

#python libs and pvpython
PVPath = '/opt/paraview/ParaView-5.9.0-RC2-MPI-Linux-Python3.8-64bit/lib/python3.8/site-packages'
pvpythonCMD = '/opt/paraview/ParaView-5.9.0-RC2-MPI-Linux-Python3.8-64bit/bin/pvpython'
os.environ["PVPath"] = PVPath
os.environ["pvpythonCMD"] = pvpythonCMD
#append paraview to python path
sys.path.append(PVPath)

FreeCADPath = '/usr/lib/freecad-daily-python3/lib'
#append FreeCAD to python path
sys.path.append(FreeCADPath)
import FreeCAD
import Part
import Mesh

HEATPath = '/home/tom/source/HEAT/github/source'
dataPath = '/home/tom/source/test/radPower/'
sys.path.append(HEATPath)

import toolsClass
tools = toolsClass.tools()
tools.rootDir = HEATPath



bigSphere = '/home/tom/source/test/radPower/100mmSphere.stl'
smallSphere = '/home/tom/source/test/radPower/10mmSphere.stl'
planeFile1 = '/home/tom/source/test/radPower/30mmPlane.stl'
cubeFile = '/home/tom/source/test/radPower/100mmCube.stl'

class MeshObj:
    def __init__(self, file):
        """
        file is STL file
        """
        self.mesh = Mesh.Mesh(file)
        self.normsCentersAreas()
        return



    def faceNormals(self, mesh):
        """
        returns normal vectors for single freecad mesh object in cartesian
        coordinates
        """
        #face normals
        normals = []
        for i, facet in enumerate(mesh.Facets):
            vec = np.zeros((3))
            for j in range(3):
                vec[j] = facet.Normal[j]
            normals.append(vec)
        return np.asarray(normals)

    def faceAreas(self, mesh):
        """
        returns face areas for mesh element
        """
        #face area
        areas = []
        for i, facet in enumerate(mesh.Facets):
            areas.append(facet.Area)
        return np.asarray(areas)

    def faceCenters(self, x, y, z):
        """
        returns centers of freecad mesh triangle in cartesian coordinates
        """
        #face centers
        centers = np.zeros((len(x), 3))
        centers[:,0] = np.sum(x,axis=1)/3.0
        centers[:,1] = np.sum(y,axis=1)/3.0
        centers[:,2] = np.sum(z,axis=1)/3.0
        return centers


    def normsCentersAreas(self):
        """
        Gets face normals and face centers.  Both norms and centers are arrays
        of length mesh.CountFacets, consisting of three components (x,y,z) per
        facet
        """
        mesh = self.mesh
        N_facets = mesh.CountFacets
        x = np.zeros((N_facets,3))
        y = np.zeros((N_facets,3))
        z = np.zeros((N_facets,3))
        for i,facet in enumerate(mesh.Facets):
            #mesh points
            for j in range(3):
                x[i][j] = facet.Points[j][0]
                y[i][j] = facet.Points[j][1]
                z[i][j] = facet.Points[j][2]
        # get face normals and face centers (units in mm)
        self.norms = self.faceNormals(mesh)
        self.centers = self.faceCenters(x,y,z)
        self.areas = self.faceAreas(mesh) / 1000000.0 #m
        self.totalArea = np.sum(self.areas)
        return

    def writeHFpc(self,centers,hf,tag=None):
        pcfile = dataPath + 'HF.csv'
        pc = np.zeros((len(centers), 4))
        pc[:,0] = centers[:,0]
        pc[:,1] = centers[:,1]
        pc[:,2] = centers[:,2]
        pc[:,3] = hf
        head = "X,Y,Z,HF"
        np.savetxt(pcfile, pc, delimiter=',',fmt='%.10f', header=head)
        #Now save a vtk file for paraviewweb
        if tag is None:
            tools.createVTKOutput(pcfile, 'points', 'HF')
        else:
            name = 'HF_'+tag
            tools.createVTKOutput(pcfile, 'points', name)
        print("Created Point Cloud")
        return


class photonPointSource:
    def __init__(self,x0,y0,z0,P_rad):
        """
        initial location and magnitude of photon radiative pointSource
        x0,y0,z0 is location of point source
        P_rad is magnitude in MW
        """
        self.P_rad = P_rad
        self.centers = np.array([[x0,y0,z0]])
        return


def mapPhotonPower(source, target, verbose=False):
    """
    maps photon power from source to target mesh assuming each
    mesh center is a photon isotropic radative source

    """
    print("========== Assigning photon power to mesh ==========")
    #Radii magnitudes and vectors
    radiiMags = np.zeros((len(target.centers),len(source.centers)))
    radiiVecs = np.zeros((len(target.centers),3,len(source.centers)))
    pdotn = np.zeros((len(target.centers),len(source.centers)))
    for j in range(len(source.centers)):
        for i in range(len(target.centers)):
            radiiMags[i,j] = np.linalg.norm(target.centers[i] - source.centers[j])
            radiiVecs[i,:,j] = (target.centers[i] - source.centers[j]) / radiiMags[i,j]
            #calculate dot product between photon trajectory and face normals
            pdotn[i,j] = np.dot(radiiVecs[i,:,j], target.norms[i])

    radiiMags /= 1000.0 #mm to m

    #distribute power from small source to target mesh
    target.qRad = np.zeros((len(target.centers)))
    target.P_rad = np.zeros((len(target.centers)))
    #TO DO - THIS CAN PROBABLY BE CONVERTED TO A MATRIX OPERATION:
    for i in range(len(target.centers)):
        target.qRad[i] = np.sum( pdotn[i,:] * source.P_rad / (4 * np.pi * radiiMags[i,:]**2) )
        target.P_rad[i] = target.qRad[i] * target.areas[i]

    #print stuff
    if verbose==True:
        print("Number of source mesh elements: {:f}".format(len(source.centers)))
        print("Number of target mesh elements: {:f}".format(len(target.centers)))
        print("Target Summation Area: {:f}".format(target.totalArea))
        print("Example target qRad[0]: {:f} MW/m^2".format(target.qRad[0]))
        #print("Theoretical target qRad: {:f} MW/m^2".format(source.P_rad / totalArea ))
        print("Summation power on mesh: {:f} MW".format(np.sum(target.P_rad)))
    return target




#initialize mesh objects
bigR = 0.1 #m
big = MeshObj(bigSphere)
smallR = 0.01 #m
small = MeshObj(smallSphere)
plane1 = MeshObj(planeFile1)
plane1W = 0.03 #m
cube = MeshObj(cubeFile)


#initialize point source
PS = photonPointSource(0,0,0,10)

print("\nPS to Small Sphere:")
small = mapPhotonPower(PS, small, verbose=True)
#print("\nSmall Sphere to Big Sphere:")
#big = mapPhotonPower(small, big, verbose=True)
#print("\nPS to Plane:")
#plane1 = mapPhotonPower(PS, plane1, verbose=True)
#print("\nPS to Cube:")
#cube = mapPhotonPower(PS, cube, verbose=True)
print("\nSmall Sphere to Cube:")
cube = mapPhotonPower(small, cube, verbose=True)

cube.writeHFpc(cube.centers,cube.qRad,tag='cube')



#OLD METHODS DO NOT USE FUNCTIONS
#
##put point source onto small spherical mesh
#smallMask = False
#if smallMask == True:
#    #distribute point source power onto small source mesh
#    print("========== Assigning point source power to mesh ==========")
#    totalArea = np.sum(mesh.areas)
#    print("Small Summation Area: {:f}".format(totalArea/1000000)) #mm^2 to m^2
#    print("Small Theoretical Area: {:f}".format(4*np.pi*smallR**2))
#    small.P_rad = P_rad * small.areas / totalArea
#    print("Small Power Summation: {:f}".format(np.sum(small.P_rad)))
#    print("Small Theoretical Power {:f}".format(P_rad))
#    small.qRad = small.P_rad / (small.areas / 1000000.0) #mm^2 to m^2
#    small.writeHFpc(small.centers,small.qRad,tag='small')
#    print("Example source qRad[0] (from point source): {:f} MW/m^2".format(small.qRad[0]))
#    print("Theoretical source qRad (from point source): {:f} MW/m^2".format(P_rad / (totalArea/1000000) )) #mm^2 to m^2
#
##distribute small mesh source power onto big sphere mesh
#bigMask = False
#if bigMask == True:
#    print("\n========== Assigning power to large sphere ==========")
#    totalArea = np.sum(big.areas) /1000000.0 #m
#    print("Number of source mesh elements: {:f}".format(len(small.centers)))
#    print("Number of target mesh elements: {:f}".format(len(big.centers)))
#    print("Big Summation Area: {:f}".format(totalArea))
#    print("Big Theoretical Area: {:f}".format(4*np.pi*bigR**2))
#    radiiMatrix = np.zeros((len(big.centers),len(small.centers)))
#    for j in range(len(small.centers)):
#        for i in range(len(big.centers)):
#            radiiMatrix[i,j] = np.linalg.norm(big.centers[i] - small.centers[j]) / 1000.0 #mm to m
#
#    #distribute power from small source mesh to large mesh
#    big.qRad = np.zeros((len(big.centers)))
#    big.P_rad = np.zeros((len(big.centers)))
#    for i in range(len(big.centers)):
#        big.qRad[i] = np.sum(small.P_rad / (4 * np.pi * radiiMatrix[i,:]**2))
#        big.P_rad[i] = big.qRad[i] * big.areas[i] / 1000000.0 #mm^2 to m^2
#
#    print("Example target qRad[0]: {:f} MW/m^2".format(big.qRad[0]))
#    print("Theoretical target qRad: {:f} MW/m^2".format(P_rad / totalArea ))
#    print("Summation power on big: {:f} MW".format(np.sum(big.P_rad)))
#
#    small.writeHFpc(big.centers,big.qRad,tag='big')
#
##distribute small mesh source power onto 30mm plane
#planeMask = False
#if planeMask == True:
#    print("\n========== Assigning power to large sphere ==========")
#    totalArea = np.sum(plane1.areas) /1000000.0 #m
#    print("Number of source mesh elements: {:f}".format(len(small.centers)))
#    print("Number of target mesh elements: {:f}".format(len(big.centers)))
#    print("Plane1 Summation Area: {:f}".format(totalArea))
#    print("Plane1 Theoretical Area: {:f}".format(plane1W**2))
#    radiiMatrix = np.zeros((len(plane1.centers),len(small.centers)))
#    for j in range(len(small.centers)):
#        for i in range(len(plane1.centers)):
#            radiiMatrix[i,j] = np.linalg.norm(plane1.centers[i] - small.centers[j]) / 1000.0 #mm to m
#
#    #distribute power from small source mesh to plane mesh
#    plane1.qRad = np.zeros((len(plane1.centers)))
#    plane1.P_rad = np.zeros((len(plane1.centers)))
#    for i in range(len(plane1.centers)):
#        plane1.qRad[i] = np.sum(small.P_rad / (4 * np.pi * radiiMatrix[i,:]**2))
#        plane1.P_rad[i] = plane1.qRad[i] * plane1.areas[i] / 1000000.0 #mm^2 to m^2
#
#    print("Example target qRad[0]: {:f} MW/m^2".format(plane1.qRad[0]))
#    print("Theoretical target qRad: {:f} MW/m^2".format(P_rad / totalArea ))
#    print("Summation power on plane: {:f} MW".format(np.sum(plane1.P_rad)))
#
#    small.writeHFpc(plane1.centers,plane1.qRad,tag='plane1')
#
