"""
reads a csv file of an axisymmetric radiation profile
creates a 3d qRad mesh from a 2D profile
revolves axisymmetric profile about Z axis
"""
import numpy as np
import pandas as pd
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
cubeFile1 = '/home/tom/source/test/radPower/100mmCube.stl'
cubeFile2 = '/home/tom/source/test/radPower/100mmCubeLower.stl'
radFile = '/home/tom/source/test/radPower/qRad_basic.dat'

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

class photonSource:
    def __init__(self,centers,P_rad):
        """
        initial location and magnitude of photon radiative pointSource
        centers is (x,y,z) is location of point source(s)
        P_rad is magnitude, indexed to match centers
        """
        self.P_rad = np.asarray(P_rad)
        self.centers = np.asarray(centers)
        return

    def write_photon_glyph(self,centers,pVec, dataPath, tag=None):
        """
        In paraview use TableToPoints => Calculator => Glyph
        Calculator should have this formula:
        (iHat*Px) + (jHat*Py) + (kHat*Pz)
        """
        print("Creating Photon Vector Point Cloud")
        #log.info("Creating Photon Vector Point Cloud")
        if tag is None:
            pcfile = dataPath + 'photon_glyph.csv'
        else:
            pcfile = dataPath + 'photon_glyph_'+tag+'.csv'
        pc = np.zeros((len(centers), 6))
        pc[:,0] = centers[:,0]
        pc[:,1] = centers[:,1]
        pc[:,2] = centers[:,2]
        pc[:,3] = pVec[:,0]
        pc[:,4] = pVec[:,1]
        pc[:,5] = pVec[:,2]
        head = "X,Y,Z,Px,Py,Pz"
        np.savetxt(pcfile, pc, delimiter=',',fmt='%.10f', header=head)
        #np.savetxt('points.asc', pc[:,:-1], delimiter=' ',fmt='%.10f')
        #print("Wrote point cloud file: " + pcfile)
        if tag is None:
            tools.createVTKOutput(pcfile, 'glyph', 'photon_glyph')
        else:
            name = 'photon_glyph_'+tag
            tools.createVTKOutput(pcfile, 'glyph', name)
        return



def mapPhotonPower(source, target, verbose=False):
    """
    maps photon power from source to target mesh assuming each
    mesh center is a photon isotropic radative source

    """
    print("========== Assigning photon power to mesh ==========")
    Nt = len(target.centers)
    Ns = len(source.centers)
    #Radii magnitudes and vectors
    radiiMags = np.zeros((Nt,Ns))
    radiiVecs = np.zeros((Nt,3,Ns))
    pdotn = np.zeros((Nt,Ns))
    for j in range(Ns):
        for i in range(Nt):
            radiiMags[i,j] = np.linalg.norm(target.centers[i] - source.centers[j])
            radiiVecs[i,:,j] = (target.centers[i] - source.centers[j]) / radiiMags[i,j]
            #calculate dot product between photon trajectory and face normals
            pdotn[i,j] = np.dot(radiiVecs[i,:,j], target.norms[i])

    #backface culling
    shadowMask = np.zeros((Nt,Ns), dtype=bool)
    shadowMask[np.where(pdotn > 0.0)] = 1

    radiiMags /= 1000.0 #mm to m

    #distribute power from source to target mesh
    source.P_radMatrix = np.repeat(source.P_rad[np.newaxis,:], Nt, axis=0)
    #with backface culling
    target.qRadMatrix = ~shadowMask * np.abs(pdotn) * source.P_radMatrix / (4*np.pi*radiiMags**2)
    #without backface culling
    #target.qRadMatrix = np.abs(pdotn) * source.P_radMatrix / (4*np.pi*radiiMags**2)
    target.qRad = np.sum(target.qRadMatrix, axis=1)
    target.P_rad = target.qRad * target.areas

    #print stuff
    if verbose==True:
        print("Number of source mesh elements: {:f}".format(len(source.centers)))
        print("Number of target mesh elements: {:f}".format(len(target.centers)))
        print("Target Summation Area: {:f}".format(target.totalArea))
        print("Example target qRad[0]: {:f} MW/m^2".format(target.qRad[0]))
        #print("Theoretical target qRad: {:f} MW/m^2".format(source.P_rad / totalArea ))
        print("Summation power on mesh: {:f} MW".format(np.sum(target.P_rad)))

    return target

def readRadFile(radFile, scaleBy=1.0):
    """
    reads a photon radiation power .dat or .csv file
    file contains R,Z,qRad[MW/m^2] on each line
    """
    data = pd.read_csv(radFile,header=0)
    arr = data.values
    arr[:,0]*=scaleBy
    arr[:,1]*=scaleBy
    return arr

def revolve2Dprofile(profile2D, Ntor, phiMin=0.0, phiMan=2*np.pi):
    """
    revolves a 2D profile about the z axis to generate a 3D profile
    assumes that the profile is axisymmetric.
    user specifies integer Ntor, the number of discrete steps in toroidal direction
    3D matrix is length: len(qRad2D) * Ntor
    3D matrix is in cylindrical coordinates: R,Phi,Z
    """
    phi = np.linspace(phiMin,phiMax,Ntor+1)[:-1] #don't return 0 and 2pi, only one of them
    profile3D = np.zeros((len(profile2D)*Ntor, 4))
    profile3D[:,0] = np.repeat(profile2D[:,0],Ntor) #R
    profile3D[:,1] = np.repeat(phi[np.newaxis,:],len(profile2D),axis=0).ravel()#phi
    profile3D[:,2] = np.repeat(profile2D[:,1],Ntor) #Z
    profile3D[:,3] = np.repeat(profile2D[:,2],Ntor) #qRad
    return profile3D


qRad2D = readRadFile(radFile, scaleBy=1000.0)
qRad3D = revolve2Dprofile(qRad2D, 1)
x,y,z = tools.cyl2xyz(qRad3D[:,0],qRad3D[:,2],qRad3D[:,1], inRadians=True)
centers = np.vstack([x,y,z]).T

#initialize mesh objects
#basic objects located at origin
#big = MeshObj(bigSphere)
#small = MeshObj(smallSphere)
#plane1 = MeshObj(planeFile1)
#initialize point source
#PS = photonSource(np.array([[0,0,0]]),[10])
#cube = MeshObj(cubeFile1) #cube at (0,0,0)

#located in divertor
SPsource = photonSource(centers,qRad3D[:,3])
cube = MeshObj(cubeFile2) #cube at (0,0,-1500)


#calculate power mappings
#print("\nPS to Small Sphere:")
#small = mapPhotonPower(PS, small, verbose=True)
print("\nStrike Point Source to Cube:")
cube = mapPhotonPower(SPsource, cube, verbose=True)
cube.writeHFpc(cube.centers,cube.qRad,tag='cube')
cube.writeHFpc(SPsource.centers,SPsource.P_rad,tag='photons')
