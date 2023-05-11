# test ray-triangle intersections on GPU
#20220719
import numpy as np
import time
from numba import jit, prange, vectorize, cuda

#initialize triangles and rays
N=100
Nt = 1000
print("N = {:d}".format(N))
print("Nt = {:d}".format(Nt))

#All random
q1 = np.random.random(size=(N,3)).astype(np.float32)
q2 = np.random.random(size=(N,3)).astype(np.float32)
t1 = np.random.random(size=(Nt,3)).astype(np.float32)
t2 = np.random.random(size=(Nt,3)).astype(np.float32)
t3 = np.random.random(size=(Nt,3)).astype(np.float32)


@jit(nopython=True)
def signedVolume2Numba(a,b,c,d, ax=1):
    sums = []
    for i in prange(b.shape[0]):
        y = np.cross(b-a, c-a) * (d-a)
        sums.append(y.sum())
    return np.array([sums])

def signedVolume2(a,b,c,d, ax=1):
    return np.sum(np.cross(b-a, c-a) * (d-a), axis=ax)


#============================SV Hybrid
def SVhybrid():
    for i in range(N):
        intersect = np.zeros((N))
        #Perform Intersection Test
#        q13D = np.repeat(q1[i,np.newaxis], Nt, axis=0)
#        q23D = np.repeat(q2[i,np.newaxis], Nt, axis=0)
        q13D = np.repeat(np.expand_dims(q1[i], -1).transpose(), Nt, axis=0)
        q23D = np.repeat(np.expand_dims(q2[i], -1).transpose(), Nt, axis=0)
        sign1 = np.sign(signedVolume2(q13D,t1,t2,t3))
        sign2 = np.sign(signedVolume2(q23D,t1,t2,t3))
        sign3 = np.sign(signedVolume2(q13D,q23D,t1,t2))
        sign4 = np.sign(signedVolume2(q13D,q23D,t2,t3))
        sign5 = np.sign(signedVolume2(q13D,q23D,t3,t1))
        test1 = (sign1 != sign2)
        test2 = np.logical_and(sign3==sign4,sign3==sign5)
        intersect[i] = np.any(np.logical_and(test1,test2))
    return intersect

#============================SV Hybrid w/ Numba
@cuda.jit(nopython=True)
def SVhybridNumba():
    intersect = np.zeros((N))
    for i in range(N):
        #Perform Intersection Test
        q13D = q1[i].repeat(Nt).reshape(-1,Nt).transpose()
        q23D = q1[i].repeat(Nt).reshape(-1,Nt).transpose()
        sign1 = np.sign(signedVolume2Numba(q13D,t1,t2,t3))
        sign2 = np.sign(signedVolume2Numba(q23D,t1,t2,t3))
        sign3 = np.sign(signedVolume2Numba(q13D,q23D,t1,t2))
        sign4 = np.sign(signedVolume2Numba(q13D,q23D,t2,t3))
        sign5 = np.sign(signedVolume2Numba(q13D,q23D,t3,t1))
        test1 = (sign1 != sign2)
        test2 = np.logical_and(sign3==sign4,sign3==sign5)
        intersect[i] = np.any(np.logical_and(test1,test2))
    return intersect

#============================SV Hybrid w/ Numba
@vectorize(['float32(float32, float32, float32, float32, float32, float32, float32, float32)'], target='cuda')
def SVhybridNumbaGPU(q1,q2,t1,t2,t3,intersect,N,Nt):
    for i in range(N):
        #Perform Intersection Test
        q13D = q1[i].repeat(Nt).reshape(-1,Nt).transpose()
        q23D = q1[i].repeat(Nt).reshape(-1,Nt).transpose()
        sign1 = np.sign(signedVolume2Numba(q13D,t1,t2,t3))
        sign2 = np.sign(signedVolume2Numba(q23D,t1,t2,t3))
        sign3 = np.sign(signedVolume2Numba(q13D,q23D,t1,t2))
        sign4 = np.sign(signedVolume2Numba(q13D,q23D,t2,t3))
        sign5 = np.sign(signedVolume2Numba(q13D,q23D,t3,t1))
        test1 = (sign1 != sign2)
        test2 = np.logical_and(sign3==sign4,sign3==sign5)
        intersect[i] = np.any(np.logical_and(test1,test2))
    return intersect



t0 = time.time()
print("\nRegular run")
intersect = SVhybrid()
tNumba1 = time.time() - t0
print("t = {:f} s".format(tNumba1))
print(np.sum(intersect))

t0 = time.time()
print("\nNumba on GPU")
q1D = cuda.to_device(q1)
q2D = cuda.to_device(q2)
t1D = cuda.to_device(t1)
t2D = cuda.to_device(t2)
t3D = cuda.to_device(t3)
intersect = np.zeros((N)).astype(np.float32)
intersectD = cuda.to_device(intersect)
N=np.astype(np.float32)
Nt=np.astype(np.float32)
ND = cuda.to_device(N)
NtD = cuda.to_device(Nt)
intersect = SVhybridNumbaGPU(q1D,q2D,t1D,t2D,t3D,intersectD, ND,NtD)
tNumba3 = time.time() - t0
print("t = {:f} s".format(tNumba2))
print(np.sum(intersect))

#t0 = time.time()
#print("\nNumba compilation on CPU")
#intersect2 = SVhybridNumba()
#tNumba1 = time.time() - t0
#print("t = {:f} s".format(tNumba1))
#print(np.sum(intersect2))

t0 = time.time()
print("\nNumba already compiled on CPU")
intersect = SVhybridNumba()
tNumba2 = time.time() - t0
print("t = {:f} s".format(tNumba2))
print(np.sum(intersect))
