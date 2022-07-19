#comparing MT, MT Hybrid, SV, SV Hybrid
#plots that accompany this script are in speedPlots.py
#created during benchmark of MT vs SV on North Haven Island, ME

#import timeit
#cmd = "np.diagonal(np.matmul(np.cross(b-a,c-a),(d-a).T))"
#cmd1 = "np.sum(np.cross(b-a, c-a) * (d-a), axis=2)"
#cmd2 = "np.sum(np.cross(np.subtract(b,a), np.subtract(c,a)) * np.subtract(d,a), axis=2)"
#cmd3 = "np.sum(np.multiply(np.cross(np.subtract(b,a), np.subtract(c,a)), np.subtract(d,a)), axis=2)"
#stp = "import numpy as np; a = np.random.random(size=(30,1000,3)); b = np.random.random(size=(30,1000,3));  c = np.random.random(size=(30,1000,3));  d = np.random.random(size=(30,1000,3))"
#timeit.timeit(stmt=cmd,setup=stp, number=100)
#
#
#
#np.sum(np.cross(b-a, c-a) * (d-a), axis=2)
#np.sum(np.cross(np.subtract(b,a), np.subtract(c,a)) * np.subtract(d,a), axis=2)
#np.sum(np.multiply(np.cross(np.subtract(b,a), np.subtract(c,a)), np.subtract(d,a)), axis=2)

import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def signedVolume(a,b,c,d):
    return np.sum(np.cross(b-a, c-a) * (d-a))

def signedVolume2(a,b,c,d, ax=1):
    return np.sum(np.cross(b-a, c-a) * (d-a), axis=ax)



N=11000
Nt = 11000
print("N = {:d}".format(N))
print("Nt = {:d}".format(Nt))

#All random
q1 = np.random.random(size=(N,3))
q2 = np.random.random(size=(N,3))
t1 = np.random.random(size=(Nt,3))
t2 = np.random.random(size=(Nt,3))
t3 = np.random.random(size=(Nt,3))

#explicit intersection
#q1 = np.array(
#[[0.15575879, 0.59526916, 0.52951038]]
#)
#q2 = np.array(
#[[0.49243683, 0.21921991, 0.39319903]]
#)
#t1 = np.array(
#[[0.85127141, 0.79532697, 0.49488921]]
#)
#t2 = np.array(
#[[0.13896428, 0.69666591, 0.14403946]]
#)
#t3 = np.array(
#[[0.13855582, 0.95749092, 0.89430449]]
#)


#All inside triangle (should yield 1s)
#a = np.array([5,-5,1])
#q1 = np.repeat(a[:,np.newaxis], N, axis=1).T
#a = np.array([5,5,1])
#q2 = np.repeat(a[:,np.newaxis], N, axis=1).T
#a = np.array([0,0,0])
#t1 = np.repeat(a[:,np.newaxis], Nt, axis=1).T
#a = np.array([10,0,0])
#t2 = np.repeat(a[:,np.newaxis], Nt, axis=1).T
#a = np.array([0,0,10])
#t3 = np.repeat(a[:,np.newaxis], Nt, axis=1).T

#All Outside Triangle (should yield 0s)
#a = np.array([5,-5,-1])
#q1 = np.repeat(a[:,np.newaxis], N, axis=1).T
#a = np.array([5,5,-1])
#q2 = np.repeat(a[:,np.newaxis], N, axis=1).T
#a = np.array([0,0,0])
#t1 = np.repeat(a[:,np.newaxis], Nt, axis=1).T
#a = np.array([10,0,0])
#t2 = np.repeat(a[:,np.newaxis], Nt, axis=1).T
#a = np.array([0,0,10])
#t3 = np.repeat(a[:,np.newaxis], Nt, axis=1).T




#calculated beforehand
E1 = (t2-t1)
E2 = (t3-t1)
D = (q2-q1)
Dmag = np.linalg.norm(D, axis=1)
for i in range(len(D)):
    #E1[i] = E1[i] / np.linalg.norm(E1, axis=1)[i]
    #E2[i] = E2[i] / np.linalg.norm(E2, axis=1)[i]
    D[i] = D[i] / np.linalg.norm(D, axis=1)[i]

#============================Moller-Trumbore Loop
mask=False
if mask==True:
    t0 = time.time()
    intersect = np.zeros((N))
    for i in range(N):
        for j in range(Nt):
            h = np.cross(D[i], E2[j])
            a = np.dot(E1[j],h)
            if a < 0.0000001:
                continue
            f=1.0/a
            s = q1[i] - t1[j]
            u = f * np.dot(s,h)
            if (u<0.0) or (u>1.0):
                continue
            q = np.cross(s,E1[j])
            v = f*np.dot(D[i],q)
            if (v<0.0) or ((u+v)>1.0):
                continue
            t = f*np.dot(E2[j],q)
            if t>0.0000001:
                intersect[i] = 1
                break
    idx = np.where(intersect > 0)[0]
    tMT = time.time() - t0
    print("MT Loop Time = {:f} s".format(tMT))
    print(len(idx))

#============================Moller-Trumbore Hybrid === THIS IS FASTEST!
t0 = time.time()
intersect = np.zeros((N))
loc1 = []
eps = 0.0000001
for i in range(N):
    h = np.cross(D[i], E2)
    a = np.sum(E1*h, axis=1)
    test1 = np.logical_and( a>-eps, a<eps) #ray parallel to triangle
    #test1 = a<eps #ray parallel to triangle
    f=1.0/a
    s = q1[i] - t1
    u = f * np.sum(s*h, axis=1)
    test2 = np.logical_or(u<0.0, u>1.0) #ray inside triangle
    q = np.cross(s,E1)
    v = f*np.sum(D[i]*q, axis=1)
    test3 =  np.logical_or(v<0.0, (u+v)>1.0) #ray inside triangle
    t = f*np.sum(E2*q, axis=1)
#    test4 = np.logical_or(t<0.0, t>np.linalg.norm(D[i])) #ray long enough to intersect triangle
#    test4 = np.abs(t)>np.linalg.norm(D[i]) #ray long enough to intersect triangle
#    test4 = t<0 #intersection
#    test4 = np.logical_or(t<0.0, t>1.0) #ray long enough to intersect triangle
    test4 = np.logical_or(t<0.0, t>Dmag[i]) #ray long enough to intersect triangle
    if np.sum(~np.any([test1,test2,test3,test4], axis=0))>0:
        intersect[i] = 1
        loc1.append(i)
    #print("TESTS")
    #print(test1)
    #print(test2)
    #print(test3)
    #print(test4)
    #print(~np.any([test1,test2,test3,test4], axis=0))
    #print("DATA")
    #print(q1)
    #print(q2)
    #print(t1)
    #print(t2)
    #print(t3)
    #print("RESULTS")
    #print(a)
    #print(u)
    #print(v)
    #print(u+v)
    #print(t)
    #print(np.linalg.norm(D))
    #print("=============================================")

tMT_H = time.time() - t0
print("MT Hybrid Time = {:f} s".format(tMT_H))
print(sum(intersect))


#============================Moller-Trumbore Hybrid Sparse
t0 = time.time()
intersect = np.zeros((N))
eps = 0.0000001
test = np.zeros((len(t1)), dtype=bool)
s = np.zeros((len(t1),3))
u = np.zeros((len(t1)))
v = np.zeros((len(t1)))
q = np.zeros((len(t1),3))
t = np.zeros((len(t1)))
f = np.zeros((len(t1)))
for i in range(N):
    h = np.cross(D[i], E2)
    a = np.sum(E1*h, axis=1)
    test = np.logical_and( a>-eps, a<eps) #ray parallel to triangle
#    test1 = a<eps
    use = np.where(test==False)[0]
    f[use]=1.0/a[use]
    s[use] = q1[i] - t1[use]
    u[use] = f[use] * np.sum(s[use]*h[use], axis=1)
    test[use] = np.logical_or(u[use]<0.0, u[use]>1.0) #ray inside triangle
    use = np.where(test==False)[0]
    q[use] = np.cross(s[use],E1[use])
    v[use] = f[use]*np.sum(D[i]*q[use], axis=1)
    test[use] = np.logical_or(v[use]<0.0, (u[use]+v[use])>1.0) #ray inside triangle
    use = np.where(test==False)[0]
    t[use] = f[use]*np.sum(E2[use]*q[use], axis=1)
#    test4[use][use2][use3] =  np.logical_or(t<0.0, t>1.0) #ray long enough to intersect triangle
    test[use] = np.logical_or(t[use]<0.0, t[use]>Dmag[i]) #ray long enough to intersect triangle
    if False in test:
        intersect[i] = 1
    #print("TESTS")
    #print(np.logical_and( a>-eps, a<eps))
    #print(np.logical_or(u<0.0, u>1.0))
    #print(np.logical_or(v<0.0, (u+v)>1.0))
    #print(np.logical_or(t<0.0, t>Dmag))
    #print(np.sum(test))
    #print("DATA")
    #print(q1)
    #print(q2)
    #print(t1)
    #print(t2)
    #print(t3)
    #print("RESULTS")
    #print(a)
    #print(u)
    #print(v)
    #print(u+v)
    #print(t)
    #print(Dmag)




tMT_H_Acc = time.time() - t0
print("MT Hybrid Sparse Time = {:f} s".format(tMT_H_Acc))
print(sum(intersect))
#============================SV Hybrid
t0 = time.time()
intersect = np.zeros((N))
for i in range(N):
    #Perform Intersection Test
    q13D = np.repeat(q1[i,np.newaxis], Nt, axis=0)
    q23D = np.repeat(q2[i,np.newaxis], Nt, axis=0)
    sign1 = np.sign(signedVolume2(q13D,t1,t2,t3))
    sign2 = np.sign(signedVolume2(q23D,t1,t2,t3))
    sign3 = np.sign(signedVolume2(q13D,q23D,t1,t2))
    sign4 = np.sign(signedVolume2(q13D,q23D,t2,t3))
    sign5 = np.sign(signedVolume2(q13D,q23D,t3,t1))
    test1 = (sign1 != sign2)
    test2 = np.logical_and(sign3==sign4,sign3==sign5)
    intersect[i] = np.any(np.logical_and(test1,test2))

tSV_H = time.time() - t0
print("SV Hybrid Time = {:f} s".format(tSV_H))
print(np.sum(intersect))
#print(intersect)



#============================SV Matrix
t0 = time.time()
q13D = np.repeat(q1[:,np.newaxis], Nt, axis=1)
q23D = np.repeat(q2[:,np.newaxis], Nt, axis=1)
sign1 = np.sign(signedVolume2(q13D,t1,t2,t3,ax=2))
sign2 = np.sign(signedVolume2(q23D,t1,t2,t3,ax=2))
sign3 = np.sign(signedVolume2(q13D,q23D,t1,t2,ax=2))
sign4 = np.sign(signedVolume2(q13D,q23D,t2,t3,ax=2))
sign5 = np.sign(signedVolume2(q13D,q23D,t3,t1,ax=2))
test1 = (sign1 != sign2)
test2 = np.logical_and(sign3==sign4,sign3==sign5)
idx = np.any(np.logical_and(test1,test2)==True, axis=1)
try:
    use = np.where(np.logical_and(test1,test2))
    use = use[0][0]
except:
     use = np.array([])
tSV_M = time.time() - t0
print("SV Matrix Time = {:f} s".format(tSV_M))
print(np.sum(idx))





#=====================PRINTS
#print("MT H loc")
#print(loc1)
#print("SV list")
#loc2 = np.where(idx==True)[0]
#print(loc2)
#if len(loc1) > len(loc2):
#    diff = np.setdiff1d(loc1,loc2)
#else:
#    diff = np.setdiff1d(loc2,loc1)
#print(diff)
#i = diff[0]
#print("i = {:d}".format(i))
#print("DATA")
#print(q1)
#print(q2)
#print(t1)
#print(t2)
#print(t3)
#h = np.cross(D[i], E2)
#a = np.sum(E1*h, axis=1)
#test1 = np.logical_and( a>-eps, a<eps) #ray parallel to triangle
#f=1.0/a
#s = q1[i] - t1
#u = f * np.sum(s*h, axis=1)
#test2 = np.logical_or(u<0.0, u>1.0) #ray inside triangle
#q = np.cross(s,E1)
#v = f*np.sum(D[i]*q, axis=1)
#test3 =  np.logical_or(v<0.0, (u+v)>1.0) #ray inside triangle
#t = f*np.sum(E2*q, axis=1)
#test4 = np.logical_or(t<0.0, t>np.linalg.norm(D[i])) #ray long enough to intersect triangle
#test4 = np.abs(t)>np.linalg.norm(D[i])
#print("TEST")
#print(test1)
#print(test2)
#print(test3)
#print(test4)
#print("RESULTS")
#print(a)
#print(u)
#print(v)
#print(u+v)
#print(t)
#print("Ds")
#print(np.linalg.norm(D[i]))
#print(D[i])
#print(Dmag[i])
#print("FACE")
#TEST = ~np.any([test1,test2,test3,test4], axis=0)
#try:
#    use = np.where(TEST == True)[0][0]
#except:
#    print(use)
#
#print(t1[use])
#print(t2[use])
#print(t3[use])
#
#q1T = q1[i]
#q2T = q2[i]
#t1T = t1[use]
#t2T = t2[use]
#t3T = t3[use]


print("===============RESULTS=============")
print("{:f}\t{:f}\t{:f}\t{:f}".format(tMT_H, tMT_H_Acc, tSV_H, tSV_M))

plotMask=False
if plotMask == True:
    x = np.hstack([t1[:,0],t2[:,0],t3[:,0]])
    y = np.hstack([t1[:,1],t2[:,1],t3[:,1]])
    z = np.hstack([t1[:,2],t2[:,2],t3[:,2]])
    arrX = [None]*5*len(t1)
    arrX[::5] = t1[:,0]
    arrX[1::5] = t2[:,0]
    arrX[2::5] = t3[:,0]
    arrX[3::5] = t1[:,0]
    arrY = [None]*5*len(t1)
    arrY[::5] = t1[:,1]
    arrY[1::5] = t2[:,1]
    arrY[2::5] = t3[:,1]
    arrY[3::5] = t1[:,1]
    arrZ = [None]*5*len(t1)
    arrZ[::5] = t1[:,2]
    arrZ[1::5] = t2[:,2]
    arrZ[2::5] = t3[:,2]
    arrZ[3::5] = t1[:,2]
    D=np.vstack([q1,q2])
    arrDx = [None]*(3*len(q1))
    arrDy = [None]*(3*len(q1))
    arrDz = [None]*(3*len(q1))

    arrDx[::3] = q1[:,0]
    arrDy[::3] = q1[:,1]
    arrDz[::3] = q1[:,2]

    arrDx[1::3] = q2[:,0]
    arrDy[1::3] = q2[:,1]
    arrDz[1::3] = q2[:,2]

    t = np.vstack([t1,t2,t3])
    try:
        x = np.array([t1T[0],t2T[0],t3T[0],t1T[0]])
        y = np.array([t1T[1],t2T[1],t3T[1],t1T[1]])
        z = np.array([t1T[2],t2T[2],t3T[2],t1T[2]])
    except:
        x = np.array([t1T,t2T,t3T,t1T])
        y = np.array([t1T,t2T,t3T,t1T])
        z = np.array([t1T,t2T,t3T,t1T])

#    fig = go.Figure(data=[go.Scatter3d(x=arrDx,
#                                   y=arrDy,
#                                   z=arrDz,
#                                   mode='lines+markers',
#                                   name="D"),
#                                   ],
#                                   )
#    fig.add_trace(go.Scatter3d(x=arrX, y=arrY, z=arrZ, mode='lines+markers'))
    DT=np.vstack([q1T,q2T])
    tT = np.vstack([t1T,t2T,t3T])
    fig = go.Figure(data=[go.Scatter3d(x=DT[:,0],
                                   y=DT[:,1],
                                   z=DT[:,2],
                                   mode='lines+markers',
                                   name="D"),
                                   ],
                                   )

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers'))


    fig.show()


#for creating a plot of the data from multiple of the above runs
N = np.array([1,10,100,1000,10000])
Nt = np.array([1,10,100,1000,10000])
