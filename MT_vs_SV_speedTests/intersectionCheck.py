import numpy as np


q1 = np.array([[1,0,0],[0,1,0]])
q2 = np.array([[-1,0,0],[0,-1,0]])

p1 = np.array([[0,0,1],[0,0,1],[0,0,1],[1,1,1]])
p2 = np.array([[0,1,-1],[1,0,-1],[0,1,0],[1,1,0]])
p3 = np.array([[0,-1,-1],[-1,0,-1],[0,0,1],[0,1,0]])
Nt = len(p1)

q13D = np.repeat(q1[:,np.newaxis], Nt, axis=1)
q23D = np.repeat(q2[:,np.newaxis], Nt, axis=1)

psiMask = np.array(
    [[1,0],
     [1,1],
     [1,1],
     [0,0]]
)

p13D = []
p23D = []
p33D = []
for i in range(len(q1)):
    useList = np.where( psiMask[:,i]==1 )[0]
    p13D.append(p1[useList])
    p23D.append(p2[useList])
    p33D.append(p3[useList])

intersectionCheck(q13D,q23D,p1,p2,p3)


def signedVolume2(a,b,c,d):
    return np.sum(np.cross(b-a, c-a) * (d-a), axis=2)

def intersectionCheck(q13D,q23D,p1,p2,p3):
    sign1 = np.sign(signedVolume2(q13D,p1,p2,p3))
    sign2 = np.sign(signedVolume2(q23D,p1,p2,p3))
    sign3 = np.sign(signedVolume2(q13D,q23D,p1,p2))
    sign4 = np.sign(signedVolume2(q13D,q23D,p2,p3))
    sign5 = np.sign(signedVolume2(q13D,q23D,p3,p1))
    test1 = (sign1 != sign2)
    test2 = np.logical_and(sign3==sign4,sign3==sign5)
    print(np.logical_and(test1,test2))
    return np.logical_and(test1,test2)


>>> intersectionCheck(q13D,q23D,p1,p2,p3)
[[ True False False False]
 [False  True False False]]
1
