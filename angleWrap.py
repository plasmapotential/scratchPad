#script for testing angle wrap cases in HEAT,
#specifically when filtering by phi
import numpy as np


#case1: at angle wrap location
#indices 0,1 should fail, 2,3 should pass
phiP1 = np.radians([177,-177,177,179.5])
phiP2 = np.radians([178, -178, -177, -179.5])
phiMin = np.radians(-179)
phiMax = np.radians(179)
test(phiP1,phiP2,phiMin,phiMax)

#case2: at angle wrap location, opposite dir
phiP1 = np.radians([177,-177,177,179.5])
phiP2 = np.radians([178, -178, -177, -179.5])
phiMin = np.radians(179)
phiMax = np.radians(-179)
test(phiP1,phiP2,phiMin,phiMax)

#case3: random location
phiP1 = np.radians([147,157,147,148.5])
phiP2 = np.radians([146,158,157,149.5])
phiMin = np.radians(148)
phiMax = np.radians(150)
test(phiP1,phiP2,phiMin,phiMax)

#case4: random location reverse
phiP1 = np.radians([147,157,147,148.5])
phiP2 = np.radians([146,158,157,149.5])
phiMin = np.radians(150)
phiMax = np.radians(148)
test(phiP1,phiP2,phiMin,phiMax)

#case5: another random location in lower hemishpere
phiP1 = np.radians([-147,-157,-147,-148.5])
phiP2 = np.radians([-146,-158,-157,-149.5])
phiMin = np.radians(-150)
phiMax = np.radians(-148)
test(phiP1,phiP2,phiMin,phiMax)

#case6: another random location in lower hemishpere reverse
phiP1 = np.radians([-147,-157,-147,-148.5])
phiP2 = np.radians([-146,-158,-157,-149.5])
phiMin = np.radians(-150)
phiMax = np.radians(-148)
test(phiP1,phiP2,phiMin,phiMax)

#case7: at 0 location
phiP1 = np.radians([2,-2,2,0.5])
phiP2 = np.radians([3,-3,-2,-0.5])
phiMin = np.radians(-1)
phiMax = np.radians(1)
test(phiP1,phiP2,phiMin,phiMax)

#case7: at 0 location reverse
phiP1 = np.radians([2,-2,2,0.5])
phiP2 = np.radians([3,-3,-2,-0.5])
phiMin = np.radians(1)
phiMax = np.radians(-1)
test(phiP1,phiP2,phiMin,phiMax)



def test(phiP1,phiP2,phiMin,phiMax):
    if np.abs(phiMin-phiMax) > np.radians(10):
        phiP1[phiP1<0] += 2*np.pi
        phiP2[phiP2<0] += 2*np.pi
        if phiMin < 0: phiMin+=2*np.pi
        if phiMax < 0: phiMax+=2*np.pi
    if phiMin > phiMax:
        min = phiMax
        max = phiMin
    else:
        min = phiMin
        max = phiMax
    test0 = np.logical_and(phiP1 < min, phiP2 < min)
    test1 = np.logical_and(phiP1 > max, phiP2 > max)
    return np.logical_or(test0,test1)
