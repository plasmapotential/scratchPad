import numpy as np
import MDSplus

shot = 204118
time = 1004
server = 'skylark.pppl.gov'
tree='EFIT01'

print('Reading shot =', shot, 'and time =', time, 'from MDS+ tree:', tree)
MDS = MDSplus.Connection(server)
MDS.openTree(tree, shot)
#===
# GEQDSK
#===
baseG = 'RESULTS:GEQDSK:'
# get time slice
signal = 'GTIME'
k = np.argmin(np.abs(MDS.get(baseG + signal).data()*1000 - time))
time0 = int(MDS.get(baseG + signal).data()[k]*1000)
if (time != time0):
    if exact:
        raise RuntimeError(tree + ' does not exactly contain time ' + str(time) + '  ->  Abort')
    else:
        print('Warning: ' + tree + ' does not exactly contain time ' + str(time) + ' the closest time is ' + str(time0))
        print('Fetching time slice ' + str(time0))
        #time = time0
# store data in dictionary
g = {'shot':shot, 'time':time}

# get all signals, use same names as in read_g_file
translate = {'BCENTR': 'Bt0'}

for signal in translate:
    try:
        g[translate[signal]] = MDS.get(baseG + signal).data()[k]
    except:
        raise ValueError(signal +' retrieval failed.')

print(g)


#===
# DERIVED
#===
baseD = 'RESULTS:DERIVED:'
# store data in dictionary
d = {}

# get all signals, use same names as in read_g_file
translate = {'PTOT':'Ptot'}

for signal in translate:
    try:
        d[translate[signal]] = MDS.get(baseD + signal).data()[k]
    except:
        raise ValueError(signal +' retrieval failed.')
print(d)


#===
# AEQDSK
#===
baseA = 'RESULTS:AEQDSK:'
# get time slice
signal = 'ATIME'
k = np.argmin(np.abs(MDS.get(baseA + signal).data()*1000 - time))
time0 = int(MDS.get(baseA + signal).data()[k]*1000)
if (time != time0):
    if exact:
        raise RuntimeError(tree + ' does not exactly contain time ' + str(time) + '  ->  Abort')
    else:
        print('Warning: ' + tree + ' does not exactly contain time ' + str(time) + ' the closest time is ' + str(time0))
        print('Fetching time slice ' + str(time0))
        #time = time0
# store data in dictionary
a = {'shot':shot, 'time':time}

# get all signals, use same names as in read_g_file
translate = {'PBINJ':'Pbeam', 'POH':'Pohmic'}

for signal in translate:
    try:
        a[translate[signal]] = MDS.get(baseA + signal).data()[k]
    except:
        raise ValueError(signal +' retrieval failed.')

print(a)
