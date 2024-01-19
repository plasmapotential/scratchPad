#prints R,Z,S
import pandas as pd
import numpy as np

# Calculate distance along curve/wall (also called S):
def distance(rawdata):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(rawdata,axis=0)**2,axis=1)))
    distance = np.insert(distance, 0, 0)
    return distance


wallFile = '/home/tlooby/SPARC/RZcontours/v3c.csv'
outFile = '/home/tlooby/SPARC/RZcontours/RZS_v3c.csv'
df = pd.read_csv(wallFile, names=['R','Z'])


dist = distance(df.values)

arr = np.vstack([df.values[:,0], df.values[:,1], dist]).T
head = 'R,Z,S'
np.savetxt(outFile, arr,  delimiter=',',fmt='%.10f', header=head)