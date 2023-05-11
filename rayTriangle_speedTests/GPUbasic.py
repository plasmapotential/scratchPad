#GPUbasic.py
#basic GPU program for testing CUDA functions
import time
from numba import vectorize
import numpy as np

@vectorize(['float32(float32, float32)'], target='cuda')
def add_ufunc(x, y):
    return x + y

n = 100000
x = np.arange(n).astype(np.float32)
y = 2 * x

t0 = time.time()
add_ufunc(x, y)  # Baseline performance with host arrays
t1 = time.time()-t0

from numba import cuda

x_device = cuda.to_device(x)
y_device = cuda.to_device(y)

print(x_device)
print(x_device.shape)
print(x_device.dtype)

t0 = time.time()
add_ufunc(x_device, y_device)
t2 = time.time()-t0

print("t1: {:f}".format(t1))
print("t2: {:f}".format(t2))
print("Gain: {:f}".format(t1/t2))
