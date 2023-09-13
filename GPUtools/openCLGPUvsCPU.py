#runs a test comparing calculation time on GPU using OpenCL vs CPU using multiprocessing
import numpy as np
import pyopencl as cl
import time
from multiprocessing import Pool, cpu_count

# Create some test data
data_size = 2304  # number of stream processors
a = np.random.rand(data_size).astype(np.float32)
b = np.random.rand(data_size).astype(np.float32)
result_gpu = np.empty_like(a)
result_cpu = np.empty_like(a)

# ---- GPU Calculation ---- #
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

program_src = """
__kernel void sum_arrays(__global float *a, __global float *b, __global float *c) {
    int gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}
"""
program = cl.Program(context, program_src).build()
a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)

start_time_gpu = time.time()

program.sum_arrays(queue, a.shape, None, a_buffer, b_buffer, result_buffer)
cl.enqueue_copy(queue, result_gpu, result_buffer).wait()

end_time_gpu = time.time()

# ---- CPU Calculation using multiprocessing ---- #
def sum_arrays_cpu(index):
    return a[index] + b[index]

pool = Pool(cpu_count())

start_time_cpu = time.time()

result_cpu = pool.map(sum_arrays_cpu, range(data_size))

end_time_cpu = time.time()

pool.close()
pool.join()

# Print results
gpuTime = end_time_gpu - start_time_gpu
cpuTime = end_time_cpu - start_time_cpu
print("GPU Calculation Time:", gpuTime)
print("CPU Calculation Time:", cpuTime)
print("GPU was {:f}X faster than CPU".format(cpuTime/gpuTime))

# Optionally, validate results
if np.allclose(result_gpu, result_cpu):
    print("Results match!")
else:
    print("Results differ!")
