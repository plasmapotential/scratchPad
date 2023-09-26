import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import multiprocessing

# Define the CUDA kernel
kernel_code = """
__global__ void square(float *a)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  a[i] = a[i] * a[i];
}
"""

# Compile the CUDA kernel
mod = SourceModule(kernel_code)
square = mod.get_function("square")

def cpu_func(data):
    return data * data

if __name__ == '__main__':
    # Data preparation
    data_length = 1000000
    data = np.random.randn(data_length).astype(np.float32)

    # GPU Calculation
    start_gpu = time.time()

    # Allocate memory on the GPU and copy the data
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    # Run the kernel
    square(data_gpu, block=(512,1,1), grid=(data_length // 512 + 1,1))

    # Copy the result back from the GPU to the CPU
    result_gpu = np.empty_like(data)
    cuda.memcpy_dtoh(result_gpu, data_gpu)

    end_gpu = time.time()

    # CPU Calculation with multiprocessing
    start_cpu = time.time()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        result_cpu = pool.map(cpu_func, data)
    end_cpu = time.time()

    print(f"GPU Time: {end_gpu - start_gpu}")
    print(f"CPU Time with multiprocessing: {end_cpu - start_cpu}")
