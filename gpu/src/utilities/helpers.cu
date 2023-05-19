#include "helpers.h"

#include <iostream>

__device__ unsigned getThIndex() { return blockIdx.x * blockDim.x + threadIdx.x; }

void checkForErrors()
{
    cudaDeviceSynchronize();
    printf("Ran into: %s\n",
           cudaGetErrorString(cudaGetLastError()));
    printf("Sync: %s\n\n", cudaGetErrorString(cudaDeviceSynchronize()));
}
