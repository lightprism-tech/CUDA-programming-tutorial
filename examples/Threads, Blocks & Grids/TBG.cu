#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadInfo()
{
    printf("Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    // Launch:
    // 2 Blocks
    // 4 Threads per Block
    printThreadInfo<<<2, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}