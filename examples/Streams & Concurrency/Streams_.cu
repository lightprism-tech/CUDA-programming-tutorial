#include <stdio.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void HelloWorldKernel(int id)
{
    printf("Hello, for stream!%d\n", id);
}

int main()
{

    cudaStream_t stream[4];

    for (int i = 0; i < 4; i++)
    {
        
        cudaStreamCreate(&stream[i]);
       
    }
    
    for (int i = 0; i < 4; i++)
    {
        HelloWorldKernel<<<1, 1, 0, stream[i]>>>(i);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    
    for (int i = 0; i < 4; i++)
    {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 4; i++)
    {
        cudaStreamDestroy(stream[i]);
    }

    return 0;
}

// output:
// Hello, for stream!0
// Hello, for stream!1
// Hello, for stream!2
// Hello, for stream!3