#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int *A, int *B, int *C)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    C[id] = A[id] + B[id];
}

int main()s
{
    const int N = 5;

    int h_A[N] = {1,2,3,4,5};
    int h_B[N] = {5,4,3,2,1};
    int h_C[N];

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N*sizeof(int));
    cudaMalloc(&d_B, N*sizeof(int));
    cudaMalloc(&d_C, N*sizeof(int));

    cudaMemcpy(d_A, h_A, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*sizeof(int), cudaMemcpyHostToDevice);

    // Launch 8 threads
    add<<<2,4>>>(d_A,d_B,d_C);

    cudaMemcpy(h_C,d_C,N*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++)
        std::cout<<h_C[i]<<" ";

    std::cout<<std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}