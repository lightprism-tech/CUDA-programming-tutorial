#include <iostream>
#include <cuda_runtime.h>

using namespace std;


__global__ void vectorAdd(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int N;

    cout << "Enter vector size: ";
    cin >> N;

   
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    cout << "\nEnter elements of Vector A:\n";
    for (int i = 0; i < N; i++)
        cin >> A[i];

    cout << "\nEnter elements of Vector B:\n";
    for (int i = 0; i < N; i++)
        cin >> B[i];

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

  
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

cudaError_t err = cudaGetLastError();
cout << "Launch Error: " << cudaGetErrorString(err) << endl;

err = cudaDeviceSynchronize();
cout << "Execution Error: " << cudaGetErrorString(err) << endl;

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

   
    cout << "\nResult Vector:\n";

    for (int i = 0; i < N; i++)
    {
        cout << C[i] << " ";
    }
    cout << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}