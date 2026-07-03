#include <iostream>
#include <cuda_runtime.h>
#include <fstream>  
#include <sstream>
#include <string>   
#include <vector>
     
using namespace std;

__global__ void matrixMultiply(float *A, float *B, float *C, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;              

    if (row < rowsA && col < colsB)
    {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k)
        {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}


bool readCSV(const string &filename,
             vector<float> &matrix,
             int rows,
             int cols)
{
    ifstream file(filename);

    if (!file.is_open())
    {
        cout << "Cannot open " << filename << endl;
        return false;
    }

    string line;

    // Skip header
    getline(file, line);

    int row = 0;

    while (getline(file, line) && row < rows)
    {
        stringstream ss(line);

        string value;

        int col = 0;

        while (getline(ss, value, ',') && col < cols)
        {
            matrix[row * cols + col] = stof(value);
            col++;
        }

        row++;
    }

    file.close();

    return true;
} 

void writeCSV(const string &filename,
              const vector<float> &matrix,
              int rows,
              int cols)
{
    ofstream file(filename);

    if (!file.is_open())
    {
        cout << "Cannot create file." << endl;
        return;
    }

    // Header
    for (int j = 0; j < cols; j++)
    {
        file << "C" << j;

        if (j != cols - 1)
            file << ",";
    }

    file << "\n";

    // Data
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            file << matrix[i * cols + j];

            if (j != cols - 1)
                file << ",";
        }

        file << "\n";
    }

    file.close();
}



int main()
{
    const int rowsA = 1000;
    const int colsA = 1000;
    const int rowsB = 1000;
    const int colsB = 1000;

    vector<float> A(rowsA * colsA);
    vector<float> B(rowsB * colsB);
    vector<float> C(rowsA * colsB);

    if (!readCSV("matrix_A.csv", A, rowsA, colsA) || !readCSV("matrix_B.csv", B, rowsB, colsB))
    {
        return -1;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, A.size() * sizeof(float));
    cudaMalloc((void **)&d_B, B.size() * sizeof(float));
    cudaMalloc((void **)&d_C, C.size() * sizeof(float));

    cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
  
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time taken for matrix multiplication: " << milliseconds << " ms" << endl;


    cudaMemcpy(C.data(), d_C, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    writeCSV("matrix_C.csv", C, rowsA, colsB);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}