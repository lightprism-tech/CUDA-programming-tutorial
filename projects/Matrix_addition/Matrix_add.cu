#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

__global__ void matrixAdd(float *A, float *B, float *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
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
        cout << "Error: Cannot open file " << filename << endl;
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
            if (value.empty())
            {
                cout << "Empty value found at Row "
                     << row
                     << " Column "
                     << col
                     << endl;
                return false;
            }

            try
            {
                matrix[row * cols + col] = stof(value);
            }
            catch (const exception &e)
            {
                cout << "Invalid value: "
                     << value
                     << endl;

                cout << "Row = "
                     << row
                     << " Column = "
                     << col
                     << endl;

                cout << "File = "
                     << filename
                     << endl;

                return false;
            }

            col++;
        }

        if (col != cols)
        {
            cout << "Row "
                 << row
                 << " contains "
                 << col
                 << " columns instead of "
                 << cols
                 << endl;

            return false;
        }

        row++;
    }

    if (row != rows)
    {
        cout << "File contains "
             << row
             << " rows instead of "
             << rows
             << endl;

        return false;
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
        cout << "Cannot create file\n";
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

    // Matrix values
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

    int rows = 1000;
    int cols = 1000;

    vector<float> h_A(rows * cols);
    vector<float> h_B(rows * cols);
    vector<float> h_C(rows * cols);

    if (!readCSV("matrix_A.csv", h_A, rows, cols))
    {
        cout << "Failed to read matrix A" << endl;
        return 1;
    }

    if (!readCSV("matrix_B.csv", h_B, rows, cols))
    {
        cout << "Failed to read matrix B" << endl;
        return 1;
    }

    cout << "A[0] = " << h_A[0] << endl;
    cout << "B[0] = " << h_B[0] << endl;

    // matrixAdd<<<1, 1>>>(nullptr, nullptr, nullptr, 0, 0); // Dummy kernel launch to initialize CUDA runtime
    float *d_A;
    float *d_B;
    float *d_C;
    cudaError_t err;

    err = cudaMalloc(&d_A, rows * cols * sizeof(float));
    if (err != cudaSuccess)
    {
        cout << "Failed to allocate device memory for matrix A" << endl;
        return 1;
    }

    err = cudaMalloc(&d_B, rows * cols * sizeof(float));
    if (err != cudaSuccess)
    {
        cout << "Failed to allocate device memory for matrix B" << endl;
        return 1;
    }

    err = cudaMalloc(&d_C, rows * cols * sizeof(float));
    if (err != cudaSuccess)
    {
        cout << "Failed to allocate device memory for matrix C" << endl;
        return 1;
    }

    cudaMemcpy(d_A,
               h_A.data(),
               rows * cols * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,
               h_B.data(),
               rows * cols * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 block(16, 16);

    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixAdd<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Time taken for matrix addition: " << milliseconds << " ms" << endl;

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess)
    {
        cudaDeviceSynchronize();
    }

    cudaError_t memcpyErr = cudaMemcpy(h_C.data(),
                                       d_C,
                                       rows * cols * sizeof(float),
                                       cudaMemcpyDeviceToHost);
    if (memcpyErr != cudaSuccess)
    {
        cout << "Failed to copy result from device to host" << endl;
    }

    writeCSV("matrix_C.csv", h_C, rows, cols);

    cout << "C[0] = " << h_C[0] << endl;
    cout << "C[1] = " << h_C[1] << endl;
    cout << "C[2] = " << h_C[2] << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
