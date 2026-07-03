#include <iostream>
#include <fstream>
#include <sstream>  
#include <string>
#include <vector>
#include <chrono>

using namespace std;

void matrixMultiplyCPU(const vector<float> &A,
                       const vector<float> &B,
                       vector<float> &C,
                       int rowsA,
                       int colsA,
                       int colsB)
{
    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsB; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < colsA; k++)
            {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
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
    // Define matrix dimensions
    int rowsA = 1000, colsA = 1000;
    int rowsB = 1000, colsB = 1000;

    // Initialize matrices A and B
    vector<float> A(rowsA * colsA);
    vector<float> B(rowsB * colsB);
    vector<float> C(rowsA * colsB); // Result matrix

    // Read matrices from CSV files
    if (!readCSV("Matrix_A.csv", A, rowsA, colsA) || !readCSV("Matrix_B.csv", B, rowsB, colsB))
    {
        return -1;
    }

    auto start = chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C, rowsA, colsA, colsB);
    auto end = chrono::high_resolution_clock::now();

    // Calculate execution time
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "CPU Execution Time: " << duration.count() << " ms" << endl;

    // Write result to CSV file
    writeCSV("Matrix_C.csv", C, rowsA, colsB);

    return 0;
}