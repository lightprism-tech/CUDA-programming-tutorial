#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

//------------------------------------------------------------
// CPU Matrix Addition
//------------------------------------------------------------
void matrixAddCPU(const vector<float> &A,
                  const vector<float> &B,
                  vector<float> &C,
                  int rows,
                  int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int index = i * cols + j;
            C[index] = A[index] + B[index];
        }
    }
}

//------------------------------------------------------------
// Read CSV
//------------------------------------------------------------
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

//------------------------------------------------------------
// Write CSV
//------------------------------------------------------------
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

//------------------------------------------------------------
// Main
//------------------------------------------------------------
int main()
{
    int rows = 1000;
    int cols = 1000;

    vector<float> h_A(rows * cols);
    vector<float> h_B(rows * cols);
    vector<float> h_C(rows * cols);

    //--------------------------------------------------------
    // Read matrices
    //--------------------------------------------------------

    if (!readCSV("matrix_A.csv", h_A, rows, cols))
        return 1;

    if (!readCSV("matrix_B.csv", h_B, rows, cols))
        return 1;

    cout << "A[0] = " << h_A[0] << endl;
    cout << "B[0] = " << h_B[0] << endl;

    //--------------------------------------------------------
    // Measure CPU time
    //--------------------------------------------------------

    auto start = chrono::high_resolution_clock::now();

    matrixAddCPU(h_A, h_B, h_C, rows, cols);

    auto stop = chrono::high_resolution_clock::now();

    auto duration =
        chrono::duration_cast<chrono::milliseconds>(stop - start);

    //--------------------------------------------------------
    // Save result
    //--------------------------------------------------------

    writeCSV("matrix_C.csv", h_C, rows, cols);

    cout << "C[0] = " << h_C[0] << endl;
    cout << "C[1] = " << h_C[1] << endl;
    cout << "C[2] = " << h_C[2] << endl;

    cout << "\nCPU Matrix Addition Time = "
         << duration.count()
         << " ms" << endl;

    return 0;
}