# CUDA Matrix Addition Project -- Step-by-Step Explanation

## Overview

This program performs matrix addition on the GPU.

Pipeline:

    matrix_A.csv
            \
             \
              ---> CPU Memory ---> GPU Memory ---> CUDA Kernel ---> GPU Memory ---> CPU Memory ---> matrix_C.csv
             /
    matrix_B.csv

------------------------------------------------------------------------

# 1. Header Files

``` cpp
#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
```

-   `iostream` -- Console input/output.
-   `cuda_runtime.h` -- CUDA APIs (`cudaMalloc`, `cudaMemcpy`, kernels).
-   `fstream` -- Read/write CSV files.
-   `sstream` -- Split CSV lines using commas.
-   `string` -- String handling.
-   `vector` -- Dynamic arrays used for host memory.

------------------------------------------------------------------------

# 2. CUDA Kernel

``` cpp
__global__ void matrixAdd(...)
```

`__global__` means the function runs on the GPU.

Each thread computes one matrix element.

## Compute the row

``` cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Example:

-   Block Y = 2
-   Block height = 16
-   Thread Y = 5

```{=html}
<!-- -->
```
    row = 2 × 16 + 5 = 37

## Compute the column

``` cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Example:

    col = 1 × 16 + 8 = 24

The thread is responsible for element:

    (37,24)

## Bounds check

``` cpp
if(row < rows && col < cols)
```

Prevents threads outside the matrix from accessing invalid memory.

## Convert 2D index to 1D

``` cpp
index = row * cols + col;
```

Example:

    row = 2
    col = 5
    cols = 1000

    index = 2*1000+5 = 2005

## Add the matrices

``` cpp
C[index] = A[index] + B[index];
```

------------------------------------------------------------------------

# 3. readCSV()

Reads the CSV file into a host vector.

Steps:

1.  Open file.
2.  Skip header.
3.  Read one row.
4.  Split by comma.
5.  Convert text to float using `stof`.
6.  Store into the vector.

The vector stores the matrix in **row-major order**.

------------------------------------------------------------------------

# 4. writeCSV()

Writes the result vector back to disk.

Steps:

1.  Create file.
2.  Write column headers.
3.  Write every matrix value.
4.  Close the file.

------------------------------------------------------------------------

# 5. main()

## Define matrix size

``` cpp
rows = 1000;
cols = 1000;
```

Total elements:

    1000 × 1000 = 1,000,000

## Host memory

``` cpp
vector<float> h_A;
vector<float> h_B;
vector<float> h_C;
```

These live in CPU memory.

## Read CSV files

``` cpp
readCSV(...)
```

Copies file contents into `h_A` and `h_B`.

## Device pointers

``` cpp
float *d_A;
float *d_B;
float *d_C;
```

These point to GPU memory.

## Allocate GPU memory

``` cpp
cudaMalloc(...)
```

Reserves space on the GPU.

Approximate size per matrix:

    1,000,000 floats × 4 bytes ≈ 4 MB

Three matrices require about **12 MB**.

## Copy CPU → GPU

``` cpp
cudaMemcpy(..., cudaMemcpyHostToDevice);
```

Copies `h_A` → `d_A` and `h_B` → `d_B`.

## Configure execution

``` cpp
dim3 block(16,16);
```

Each block has:

    16 × 16 = 256 threads

Grid:

``` cpp
dim3 grid(
    (cols + block.x - 1) / block.x,
    (rows + block.y - 1) / block.y
);
```

For a 1000×1000 matrix:

    Grid ≈ 63 × 63 blocks

Total launched threads:

    63 × 63 × 256 = 1,016,064

Extra threads safely exit because of the bounds check.

## Launch kernel

``` cpp
matrixAdd<<<grid, block>>>(...);
```

CUDA launches over one million threads.

Each thread computes exactly one element.

## Error checking

``` cpp
cudaGetLastError();
```

Checks launch errors.

## Copy GPU → CPU

``` cpp
cudaMemcpy(..., cudaMemcpyDeviceToHost);
```

Copies the result into `h_C`.

## Save result

``` cpp
writeCSV("matrix_C.csv", h_C, rows, cols);
```

Writes the GPU result to disk.

## Free GPU memory

``` cpp
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```

Releases GPU resources.

------------------------------------------------------------------------

# Complete Execution Flow

    matrix_A.csv
            │
            ▼
    readCSV()

    matrix_B.csv
            │
            ▼
    readCSV()

            │
            ▼
    Host Vectors (CPU)

            │
            ▼
    cudaMalloc()

            │
            ▼
    cudaMemcpy (Host → Device)

            │
            ▼
    CUDA Kernel

            │
            ▼
    cudaMemcpy (Device → Host)

            │
            ▼
    writeCSV()

            │
            ▼
    matrix_C.csv

------------------------------------------------------------------------

# CUDA Concepts Used

-   CUDA kernel (`__global__`)
-   Thread indexing
-   Blocks and Grids
-   Bounds checking
-   Row-major indexing
-   GPU memory allocation (`cudaMalloc`)
-   Host/device memory copy (`cudaMemcpy`)
-   Kernel launch
-   CUDA error checking
-   GPU memory cleanup (`cudaFree`)
...........