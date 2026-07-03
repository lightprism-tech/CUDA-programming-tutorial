# CUDA Matrix Multiplication (1000×1000) – Complete Explanation

## Results

Your measured results:

| Implementation | Time |
|---|---:|
| CUDA GPU | **6.408 ms** |
| CPU (-O2) | **881 ms** |

Estimated speedup:

```
881 / 6.408 ≈ 137.5×
```

## CUDA Code Explanation

### 1. Kernel

```cpp
__global__ void matrixMultiply(float *A, float *B, float *C,
                               int rowsA, int colsA, int colsB)
```

The kernel runs on the GPU. Each CUDA thread computes one element of the output matrix.

### 2. Thread Coordinates

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Each thread determines which output element C[row][col] it is responsible for.

### 3. Boundary Check

```cpp
if (row < rowsA && col < colsB)
```

Extra threads launched because of block rounding simply exit.

### 4. Dot Product

```cpp
float sum = 0.0f;

for(int k=0; k<colsA; k++)
{
    sum += A[row*colsA+k] * B[k*colsB+col];
}
```

The thread computes the dot product of one row of A and one column of B.

### 5. Store Result

```cpp
C[row * colsB + col] = sum;
```

The computed value is written into the output matrix.

## Host-side Workflow

1. Read CSV files into CPU vectors.
2. Allocate GPU memory with `cudaMalloc`.
3. Copy A and B to the GPU using `cudaMemcpy`.
4. Configure the execution grid (`16×16` threads per block).
5. Launch the kernel.
6. Synchronize the GPU.
7. Measure kernel time using CUDA Events.
8. Copy the result back to the CPU.
9. Write `matrix_C.csv`.
10. Free GPU memory.

## Grid Configuration

```cpp
dim3 threadsPerBlock(16,16);

dim3 numBlocks(
    (colsB + 15) / 16,
    (rowsA + 15) / 16
);
```

For a 1000×1000 matrix this launches a 63×63 grid.

## CPU Algorithm

The CPU implementation uses three nested loops:

```text
for each row
    for each column
        for each k
            sum += A[row][k] * B[k][col]
```

Only one CPU thread performs all computations.

## Comparison

| Feature | CPU | CUDA GPU |
|---|---|---|
| Parallelism | Single thread | Thousands of threads |
| Computation | Sequential | Parallel |
| Measured Time | 881 ms | 6.408 ms |
| Speedup | — | ~137.5× |

## Conclusion

Matrix multiplication is a compute-intensive algorithm. The GPU distributes the work across thousands of CUDA threads, making it substantially faster than the sequential CPU implementation in this benchmark.
