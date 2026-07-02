# CUDA Kernel Optimization Example: Bounds Checking

## Objective

Learn the first and most important CUDA kernel optimization: **Bounds
Checking**.

------------------------------------------------------------------------

## Problem

We want to add two vectors.

    A = [1,2,3,4,5]
    B = [5,4,3,2,1]
    C = [6,6,6,6,6]

Each GPU thread computes one element.

------------------------------------------------------------------------

## Version 1: Naive Kernel (Unsafe)

``` cpp
__global__ void add(int *A, int *B, int *C)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    C[id] = A[id] + B[id];
}
```

Kernel launch:

``` cpp
add<<<2,4>>>(d_A, d_B, d_C);
```

### What does `<<<2,4>>>` mean?

-   2 Blocks
-   4 Threads per Block

Total threads:

    2 × 4 = 8 Threads

Global thread IDs:

    Block   Thread   Global ID
  ------- -------- -----------
        0        0           0
        0        1           1
        0        2           2
        0        3           3
        1        0           4
        1        1           5
        1        2           6
        1        3           7

The vector has only **5 elements** (`N = 5`).

Valid IDs:

    0 1 2 3 4

Invalid IDs:

    5 6 7

These threads try to access:

``` cpp
A[5]
A[6]
A[7]
```

This is called **Out-of-Bounds Memory Access**.

Possible results:

-   Program crash
-   Garbage output
-   Undefined behavior

------------------------------------------------------------------------

## Version 2: Optimized Kernel (Safe)

``` cpp
__global__ void add(int *A, int *B, int *C, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N)
    {
        C[id] = A[id] + B[id];
    }
}
```

Kernel launch:

``` cpp
add<<<2,4>>>(d_A, d_B, d_C, N);
```

### What happens?

-   Threads 0--4 satisfy `id < N` and perform the addition.
-   Threads 5--7 fail the condition and immediately exit.

No invalid memory access occurs.

------------------------------------------------------------------------

## Why Launch Extra Threads?

Suppose:

-   `N = 1000`
-   `256 threads per block`

Blocks required:

``` text
blocks = (1000 + 255) / 256 = 4
```

Threads launched:

``` text
4 × 256 = 1024
```

Extra threads:

``` text
1024 - 1000 = 24
```

Those extra threads simply skip the computation because of:

``` cpp
if (id < N)
```

This is the standard CUDA programming pattern.

------------------------------------------------------------------------

## Why Is This an Optimization?

Without bounds checking:

    Extra Thread
          ↓
    Reads invalid memory
          ↓
    Crash / Undefined Behaviour

With bounds checking:

    Extra Thread
          ↓
    Condition fails
          ↓
    Thread exits safely

------------------------------------------------------------------------

# CUDA Optimization Roadmap

1.  Bounds Checking
2.  Coalesced Memory Access
3.  Shared Memory
4.  Reduce Branch Divergence
5.  Loop Unrolling
6.  Occupancy Tuning
7.  Asynchronous Memory Copies
8.  Streams and Concurrency

Bounds checking is the first optimization every CUDA programmer should
implement.
