// indexing.cu — 1D, 2D, and 3D thread indexing examples
// Compile: nvcc indexing.cu -o indexing
// Run:     ./indexing

#include <stdio.h>

// ─── 1D Indexing ──────────────────────────────────────────────────────────────
__global__ void index_1d(int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        printf("[1D] global_id=%d\n", idx);
}

// ─── 2D Indexing ──────────────────────────────────────────────────────────────
__global__ void index_2d(int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        int flat_idx = row * cols + col;
        printf("[2D] row=%d col=%d flat=%d\n", row, col, flat_idx);
    }
}

// ─── 3D Indexing ──────────────────────────────────────────────────────────────
__global__ void index_3d(int X, int Y, int Z) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < X && y < Y && z < Z)
        printf("[3D] x=%d y=%d z=%d\n", x, y, z);
}

int main() {
    printf("=== 1D: 16 threads ===\n");
    index_1d<<<2, 8>>>(16);
    cudaDeviceSynchronize();

    printf("\n=== 2D: 4x4 grid ===\n");
    dim3 block2(2, 2), grid2(2, 2);
    index_2d<<<grid2, block2>>>(4, 4);
    cudaDeviceSynchronize();

    printf("\n=== 3D: 2x2x2 grid ===\n");
    dim3 block3(1, 1, 1), grid3(2, 2, 2);
    index_3d<<<grid3, block3>>>(2, 2, 2);
    cudaDeviceSynchronize();

    return 0;
}
