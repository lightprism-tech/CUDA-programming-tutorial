# Module 05 — Threads, Blocks & Grids

> **Estimated time:** ~3 hours  
> **Prerequisites:** [Module 04 — CUDA Setup & Toolchain](../module-04-cuda-setup/README.md)  
> **Next module:** [Module 06 — Memory Model](../module-06-memory-model/README.md)

---

## What You Will Learn

- What a CUDA thread is and how it differs from a CPU thread
- Built-in thread variables and how to calculate a global thread ID
- What a warp is and why it matters for performance
- Warp divergence — the #1 performance killer and how to avoid it
- Thread synchronisation with `__syncthreads()`
- Block dimensions (1D, 2D, 3D) and how to choose block size
- Shared memory — the fastest memory on the GPU
- GPU Occupancy — keeping the SM busy
- Grid dimensions and the three essential data-mapping patterns
- How the GPU schedules blocks across Streaming Multiprocessors

---

## Part 1 — Threads

### 1.1 What Is a Thread?

A **thread** is the smallest unit of execution in CUDA. It is a single worker that runs your GPU function (kernel). When you launch a kernel, the GPU creates thousands of these threads simultaneously — each one runs the same code, but works on a different piece of data using its unique ID.

Compare this to CPU programming, where code runs as a single thread line by line:

```cpp
// CPU — one thread, processes ONE element at a time
void add_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];   // sequential: one addition per iteration
    }
}

// CUDA — thousands of threads, each processes ONE element simultaneously
__global__ void add_gpu(float* A, float* B, float* C) {
    int i = threadIdx.x;      // each thread has its own unique i
    C[i] = A[i] + B[i];       // all threads execute this at the same time
}

// Launch 1000 threads → 1000 additions happen simultaneously
add_gpu<<<1, 1000>>>(A, B, C);
```

---

### 1.2 Built-in Thread Variables

Every CUDA thread automatically receives these read-only variables. You do not create them — the CUDA runtime fills them in:

| Variable | Type | Meaning | Example Value |
| :--- | :--- | :--- | :--- |
| `threadIdx.x` | `uint3` | Thread position within its block (x-axis) | `0` to `blockDim.x - 1` |
| `threadIdx.y` | `uint3` | Thread position within its block (y-axis) | `0` to `blockDim.y - 1` |
| `threadIdx.z` | `uint3` | Thread position within its block (z-axis) | `0` to `blockDim.z - 1` |
| `blockIdx.x` | `uint3` | Which block this thread belongs to (x-axis) | `0` to `gridDim.x - 1` |
| `blockIdx.y` | `uint3` | Which block this thread belongs to (y-axis) | `0` to `gridDim.y - 1` |
| `blockDim.x` | `dim3` | Total threads per block in x-direction | `256` (you set this) |
| `blockDim.y` | `dim3` | Total threads per block in y-direction | `1` (default) |
| `gridDim.x` | `dim3` | Total blocks in the grid (x-direction) | `N/256` (you set this) |
| `warpSize` | `int` | Threads per warp — always 32 | `32` |

---

### 1.3 Calculating a Global Thread ID

`threadIdx.x` only identifies a thread *within its block* (0 to blockDim-1). To get a globally unique ID across the entire GPU, you combine block position and thread position — like apartment numbering: **(floor number × apartments per floor) + door number**.

```cpp
// ── 1D ID (most common — arrays, vectors) ──────────────────────────────
int global_id = blockIdx.x * blockDim.x + threadIdx.x;

// Visualisation with blockDim.x = 4:
//  Block 0:  thread 0 → id=0  | thread 1 → id=1  | thread 2 → id=2  | thread 3 → id=3
//  Block 1:  thread 0 → id=4  | thread 1 → id=5  | thread 2 → id=6  | thread 3 → id=7
//  Block 2:  thread 0 → id=8  | thread 1 → id=9  | thread 2 → id=10 | thread 3 → id=11

// ── 2D ID (images, matrices) ───────────────────────────────────────────
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int pixel_id = row * image_width + col;   // convert 2D → 1D memory index

// ── 3D ID (volumes, medical imaging, CFD simulations) ─────────────────
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

---

### 1.4 Warps — The Real Hardware Execution Unit

You write code in threads, but the GPU hardware executes in groups of **32 threads** called **warps**. All 32 threads in a warp execute the exact same instruction at the exact same clock cycle. This model is called **SIMT — Single Instruction, Multiple Threads**.

```
You launch 256 threads per block
GPU divides them into 256 / 32 = 8 warps

Warp 0 → threads   0–31  → all execute: C[i] = A[i] + B[i]  (simultaneously)
Warp 1 → threads  32–63  → all execute: C[i] = A[i] + B[i]  (simultaneously)
Warp 2 → threads  64–95  → all execute: C[i] = A[i] + B[i]  (simultaneously)
...
Warp 7 → threads 224–255 → all execute: C[i] = A[i] + B[i]  (simultaneously)

The SM scheduler picks a ready warp each clock cycle.
If Warp 0 is waiting on a memory access → scheduler switches to Warp 1.
This is how GPUs hide memory latency.
```

> [!IMPORTANT]
> Warp size is always **32** on all current NVIDIA hardware. Designing your code around this number (block sizes as multiples of 32, data aligned to warp boundaries) is one of the most impactful optimisations you can make.

---

### 1.5 Warp Divergence — The #1 Performance Killer

Because all 32 threads in a warp must run the same instruction, what happens when some threads take an `if` branch and others take `else`? The GPU must run **both branches serially** — threads that are not supposed to execute a branch are masked off (made idle). This is **warp divergence**, and it can cut throughput in half.

```cpp
// ── BAD — causes warp divergence ──────────────────────────────────────
__global__ void diverge(int* data, int* out) {
    int i = threadIdx.x;

    if (i % 2 == 0) {
        out[i] = data[i] * 2;   // even threads take this path
    } else {
        out[i] = data[i] + 10;  // odd threads take this path
    }
    // The warp runs BOTH branches.
    // Threads not in the active branch are idle.
    // Effective throughput is HALVED.
}

// ── BETTER — restructure data so whole warps take the same path ────────
// If all elements in threads 0-31 are "even-type" and
// elements in threads 32-63 are "odd-type", there is no divergence:
__global__ void no_diverge(int* data, int* out, int threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = data[i] * 2;    // all threads in warp do the same work
}
// Key principle: arrange your data so threads 0-31 (one warp)
// always take the SAME branch. Group similar work together.
```

> [!TIP]
> **Rule:** Design your data layout so threads within a single warp (groups of 32) all follow the same code path. Warp-level uniform conditions have zero divergence cost.

---

### 1.6 Thread Synchronisation — `__syncthreads()`

Threads within a block can share data through **shared memory**, but you must guarantee that all threads finish writing before anyone starts reading. `__syncthreads()` is a **block-level barrier** — no thread can pass it until every thread in the block has reached it.

```cpp
__global__ void sync_example(float* data) {
    __shared__ float sdata[256];   // fast on-chip memory, shared within block
    int tid = threadIdx.x;

    // Phase 1: every thread writes its own value to shared memory
    sdata[tid] = data[tid] * 2.0f;

    // ── BARRIER ──────────────────────────────────────────────────────
    __syncthreads();
    // All 256 threads have now finished writing.
    // Safe to read any element of sdata.

    // Phase 2: read a neighbour's value (safe because of the barrier)
    if (tid > 0) {
        data[tid] = sdata[tid] + sdata[tid - 1];
    }
}
```

**Other synchronisation tools:**

| Function | Scope | Use Case |
| :--- | :--- | :--- |
| `__syncthreads()` | Entire block | Shared memory read-after-write safety |
| `__syncwarp()` | One warp (32 threads) | Warp-level data exchange (faster) |
| `atomicAdd(&x, val)` | Any memory | Thread-safe accumulation across threads |
| `cudaDeviceSynchronize()` | Host ↔ Device | Wait for all GPU work before CPU continues |

---

## Part 2 — Blocks

### 2.1 What Is a Block?

A **block** is a group of threads that run on the same Streaming Multiprocessor (SM). Because they share the same SM, threads within a block can:
- Read and write **shared memory** (fast on-chip cache)
- Synchronise with each other using `__syncthreads()`

Threads in **different blocks cannot** directly share data or synchronise. Think of a block as a classroom — students (threads) in the same classroom can pass notes (shared memory) and wait for each other, but students in different classrooms cannot.

---

### 2.2 Block Dimensions — 1D, 2D, 3D

A block can be configured as 1D, 2D, or 3D to match the shape of your data:

```cpp
// ── 1D block — for arrays and vectors ─────────────────────────────────
dim3 block_1D(256);          // 256 threads along x
                              // threadIdx.x = 0..255

// ── 2D block — for images and matrices ────────────────────────────────
dim3 block_2D(16, 16);       // 16×16 = 256 threads total
                              // threadIdx.x = column (0..15)
                              // threadIdx.y = row    (0..15)

// ── 3D block — for volumes (medical imaging, fluid dynamics) ──────────
dim3 block_3D(8, 8, 4);      // 8×8×4 = 256 threads total

// ── Hardware limits ────────────────────────────────────────────────────
// Max threads per block:           1024
// Max blockDim.x:                  1024
// Max blockDim.y:                  1024
// Max blockDim.z:                  64
// Constraint: blockDim.x × blockDim.y × blockDim.z ≤ 1024
```

---

### 2.3 How to Choose Block Size

Block size is one of the most important performance decisions in CUDA. The core rule: **always use a multiple of 32** (the warp size).

| Block Size | Warps | Notes |
| :---: | :---: | :--- |
| 32 | 1 | Simple but very low SM occupancy |
| 64 | 2 | Still often too low occupancy |
| 128 | 4 | Good starting point for simple kernels |
| **256** | **8** | **Most common — good default choice** |
| 512 | 16 | Higher occupancy when shared memory allows |
| 1024 | 32 | Maximum — often reduces occupancy due to resource pressure |

> [!TIP]
> **Start with 256 threads per block.** Profile with NVIDIA Nsight Compute and adjust. Always use multiples of 32 to avoid partial warps.

---

### 2.4 Shared Memory — The Block's Fast Cache

Every block gets a small, extremely fast piece of **on-chip memory** called shared memory. All threads in the block can read and write it. It is approximately **100× faster** than global (GPU DRAM) memory.

The classic use case is **tiling** — loading a tile of data from slow global memory into fast shared memory once, then reusing it many times:

```cpp
#define TILE 16

__global__ void matmul(float* A, float* B, float* C, int N) {
    // Declare tiles in shared memory (lives on the SM chip — very fast)
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {

        // Step 1: all threads cooperate to load one tile from global memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();   // wait — everyone must finish loading before computing

        // Step 2: compute using fast shared memory — no global memory access
        for (int k = 0; k < TILE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();   // wait before overwriting tiles in next iteration
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

Without tiling, each element of A and B is read from global memory multiple times. With tiling, each tile is read **once** into shared memory and reused TILE times — dramatically reducing global memory bandwidth consumption.

---

### 2.5 Occupancy — Keeping the SM Busy

**Occupancy** is the ratio of active warps on an SM to the maximum number of warps the SM supports. High occupancy lets the warp scheduler switch to another warp while one is stalled on memory — hiding memory latency.

Occupancy is limited by three resources per SM:

| Resource | Effect |
| :--- | :--- |
| **Registers** | More registers per thread → fewer threads fit → lower occupancy |
| **Shared memory** | More shared memory per block → fewer blocks per SM → lower occupancy |
| **Block size** | Too small → too many blocks to manage. Too large → can't fill SM |

```cpp
// Auto-tune: let CUDA find the optimal block size for your kernel
int min_grid_size, optimal_block_size;

cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size,
    &optimal_block_size,
    myKernel,   // pointer to your kernel function
    0,          // dynamic shared memory bytes per block (0 if none)
    0           // maximum block size limit (0 = no limit)
);

printf("Optimal block size: %d threads\n", optimal_block_size);
printf("Minimum grid size:  %d blocks\n",  min_grid_size);
```

> [!NOTE]
> Target occupancy of **50–75%** for most kernels. 100% occupancy is not always achievable and not always beneficial — register spilling (overflow to slow local memory) can be worse than reduced occupancy.

---

## Part 3 — Grids

### 3.1 What Is a Grid?

A **grid** is the complete collection of all blocks launched by a single kernel call. If a block is a classroom, the grid is the entire school. The total thread count for a kernel launch is:

```
Total Threads = gridDim.x × gridDim.y × gridDim.z
              × blockDim.x × blockDim.y × blockDim.z
```

---

### 3.2 Grid Dimensions — 1D, 2D, 3D

```cpp
// ── 1D grid — for a 1-million-element array ────────────────────────────
int N       = 1024 * 1024;             // 1,048,576 elements
int threads = 256;
int blocks  = (N + threads - 1) / threads;  // ceiling division = 4096 blocks
// Total: 4096 blocks × 256 threads = 1,048,576 threads

dim3 grid_1D(blocks);

// ── 2D grid — for a 1024×1024 image ───────────────────────────────────
dim3 block2D(16, 16);                  // 256 threads per block
dim3 grid2D(1024 / 16, 1024 / 16);    // 64×64 = 4096 blocks

// ── 3D grid — for a 64×64×64 volume ───────────────────────────────────
dim3 block3D(8, 8, 4);
dim3 grid3D(64 / 8, 64 / 8, 64 / 4); // 8×8×16 = 1024 blocks

// ── Hardware limits for grid dimensions ───────────────────────────────
// gridDim.x ≤ 2³¹ − 1   (very large)
// gridDim.y ≤ 65,535
// gridDim.z ≤ 65,535
```

---

### 3.3 Mapping Data to the Grid — Three Essential Patterns

The art of CUDA programming is matching your data structure to the right grid and block layout:

```cpp
// ── PATTERN 1: 1D array ───────────────────────────────────────────────
__global__ void process_1D(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] = sqrtf(data[i]);
}
// Launch:
// process_1D<<<(N + 255) / 256, 256>>>(data, N);


// ── PATTERN 2: 2D matrix / image ─────────────────────────────────────
__global__ void process_2D(float* img, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x → column
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // y → row
    if (col < W && row < H) {
        int idx = row * W + col;       // convert 2D position to 1D memory index
        img[idx] = img[idx] * 0.5f;   // halve brightness of each pixel
    }
}
// Launch:
// dim3 block(16, 16);
// dim3 grid((W + 15) / 16, (H + 15) / 16);
// process_2D<<<grid, block>>>(img, W, H);


// ── PATTERN 3: grid-stride loop (recommended for large N) ────────────
// Instead of launching exactly N threads, launch a fixed number and
// have each thread process multiple elements. More flexible and often faster.
__global__ void process_stride(float* data, int N) {
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;   // total threads in the grid

    while (i < N) {
        data[i] = sqrtf(data[i]);
        i += stride;   // jump to the next element for this thread
    }
}
// Launch with a fixed, hardware-optimal grid size:
// int threads = 256;
// int blocks  = min((N + threads - 1) / threads, 2048);
// process_stride<<<blocks, threads>>>(data, N);
```

> [!TIP]
> **Prefer the grid-stride loop (Pattern 3)** for production code. It handles any value of N without risking out-of-bounds access, and gives the compiler better optimisation opportunities.

---

### 3.4 How Blocks Are Scheduled on SMs

You do not control which SM runs which block — the GPU hardware scheduler decides. This is intentional: it is what allows CUDA programs to **scale automatically** across GPU generations with different SM counts.

```
RTX 4090 has 128 SMs
You launch 4096 blocks

GPU scheduler distributes blocks across SMs:
  SM 0   → blocks  0, 128, 256, 384 ...
  SM 1   → blocks  1, 129, 257, 385 ...
  ...
  SM 127 → blocks 127, 255, 383, 511 ...

Blocks may run in ANY order. You cannot predict which SM gets which block.
```

Because of this, **blocks must be independent** — they cannot share data or synchronise with each other directly. If Block 0 tries to wait for Block 1, you get a **deadlock**.

```cpp
// ── WRONG — blocks cannot communicate directly ─────────────────────────
// Block 0: shared_flag = 1;
// Block 1: while (shared_flag != 1) { }   // DEADLOCK — Block 1 may run
//                                          // on a different SM that
//                                          // never sees Block 0's write

// ── CORRECT — use two separate kernel launches ─────────────────────────
kernel_phase_1<<<grid, block>>>(d_data);   // all blocks complete
cudaDeviceSynchronize();                   // CPU waits for all GPU work to finish
kernel_phase_2<<<grid, block>>>(d_data);   // fresh launch — reads phase 1 results
```

> [!IMPORTANT]
> If blocks need to communicate, **split the work into two kernel launches**. The synchronisation point is `cudaDeviceSynchronize()` between launches. This is also why kernel launches are so lightweight — the GPU can freely schedule blocks anywhere.

---

## Summary

| Concept | Key Number | Rule of Thumb |
| :--- | :---: | :--- |
| Threads per warp | **32** | Always design around warp boundaries |
| Max threads per block | **1024** | Start with 256; profile and adjust |
| Warp divergence cost | up to **2×** slowdown | Ensure threads in same warp take same path |
| Shared memory speed | ~**100×** faster than global | Cache data your block accesses multiple times |
| Target occupancy | **50–75%** | Balance registers, shared memory, and block size |
| Block independence | absolute | Never sync between blocks; use separate kernels |

---

## Deep Learning Performance References

The concepts in this module directly underpin how modern deep learning frameworks achieve GPU acceleration:

- **CUDA C++ Programming Guide — Execution Model:** The authoritative reference for the thread hierarchy, warp execution, and synchronisation primitives covered in this module. [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)
- **GPU Performance Background:** Understand how thread-level and warp-level decisions map to math-limited vs memory-limited performance in neural networks. [NVIDIA Deep Learning Performance User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- **CUDA Best Practices Guide — Execution Configuration:** Practical recommendations for grid and block sizing, occupancy tuning, and latency hiding in production kernels. [NVIDIA CUDA Best Practices — Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#execution-configuration)
- **Occupancy Calculator:** Interactive tool to analyse how your register and shared memory usage affects warp occupancy. [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)

---

➡️ **Next:** [Module 06 — Memory Model](../module-06-memory-model/README.md)
