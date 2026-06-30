# Module 06 — Memory Model

> **Estimated time:** ~4 hours  
> **Prerequisites:** [Module 05 — Threads, Blocks & Grids](../module-05-threads-blocks-grids/README.md)  
> **Next module:** [Module 07 — Kernel Optimization](../module-07-kernel-optimization/README.md)

---

## What You Will Learn

- Why memory is the #1 bottleneck in GPU performance
- All six CUDA memory types and when to use each
- Registers — automatic, fastest storage per thread
- Shared memory — static vs. dynamic allocation and bank conflicts
- Global memory — coalescing, the most important global memory rule
- Constant memory — read-only broadcast for filter coefficients and lookup tables
- Texture memory — cached 2D spatial access and hardware interpolation
- Unified memory — simplified programming with migration trade-offs

---

## Part 1 — The Memory Hierarchy

Memory is the most important topic for GPU performance. The compute cores are fast — they are almost always waiting for data. Understanding which memory to use when is what separates a **10× speedup** from a **100× speedup**.

| Memory | Location | Speed | Size | Scope | Lifetime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Registers** | On-chip SM | ~Instant | ~64 KB/SM | 1 thread | Thread |
| **Shared Memory** | On-chip SM | ~1–4 cycles | 48–164 KB/SM | Block | Block |
| **L1 Cache** | On-chip SM | ~10–30 cycles | Auto-managed | Block/SM | Auto |
| **L2 Cache** | On-chip | ~100 cycles | 40–80 MB | All threads | App |
| **Global Memory** | Off-chip | ~200–800 cycles | 8–80 GB | All threads | App |
| **Constant Memory** | Off-chip* | ~1 cycle* | 64 KB | All (R/O) | App |
| **Texture Memory** | Off-chip* | Cached | Unlimited | All (R/O) | App |
| **Unified Memory** | CPU + GPU | Variable | CPU RAM | CPU + GPU | Manual |

> [!IMPORTANT]
> **Memory access pattern summary:** Registers (fastest, automatic) → Shared (fast, manual) → Constant (fast if broadcast) → Global (slow, coalesce!) → Unified (flexible, slowest).

---

## Part 2 — Registers

### 2.1 The Fastest Memory

Registers are on-chip storage assigned to each thread. They are effectively zero-latency — the fastest possible memory. Every local variable in your kernel uses a register automatically. No special syntax is needed.

```cpp
// Registers — local variables automatically use registers
__global__ void register_demo(float* data, int N) {
    int i = threadIdx.x;          // ← stored in register
    float x = data[i];            // ← stored in register
    float y = x * x;              // ← register operation (fast!)
    float z = y + x + 1.0f;       // ← register operation
    data[i] = z;                  // write back to global
}
```

### 2.2 Register Spilling

If your kernel uses too many variables, CUDA **spills** them to **local memory** (off-chip, slow). Check for spills with:

```bash
nvcc --ptxas-options=-v my_kernel.cu
```

Look for output like: `spills to local memory`.

> [!TIP]
> Keep kernels lean — fewer live variables per thread means more registers available and higher occupancy. Profile before micro-optimising register usage.

---

## Part 3 — Shared Memory

Shared memory was introduced in [Module 05](../module-05-threads-blocks-grids/README.md). It is on-chip and roughly **100× faster** than global memory. Two ways to declare it:

### 3.1 Static Allocation (compile-time size)

```cpp
__global__ void static_shared() {
    __shared__ float sdata[256];   // 256 × 4 bytes = 1 KB
    __shared__ int flags[32];      // another 128 bytes
    // Both arrays share the block's shared memory pool
}
```

### 3.2 Dynamic Allocation (launch-time size)

```cpp
__global__ void dynamic_shared(int n) {
    extern __shared__ float sdata[];   // size unknown at compile time
    sdata[threadIdx.x] = threadIdx.x * 2.0f;
    __syncthreads();
    // ...
}

// Launch with dynamic shared memory size (3rd argument in angle brackets):
int shared_bytes = 256 * sizeof(float);
dynamic_shared<<<grid, 256, shared_bytes>>>(256);
//                              ^^^^^^^^^^^^ size in bytes
```

### 3.3 Bank Conflicts

Shared memory is split into **32 banks**. If multiple threads in a warp access different addresses that map to the **same bank**, those accesses are **serialised** (slow).

```
BAD:  thread 0 → sdata[0], thread 1 → sdata[32]  (same bank if stride = 32)
GOOD: thread 0 → sdata[0], thread 1 → sdata[1]   (stride-1, no conflict)
```

> [!TIP]
> **Avoid bank conflicts** by ensuring threads access stride-1 elements. Pad shared arrays when necessary (e.g. `TILE+1` columns in matrix tiling).

---

## Part 4 — Global Memory

Global memory is the large off-chip VRAM (your 8/12/24 GB GPU memory). It is slow (~200 cycles latency) but huge. Every `cudaMalloc` allocates global memory.

The key to fast global memory access is **coalescing**.

### 4.1 Memory Coalescing — The #1 Global Memory Rule

The GPU fetches global memory in cache lines of **128 bytes**. If 32 consecutive threads (a warp) each access consecutive memory addresses, the entire warp's data fits in 1–4 cache lines — called **coalesced access**. If they access scattered addresses, each needs a separate fetch — terrible performance.

```cpp
// ── COALESCED — consecutive threads, consecutive addresses ─────────────
// Thread 0 reads data[0], Thread 1 reads data[1], ...
__global__ void coalesced(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = data[i] * 2.0f;   // stride-1 = COALESCED = fast
}

// ── STRIDED — threads jump around in memory ─────────────────────────────
__global__ void strided(float* data) {
    int i = threadIdx.x * 32;   // Thread 0→[0], Thread 1→[32], Thread 2→[64]
    data[i] = data[i] * 2.0f;   // 32 separate cache lines = ~32× slower!
}

// ── RANDOM — worst case ───────────────────────────────────────────────
__global__ void random_access(float* data, int* indices) {
    int i = indices[threadIdx.x];   // unpredictable addresses
    data[i] = data[i] * 2.0f;       // no coalescing possible
}
```

> [!IMPORTANT]
> **Golden rule:** Thread N should access element N of your array. Consecutive threads → consecutive addresses → coalesced = maximum bandwidth.

---

## Part 5 — Constant Memory

Constant memory is **64 KB** of special read-only memory with a dedicated cache. When all threads in a warp read the **same address**, it is served from cache in a single **broadcast** — extremely fast. Perfect for lookup tables, filter coefficients, and model parameters.

```cpp
// Constant memory — broadcast read for filter coefficients
// Declare in global scope (not inside any function)
__constant__ float kernel_weights[25];   // 5×5 convolution filter
__constant__ int lut[256];               // lookup table

void setup() {
    float weights[25] = { /* filter values */ };
    // Copy from CPU to constant memory (one time)
    cudaMemcpyToSymbol(kernel_weights, weights, 25 * sizeof(float));
}

__global__ void conv(float* img, float* out, int W, int H) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < 25; k++)
        sum += img[/* neighbour index */] * kernel_weights[k];  // all threads read SAME k
    out[row * W + col] = sum;
}
```

> [!NOTE]
> Constant memory is fastest when all threads in a warp read the **same** address. If each thread reads a different constant address, performance degrades toward uncached global memory.

---

## Part 6 — Texture Memory

Texture memory is cached and optimised for **2D spatial locality**. Normal caches assume you will access nearby linear addresses. The texture cache assumes you will access nearby **2D** addresses — perfect for images. It also supports hardware interpolation and boundary clamping.

Modern CUDA uses **texture objects** (not the legacy `texture<>` reference API):

```cpp
// Texture memory — hardware bilinear interpolation
cudaTextureObject_t tex;
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cudaArray;

cudaTextureDesc texDesc = {};
texDesc.addressMode[0] = cudaAddressModeClamp;   // clamp at edges
texDesc.filterMode = cudaFilterModeLinear;       // bilinear interpolation!
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 1;                    // use 0.0–1.0 coords

cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

__global__ void sample(cudaTextureObject_t tex, float* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = (x + 0.5f) / W;
    float v = (y + 0.5f) / H;
    out[y * W + x] = tex2D<float>(tex, u, v);   // hardware-interpolated!
}
```

---

## Part 7 — Unified Memory

Unified Memory lets you use a **single pointer** that both CPU and GPU can access. CUDA automatically migrates pages between CPU and GPU as needed. Great for getting started, but adds overhead from page migrations.

```cpp
// ── Traditional: separate CPU and GPU buffers + manual memcpy ─────────
float *h_data = (float*)malloc(N * sizeof(float));
float *d_data;
cudaMalloc(&d_data, N * sizeof(float));
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_data);
cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

// ── Unified Memory: one pointer, no manual copies ─────────────────────
float *data;
cudaMallocManaged(&data, N * sizeof(float));   // ← single allocation

for (int i = 0; i < N; i++) data[i] = i;       // CPU writes
kernel<<<grid, block>>>(data);                  // GPU reads/writes
cudaDeviceSynchronize();                        // wait for GPU
printf("%f\n", data[0]);                        // CPU reads — just works!
cudaFree(data);                                 // one free for both
```

> [!TIP]
> Page faults on first access add latency. Use `cudaMemPrefetchAsync()` to pre-migrate pages to the GPU before kernel launch for better performance.

---

## Summary

| Memory Type | Speed | When to Use |
| :--- | :--- | :--- |
| **Registers** | Fastest | Automatic — local variables, temporaries |
| **Shared** | ~100× global | Data reused within a block (tiling) |
| **Constant** | Fast (broadcast) | Read-only coefficients, LUTs — same value per warp |
| **Global** | Slow | Large datasets — always coalesce accesses |
| **Texture** | Cached 2D | Images, spatial sampling, interpolation |
| **Unified** | Flexible, slower | Prototyping, simpler host code |

---

## Deep Learning Performance References

- **CUDA C++ Programming Guide — Memory Hierarchy:** Official description of global, shared, constant, and texture memory. [NVIDIA CUDA Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- **CUDA Best Practices Guide — Memory Optimisations:** Coalescing, shared memory, and occupancy trade-offs. [NVIDIA Best Practices — Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- **Unified Memory Programming:** When managed memory helps and when explicit copies win. [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)

---

➡️ **Next:** [Module 07 — Kernel Optimization](../module-07-kernel-optimization/README.md)
