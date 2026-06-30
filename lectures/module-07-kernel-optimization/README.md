# Module 07 — Kernel Optimization

> **Estimated time:** ~5 hours  
> **Prerequisites:** [Module 06 — Memory Model](../module-06-memory-model/README.md)  
> **Next module:** [Module 08 — Streams & Concurrency](../module-08-streams-concurrency/README.md) *(coming soon)*

---

## What You Will Learn

- CUDA function qualifiers — `__global__`, `__device__`, `__host__`, and `__host__ __device__`
- Kernel launch configuration — the full `<<<>>>` syntax
- Asynchronous execution and CUDA streams
- Error handling — never skip checking CUDA API returns
- A complete 2D Gaussian blur kernel combining shared and constant memory
- A practical kernel optimisation checklist

---

## Part 1 — Kernels Overview

A **kernel** is a function written in CUDA C that runs on the GPU. It is the bridge between your CPU program and the GPU's massive parallelism. Understanding kernels deeply — how they are declared, launched, and optimised — is the core skill of CUDA programming.

---

## Part 2 — CUDA Function Qualifiers

CUDA uses special keywords to tell the compiler **where** a function runs and **who** can call it.

| Qualifier | Executes On | Called From | Returns | Key Rules |
| :--- | :--- | :--- | :--- | :--- |
| `__global__` | GPU | CPU (or GPU*) | `void` only | Entry point — launched with `<<<>>>` |
| `__device__` | GPU | GPU only | any type | Helper functions for kernels |
| `__host__` | CPU | CPU only | any type | Same as normal C function |
| `__host__ __device__` | Both | Both | any type | Compiles for both — math helpers |

```cpp
// All three qualifiers working together
__device__ float square(float x) {          // GPU helper
    return x * x;
}

__host__ __device__ float clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x); // works on CPU and GPU
}

__global__ void process(float* data, int N) { // the kernel
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = square(data[i]);          // calls __device__ function
        data[i] = clamp(v, 0.0f, 1.0f);     // calls __host__ __device__
    }
}

// CPU code:
float x = clamp(5.0f, 0.0f, 1.0f);          // same function, runs on CPU
```

---

## Part 3 — Kernel Launch Configuration

The triple angle bracket syntax `<<<>>>` has up to **four** arguments:

```cpp
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args);
//       ^^^^^^^  ^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^
//       [1]      [2]       [3] optional    [4] optional

// [1] gridDim:        int or dim3 — how many blocks
// [2] blockDim:       int or dim3 — threads per block
// [3] sharedMemBytes: bytes of dynamic shared mem per block (default 0)
// [4] stream:         which CUDA stream (default 0 = default stream)
```

```cpp
// All valid launch configs:
add<<<1000, 256>>>(d_A, d_B, d_C, N);
add<<<dim3(100, 10), dim3(16, 16)>>>(d_img, W, H);
add<<<1000, 256, 1024>>>(d_A, N);              // 1 KB dynamic shared mem
add<<<1000, 256, 0, myStream>>>(d_A, N);       // async stream
```

> [!IMPORTANT]
> Kernel launches are **asynchronous** — the CPU continues immediately after launching. GPU work and CPU work can overlap in time.

---

## Part 4 — Asynchronous Execution & Streams

Kernel launches return immediately — the CPU does not wait for the GPU. **Streams** are sequences of GPU operations that execute in order. Operations in **different** streams can overlap.

```cpp
// ── DEFAULT STREAM — all ops are sequential ───────────────────────────
kernel_A<<<grid, block>>>();        // starts
kernel_B<<<grid, block>>>();        // waits for A to finish
cudaDeviceSynchronize();            // CPU waits for all GPU work

// ── MULTIPLE STREAMS — overlap kernels and memcpy ─────────────────────
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// These two kernels can run CONCURRENTLY on the GPU!
kernel_A<<<grid, block, 0, stream1>>>(d_A);
kernel_B<<<grid, block, 0, stream2>>>(d_B);

// Async memcpy — overlaps with GPU kernel in another stream!
cudaMemcpyAsync(d_C, h_C, bytes, cudaMemcpyHostToDevice, stream1);

cudaStreamSynchronize(stream1);     // wait for stream1 only
cudaStreamSynchronize(stream2);     // wait for stream2 only
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

> [!NOTE]
> Streams are covered in depth in [Module 08](../module-08-streams-concurrency/README.md). For now, remember: default stream = sequential; multiple streams = concurrency.

---

## Part 5 — Error Handling

CUDA functions return `cudaError_t` — always check it. Kernel errors are **deferred** and are often only caught on the next `cudaDeviceSynchronize()` call.

```cpp
// Macro for safe CUDA calls (add to your project!)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Use it on every CUDA API call:
CUDA_CHECK(cudaMalloc(&d_data, bytes));
CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

// Kernel error checking:
myKernel<<<grid, block>>>(d_data, N);
CUDA_CHECK(cudaGetLastError());         // check launch error
CUDA_CHECK(cudaDeviceSynchronize());   // check kernel execution error
```

### Common CUDA errors

| Error | Typical Cause |
| :--- | :--- |
| `cudaErrorInvalidValue` | Bad argument (NULL pointer, wrong size) |
| `cudaErrorInvalidDevice` | Device ID out of range |
| `cudaErrorOutOfMemory` | Not enough VRAM |
| `cudaErrorInvalidConfiguration` | Block size > 1024 |
| `cudaErrorLaunchTimeout` | Kernel ran too long (Windows TDR) |

---

## Part 6 — Complete Kernel: 2D Gaussian Blur

Putting everything together — a real kernel using shared memory, constant memory, proper bounds checking, and 2D grid mapping:

```cpp
// Complete 2D Gaussian blur — shared + constant memory + 2D grid
#define RADIUS 2    // blur radius
#define BLOCK  16    // block tile size

// Gaussian weights in constant memory (fast broadcast)
__constant__ float gauss[5] = {0.06f, 0.24f, 0.40f, 0.24f, 0.06f};

__global__ void gaussBlur(float* in, float* out, int W, int H) {
    // Shared memory tile (includes halo for border pixels)
    __shared__ float tile[BLOCK + 2 * RADIUS][BLOCK + 2 * RADIUS];

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * BLOCK + tx;
    int row = blockIdx.y * BLOCK + ty;

    // Load tile (each thread loads one pixel + some load halo)
    int sc = col - RADIUS, sr = row - RADIUS;
    sc = max(0, min(sc, W - 1));   // clamp to image bounds
    sr = max(0, min(sr, H - 1));
    tile[ty][tx] = in[sr * W + sc];

    __syncthreads();   // wait for all loads

    // Compute blur (only for interior threads)
    if (col < W && row < H) {
        float sum = 0.0f;
        for (int dy = -RADIUS; dy <= RADIUS; dy++)
            for (int dx = -RADIUS; dx <= RADIUS; dx++)
                sum += tile[ty + RADIUS + dy][tx + RADIUS + dx]
                     * gauss[dy + RADIUS] * gauss[dx + RADIUS];
        out[row * W + col] = sum;
    }
}

// Launch:
// dim3 block(BLOCK, BLOCK);
// dim3 grid((W + BLOCK - 1) / BLOCK, (H + BLOCK - 1) / BLOCK);
// gaussBlur<<<grid, block>>>(d_in, d_out, W, H);
```

This kernel demonstrates:

- **2D grid/block** mapping for images
- **Shared memory tiling** with a halo region for convolution
- **Constant memory** for filter weights (broadcast across the warp)
- **`__syncthreads()`** between load and compute phases
- **Bounds checking** on output writes

---

## Part 7 — Kernel Optimisation Checklist

Use this checklist before and after profiling every performance-critical kernel:

| # | Optimisation | What to Do |
| :---: | :--- | :--- |
| 1 | **Coalesced memory** | Thread `i` accesses element `i` — stride-1 pattern |
| 2 | **Shared memory cache** | Data reused by block → load once to `__shared__` |
| 3 | **No warp divergence** | Threads in same warp take same code path |
| 4 | **Bounds check** | Always guard: `if (i < N)` |
| 5 | **`__syncthreads()`** | After every shared memory write, before read |
| 6 | **Block size = 32×n** | Always a multiple of 32 (warp size) |
| 7 | **Error checking** | `CUDA_CHECK()` on every API call |
| 8 | **Stream overlap** | Use streams to hide memory transfer latency |
| 9 | **Profiling** | Use `nsys` / `ncu` before optimising blindly |

```cpp
// Example: checklist applied to a simple vector kernel
__global__ void optimised_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // coalesced index
    if (i < N) {                                      // bounds check
        C[i] = A[i] + B[i];                           // no divergence
    }
}
// Launch: optimised_add<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
//         block size 256 = 8 warps ✓
```

> [!TIP]
> **Profile first, optimise second.** Nsight Systems (`nsys`) shows timeline and overlap; Nsight Compute (`ncu`) shows per-kernel memory throughput, occupancy, and stall reasons.

---

## Summary

| Topic | Key Takeaway |
| :--- | :--- |
| Function qualifiers | `__global__` = kernel entry; `__device__` = GPU helper |
| Launch syntax | `<<<grid, block, sharedBytes, stream>>>` |
| Async execution | CPU does not wait unless you synchronise |
| Streams | Separate streams enable kernel and memcpy overlap |
| Error handling | Check every API call + `cudaGetLastError()` after launch |
| Real kernel | Gaussian blur = tiling + constant weights + 2D grid |
| Optimisation | Coalesce → share → avoid divergence → profile |

---

## Deep Learning Performance References

- **CUDA C++ Programming Guide — Kernel Launch:** Full launch configuration reference. [NVIDIA CUDA Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)
- **CUDA C++ Programming Guide — CUDA C Runtime:** Streams, events, and asynchronous operations. [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
- **CUDA Best Practices Guide — Performance Guidelines:** Systematic optimisation workflow. [NVIDIA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- **Nsight Compute / Nsight Systems:** Production profiling tools. [NVIDIA Nsight](https://developer.nvidia.com/tools-overview)

---

➡️ **Next:** [Module 08 — Streams & Concurrency](../module-08-streams-concurrency/README.md) *(coming soon)*
