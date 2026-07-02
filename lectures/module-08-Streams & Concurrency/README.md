# Module 08 — Streams & Concurrency

> **Estimated time:** ~5 hours  
> **Prerequisites:** [Module 07-kernel-optimization](/lectures/module-07-kernel-optimization/README.md)  
> **Next module:** [module 09-CUDA-libraries](/lectures/module-09-CUDA-Libraries/README.md)


# CUDA Streams & Concurrency — Deep Dive with Simple Examples

Explaining CUDA streams, pinned memory, and cudaEvent timing using everyday analogies and beginner-friendly code examples.

---

## PART 1: Understanding the Default Behavior (No Streams)

### The Kitchen Analogy

Imagine a **GPU is a kitchen with a chef**, and the **CPU is a waiter** taking orders.

By default, CUDA uses something called the **"default stream"** (also called stream 0). Everything you send to the GPU goes into this **one single queue**, like one waiter placing orders one at a time and waiting for each to complete before placing the next.

### Simple Example — Copy, Compute, Copy Back

```cpp
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);  // Step 1: send ingredients to kitchen
myKernel<<<1, 256>>>(d_A);                            // Step 2: chef cooks
cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);  // Step 3: bring dish back to table
```

Here's what happens on a timeline:

```
Time  ---->
[ Copy to GPU ][ Compute ][ Copy back ]
    2 sec         3 sec       2 sec
```

**Total time = 2 + 3 + 2 = 7 seconds.**

Nothing overlaps. The copy engine is idle during compute. The compute cores are idle during copying. This is wasteful — like a chef standing around while ingredients are being delivered, instead of prepping something else.

---

## PART 2: Introducing Streams — Multiple Independent Queues

### The Multi-Lane Highway Analogy

A **stream** is like a **lane on a highway**. Cars (GPU operations) in the *same lane* must travel in order, one after another. But cars in *different lanes* can drive **at the same time**.

```cpp
cudaStream_t streamA, streamB;
cudaStreamCreate(&streamA);
cudaStreamCreate(&streamB);
```

Now you have two "lanes." Anything you put into `streamA` runs in order relative to other `streamA` tasks — but can run **simultaneously** with whatever is in `streamB`.

### Very Simple Example: Two Independent Tasks

Say you have two completely separate arrays to double: `A` and `B`. They don't depend on each other at all.

```cpp
// WITHOUT streams (default stream — everything sequential)
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
doubleKernel<<<1, 256>>>(d_A);
cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
doubleKernel<<<1, 256>>>(d_B);
cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
```

Timeline (each step = 1 second, 6 steps total = **6 seconds**):
```
[Copy A][Compute A][Copy A back][Copy B][Compute B][Copy B back]
```

Now with **streams**:

```cpp
// WITH streams — A and B can overlap
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, streamA);
doubleKernel<<<1, 256, 0, streamA>>>(d_A);
cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, streamA);

cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, streamB);
doubleKernel<<<1, 256, 0, streamB>>>(d_B);
cudaMemcpyAsync(h_B, d_B, size, cudaMemcpyDeviceToHost, streamB);
```

Timeline (things now overlap because A's compute can happen while B is copying, etc.):
```
Stream A: [Copy A][Compute A][Copy A back]
Stream B:         [Copy B][Compute B][Copy B back]
```

Roughly **4 seconds instead of 6** — because while A is computing, B's data is already copying in. This is called **pipelining**.

**Simple rule to remember:** Streams only help if tasks are truly *independent* (A doesn't need B's result, and vice versa). If B depends on A's output, they must stay in the same stream or be synchronized.

---

## PART 3: Pinned Memory — Explained Super Simply

### Why does async copying even need special memory?

Your computer's RAM is managed by the operating system. Normal memory (from `malloc`) is **pageable** — meaning the OS can move it to a different physical location in RAM whenever it wants (e.g., to free up space, or swap to disk).

**Analogy:** Imagine you order a package for pickup at a giant warehouse, but the warehouse workers keep shuffling boxes to different shelves to reorganize. A delivery truck (the GPU's DMA copy engine) that wants to grab your box directly, without asking a person, **can't safely do that** — because the box might have moved mid-grab!

So, for a **pageable** memory copy, CUDA has to do this internally:

```
Your data (pageable) → CUDA copies it to a temporary pinned buffer → GPU copies from that buffer
```

This extra "middleman" step:
1. Adds time
2. **Blocks true async behavior** — the CPU thread might have to wait for that staging copy

### Pinned Memory = Reserved, Non-Movable Memory

```cpp
float *h_data;
cudaMallocHost(&h_data, 1000 * sizeof(float));  // pinned allocation
```

**Analogy continued:** Pinned memory is like reserving a locker with a fixed, guaranteed location — number 42, ground floor, never moved. Now the delivery truck (DMA engine) can drive straight to locker 42 and grab it directly, with zero risk of it having moved. This is what makes truly asynchronous, overlapped transfers possible.

### Side-by-Side Comparison

| | Pageable Memory (`malloc`) | Pinned Memory (`cudaMallocHost`) |
|---|---|---|
| Can OS move it? | Yes, anytime | No, locked in place |
| `cudaMemcpyAsync` truly async? | Often NO (silently blocks) | YES |
| Allocation speed | Fast | Slightly slower |
| Amount available | Lots (just system RAM) | Limited (careful not to over-allocate) |

### Tiny Code Example Showing the Difference

```cpp
// Pageable memory — DON'T expect real overlap
float *h_pageable = (float*)malloc(size);
cudaMemcpyAsync(d_data, h_pageable, size, cudaMemcpyHostToDevice, stream1); 
// ^ this might actually behave like a blocking cudaMemcpy!

// Pinned memory — REAL overlap happens
float *h_pinned;
cudaMallocHost(&h_pinned, size);
cudaMemcpyAsync(d_data, h_pinned, size, cudaMemcpyHostToDevice, stream1);
// ^ this truly returns immediately, letting CPU move on while GPU copies in background
```

**Rule of thumb:** If you want overlap (streams doing real concurrent work), **always use pinned memory** for the data you're transferring asynchronously.

---

## PART 4: cudaEvent Timing — Explained Super Simply

### Why can't I just use a normal stopwatch?

```cpp
clock_t start = clock();
myKernel<<<1, 256>>>(data);
clock_t end = clock();
printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
```

This is **wrong**, and here's why: `myKernel<<<...>>>()` is **asynchronous**. The CPU launches it and immediately continues to the next line — it does NOT wait for the GPU to finish. So `clock()` measures almost nothing — just the tiny time it took to *launch* the kernel, not how long the GPU actually took to compute.

**Analogy:** Imagine you tell a chef "start cooking pasta" and you immediately check your watch and say "done!" without actually waiting for the pasta to cook. That's what a CPU timer around an async kernel launch does — it's measuring the wrong thing.

### The Fix: cudaEvents (GPU-Side Timestamps)

A `cudaEvent_t` is a **flag you drop directly into the GPU's task queue (stream)**. The GPU itself marks the exact time it reaches that flag — no guessing from the CPU side.

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);              // drop flag "start" into the queue
myKernel<<<1, 256>>>(data);          // the actual work
cudaEventRecord(stop);               // drop flag "stop" into the queue

cudaEventSynchronize(stop);          // CPU waits here until GPU reaches "stop" flag

float ms;
cudaEventElapsedTime(&ms, start, stop);  // ask: how much time passed between the 2 flags?
printf("Kernel took %.3f ms\n", ms);
```

**Analogy:** Think of a train (the GPU stream) traveling down a track. You place a sensor at "Station Start" and another at "Station Stop." As the train passes each sensor, it logs the exact timestamp. Later, you subtract the two timestamps to know exactly how long the train took between stations — accurate, because it's measured *by the track itself*, not by someone glancing at their watch from far away.

### Step-by-Step Breakdown of Each Function

| Function | What it does in plain English |
|---|---|
| `cudaEventCreate(&event)` | Create a blank "flag" object (not yet placed anywhere) |
| `cudaEventRecord(event)` | Drop that flag into the current stream's queue, right at this point |
| `cudaEventRecord(event, stream)` | Same, but explicitly into a specific stream |
| `cudaEventSynchronize(event)` | CPU pauses and waits until the GPU has actually reached that flag |
| `cudaEventElapsedTime(&ms, start, stop)` | Calculates milliseconds between two flags |

---

## PART 5: Full Combined Example

Timing how much faster **two streams doing independent work** are, using pinned memory and events.

```cpp
#include <cstdio>

__global__ void doubleValues(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2.0f;
}

int main() {
    int n = 1 << 20;           // ~1 million elements
    size_t size = n * sizeof(float);

    // 1. Allocate PINNED host memory (so async copies actually overlap)
    float *h_A, *h_B;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    for (int i = 0; i < n; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // 2. Allocate device memory
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // 3. Create two streams (two "lanes")
    cudaStream_t streamA, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    // 4. Create events to time everything
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 5. Record start
    cudaEventRecord(start);

    // 6. Launch work in BOTH streams — they can overlap
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, streamA);
    doubleValues<<<blocks, threads, 0, streamA>>>(d_A, n);
    cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, streamA);

    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, streamB);
    doubleValues<<<blocks, threads, 0, streamB>>>(d_B, n);
    cudaMemcpyAsync(h_B, d_B, size, cudaMemcpyDeviceToHost, streamB);

    // 7. Record stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   // wait for everything to finish

    // 8. Measure total elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Total time with streams: %.3f ms\n", ms);

    // Cleanup
    cudaFreeHost(h_A); cudaFreeHost(h_B);
    cudaFree(d_A); cudaFree(d_B);
    cudaStreamDestroy(streamA); cudaStreamDestroy(streamB);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
```

If you ran this **without streams** (everything in default stream, sequential) versus **with two streams**, you'd typically see the streamed version finish noticeably faster — because copy and compute overlap between A and B.

---

## Final Simple Recap

1. **Normally**, GPU work happens one step at a time — copy, then compute, then copy back — with lots of idle waiting. Like one cashier serving customers one at a time.

2. **Streams** = separate queues/lanes. Work in different streams can run **at the same time**, as long as they don't depend on each other. Like opening multiple cashier lines.

3. **Pinned memory** = CPU memory that's "locked" in place so the GPU can grab it directly and quickly, without an extra safety-copy step. This is **required** for real overlap to happen — without it, your "async" copies quietly become blocking copies.

4. **cudaEvents** = precise stopwatches placed directly on the GPU's timeline, letting you measure exactly how long GPU operations take — because a normal CPU-side timer is useless due to the asynchronous nature of GPU calls.

Together: **streams give you the opportunity to overlap work, pinned memory makes that overlap actually happen, and cudaEvents let you prove/measure that it worked.**


➡️ **Next module:** [module 09-CUDA-libraries](/lectures/module-09-CUDA-Libraries/README.md)