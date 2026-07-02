<div align="center">

# CUDA Programming Tutorial

**A complete, structured CUDA learning path — from zero GPU knowledge to advanced GPU programming.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com)

</div>

---

## Overview

This tutorial teaches CUDA GPU programming from the ground up. You will start with GPU architecture fundamentals, build solid C++ foundations, and progressively work up to writing high-performance CUDA kernels for real-world applications — including deep learning, image processing, and scientific simulations.

**Prerequisites:** Basic understanding of any programming language. No prior GPU experience required.

---

## What You Will Learn

- **GPU Architecture** — SMs, warps, CUDA cores, memory hierarchy
- **C++ Foundations** — Variables, arrays, functions, pointers, dynamic memory
- **CUDA Programming** — Kernels, launch syntax, thread indexing
- **Memory Management** — Global, shared, constant, texture, registers
- **Kernel Optimization** — Coalescing, tiling, occupancy, warp divergence
- **Deep Learning Performance** — Math-limited vs. memory-limited workloads, arithmetic intensity
- **Streams & Concurrency** — Overlapping compute and data transfer
- **CUDA Libraries** — cuBLAS, cuFFT, Thrust, CUTLASS
- **Multi-GPU Programming** — NVLink, NCCL, peer access

---

## Full Learning Roadmap

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1 — FOUNDATIONS                        │
│                                                                 │
│  Module 01        Module 02         Module 03       Module 04  │
│  GPU Fundamentals C++ Basics        Pointers &      CUDA Setup │
│  Architecture     Variables,Loops   Memory          & Toolkit  │
│  Warps, SMs       Functions,Arrays  Heap,Stack      nvcc,cuDNN │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    PHASE 2 — CORE PROGRAMMING                   │
│                                                                 │
│  Module 05                   Module 06         Module 07       │
│  Threads, Blocks & Grids     Memory Model      Kernel          │
│  1D/2D/3D indexing           Global, Shared    Optimization    │
│  Warp fundamentals           Constant, Texture Coalescing,Tiling│
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    PHASE 3 — INTERMEDIATE                       │
│                                                                 │
│  Module 08                             Module 09               │
│  Streams & Concurrency                 CUDA Libraries           │
│  Overlapping compute & transfer        cuBLAS, cuFFT, Thrust   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    PHASE 4 — ADVANCED                           │
│                                                                 │
│  Module 10                             Module 11               │
│  Multi-GPU Programming                 Advanced CUDA           │
│  NVLink, NCCL, Peer Access             Graphs, Coop. Groups    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Index

### Phase 1 — Foundations

| Module | Title | Topics | Time | Status |
| :---: | :--- | :--- | :---: | :---: |
| **01** | [GPU Fundamentals](lectures/module-01-gpu-fundamentals/README.md) | CPU vs GPU, SMs, Warps, SIMT, Thread hierarchy, Memory hierarchy, Deep Learning performance bounds | ~2h | Ready |
| **02** | [C++ Basics for CUDA](lectures/module-02-cpp-basics/README.md) | Variables, data types, loops, functions, arrays, pointers intro, dynamic memory | ~2h | Ready |
| **03** | [C++ Pointers & Memory](lectures/module-03-pointers-and-memory/README.md) | Stack vs Heap, pointer arithmetic, references, smart pointers, memory safety | ~3h | Ready |
| **04** | [CUDA Setup & Toolchain](lectures/module-04-cuda-setup/README.md) | CUDA Toolkit install (Win/Linux), cuDNN, nvcc compiler, VS Code, Hello CUDA | ~1.5h | Ready |

### Phase 2 — Core Programming

| Module | Title | Topics | Time | Status |
| :---: | :--- | :--- | :---: | :---: |
| **05** | [Threads, Blocks & Grids](lectures/module-05-threads-blocks-grids/README.md) | Threads, warps, warp divergence, `__syncthreads()`, block dims, shared memory, occupancy, grid patterns | ~3h | Ready |
| **06** | [Memory Model](lectures/module-06-memory-model/README.md) | Global, shared, constant, texture, unified memory, coalescing | ~4h | Ready |
| **07** | [Kernel Optimization](lectures/module-07-kernel-optimization/README.md) | Function qualifiers, launch config, streams, error handling, Gaussian blur | ~5h | Ready |

### Phase 3 — Intermediate

| Module | Title | Topics | Time | Status |
| :---: | :--- | :--- | :---: | :---: |
| **08** | [Streams & Concurrency](lectures/module-08-Streams%20&%20Concurrency/README.md) | CUDA streams, pinned memory, cudaEvent timing | ~3h | Planned |
| **09** | CUDA Libraries | Thrust, cuBLAS, cuFFT, cuRAND, CUTLASS | ~4h | Planned |

### Phase 4 — Advanced

| Module | Title | Topics | Time | Status |
| :---: | :--- | :--- | :---: | :---: |
| **10** | Multi-GPU Programming | Peer-to-peer, NVLink, NCCL, data partitioning | ~4h | Planned |
| **11** | Advanced CUDA | CUDA Graphs, cooperative groups, dynamic parallelism | ~5h | Planned |

**Total estimated time: ~36.5 hours**

---

## Student Journey

Follow this path in order for the best learning experience:

```
Step 1  →  Module 01: GPU Fundamentals         (Start here — understand the hardware)
Step 2  →  Module 02: C++ Basics for CUDA      (Build your C++ foundation)
Step 3  →  Module 03: C++ Pointers & Memory    (Master pointers — critical for CUDA)
Step 4  →  Module 04: CUDA Setup & Toolchain   (Install and configure your environment)
Step 5  →  Module 05: Threads, Blocks & Grids  (Write your first real kernels)
Step 6  →  Module 06: Memory Model             (Understand where your data lives)
Step 7  →  Module 07: Kernel Optimization      (Make your code fast)
Step 8  →  Module 08: Streams & Concurrency    (Overlap work for maximum throughput)
Step 9  →  Module 09: CUDA Libraries           (Use battle-tested GPU libraries)
Step 10 →  Module 10: Multi-GPU Programming    (Scale to multiple GPUs)
Step 11 →  Module 11: Advanced CUDA            (Production-grade techniques)
```

---

## Learning Paths

### Beginner Path

> New to GPU programming? Start here.

```
Modules 01 → 02 → 03 → 04 → 05
Time: ~11.5 hours
Goal: Understand GPU architecture and write your first CUDA kernels
```

### Intermediate Path

> Comfortable writing kernels? Level up.

```
Modules 05 → 06 → 07 → 08
Time: ~15 hours
Goal: Write fast, optimized CUDA code
```

### Advanced Path

> Ready for production-grade CUDA?

```
Modules 09 → 10 → 11
Time: ~13 hours
Goal: Multi-GPU, custom libraries, and advanced techniques
```

### AI/ML Engineer Fast Track

> Already know C++? Jump straight to GPU-accelerated deep learning.

```
Modules 01 → 04 → 05 → 06 → 09
Time: ~14.5 hours
Goal: GPU-accelerated neural network operations with cuBLAS and cuDNN
```

---

## Projects Timeline

| # | Project | After Module | Difficulty |
| :---: | :--- | :---: | :---: |
| 01 | Vector Addition | 04 | Beginner |
| 02 | Matrix Addition | 05 | Beginner |
| 03 | Matrix Multiplication | 06 | Intermediate |
| 04 | Image Processing Pipeline | 06 | Intermediate |
| 05 | 2D Convolution | 07 | Advanced |
| 06 | Particle Simulation | 08 | Advanced |
| 07 | GPU Ray Tracer | 09 | Expert |
| 08 | Neural Network from Scratch | 10 | Expert |
| 09 | Physics Simulation Engine | 10 | Expert |
| 10 | GPU Database Engine | 11 | Expert |

---

## Progress Tracker

Copy this into your own notes and check boxes as you complete each item:

```markdown
## My CUDA Progress

### Phase 1 — Foundations
- [ ] Module 01: GPU Fundamentals
- [ ] Module 02: C++ Basics for CUDA
- [ ] Module 03: C++ Pointers & Memory
- [ ] Module 04: CUDA Setup & Toolchain

### Phase 2 — Core Programming
- [ ] Module 05: Threads, Blocks & Grids
- [ ] Module 06: Memory Model
- [ ] Module 07: Kernel Optimization

### Phase 3 — Intermediate
- [ ] Module 08: Streams & Concurrency
- [ ] Module 09: CUDA Libraries

### Phase 4 — Advanced
- [ ] Module 10: Multi-GPU Programming
- [ ] Module 11: Advanced CUDA

### Projects
- [ ] Project 01: Vector Addition
- [ ] Project 02: Matrix Addition
- [ ] Project 03: Matrix Multiplication
- [ ] Project 04: Image Processing Pipeline
- [ ] Project 05: 2D Convolution
- [ ] Project 06: Particle Simulation
- [ ] Project 07: GPU Ray Tracer
- [ ] Project 08: Neural Network from Scratch
- [ ] Project 09: Physics Simulation Engine
- [ ] Project 10: GPU Database Engine
```

---

## Repository Structure

```
cuda-programming-tutorial/
│
├── lectures/               Main learning content (start here)
│   ├── module-01-gpu-fundamentals/
│   ├── module-02-cpp-basics/
│   ├── module-03-pointers-and-memory/
│   ├── module-04-cuda-setup/
│   ├── module-05-threads-blocks-grids/
│   ├── module-06-memory-model/
│   └── module-07-kernel-optimization/
│
├── exercises/              Practice problems (Easy / Medium / Hard)
├── projects/               Complete buildable end-to-end projects
├── quizzes/                Self-assessment quizzes per module
├── benchmarks/             CPU vs GPU performance comparisons
├── cheatsheets/            Quick-reference cards
├── resources/              Curated books, videos, papers
├── diagrams/               GPU architecture diagrams
├── examples/               Minimal runnable code snippets
└── docs/                   Extended documentation and glossary
```

---

## Quick Start

### 1. Verify Your Hardware

```bash
# Linux
lspci | grep -i nvidia

# Windows — Open Device Manager → Display Adapters
# Look for an NVIDIA card (GeForce, RTX, Quadro, or Tesla)
```

### 2. Install CUDA Toolkit

Visit [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and select your OS.

### 3. Verify Installation

```bash
nvidia-smi          # Shows GPU info + max supported CUDA version
nvcc --version      # Shows CUDA compiler version
```

### 4. Compile Your First Program

```cpp
// hello.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU Thread %d!\n", threadIdx.x);
}

int main() {
    hello<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```bash
nvcc hello.cu -o hello && ./hello
```

---

## Key External Resources

| Resource | Link |
| :--- | :--- |
| NVIDIA CUDA Toolkit | [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) |
| CUDA C++ Programming Guide | [docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) |
| CUDA Best Practices Guide | [docs.nvidia.com/cuda/cuda-c-best-practices-guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) |
| GPU Deep Learning Performance | [docs.nvidia.com/deeplearning/performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) |
| C++ Reference | [en.cppreference.com](https://en.cppreference.com) |
| NVIDIA Nsight Profiler | [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems) |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Adding or improving lecture modules
- Submitting examples, exercises, and projects
- Code style, testing, and pull request process

---

## License

[MIT](LICENSE) — Free to use, share, and modify for personal and commercial projects.

---

<div align="center">

**Start with [Module 01 — GPU Fundamentals](lectures/module-01-gpu-fundamentals/README.md)**

</div>
