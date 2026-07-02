# Module 08 — CUDA-Libraries

> **Estimated time:** ~5 hours  
> **Prerequisites:** [module-8 Streams & Concurrency](/lectures/module-08-Streams%20&%20Concurrency/README.md)  
> **Next module:** comings soon
# CUDA Libraries 

Each library below follows the same structure:
- **Analogy** (real-life comparison)
- **Code Example** (minimal, actual API usage)
- **Step-by-step walkthrough** of the code
- **Real-world use case**

---

## 1. cuBLAS — Matrix & Vector Math

### What it does
Performs linear algebra: multiplying matrices, multiplying vectors, scaling numbers — the core math behind almost all AI models.

### Analogy
Imagine you have two giant spreadsheets of numbers and need to multiply them together, cell by cell, row by row. Doing this by hand (or even on a CPU) is slow. cuBLAS uses the GPU's thousands of tiny "workers" to do all the multiplications **at the same time**.

### Code Example (multiplying two matrices)
```c
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
    int N = 3;  // 3x3 matrices
    float A[9] = {1,2,3, 4,5,6, 7,8,9};
    float B[9] = {9,8,7, 6,5,4, 3,2,1};
    float C[9] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    // C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
```

### Step-by-step
1. Create your matrices normally in CPU memory (`A`, `B`, `C`).
2. Copy them to the GPU memory (`cudaMalloc` + `cudaMemcpy`).
3. Open a "cuBLAS session" (`cublasCreate`).
4. Call `cublasSgemm` — this one function does the entire matrix multiplication on the GPU.
5. Copy the result back to CPU memory to use it.

### Real-world use
Every layer in a neural network (like ChatGPT or image classifiers) multiplies matrices of numbers. cuBLAS is the engine under the hood of PyTorch/TensorFlow doing this.

---

## 2. cuSOLVER — Solving Systems of Equations

### What it does
Solves equations like `Ax = b` (find x), and breaks matrices into simpler building blocks (called factorization).

### Analogy
If cuBLAS is "multiply these numbers," cuSOLVER is "figure out what unknown values make this equation true" — like solving a big algebra puzzle, but with thousands of variables at once.

### Code Example (solving Ax = b using LU decomposition)
```c
#include <cusolverDn.h>
#include <cuda_runtime.h>

int main() {
    int n = 3;
    // Matrix A (column-major) and vector b
    float A[9] = {4,2,1, 3,5,2, 2,1,3};
    float B[3] = {10, 8, 7};

    float *d_A, *d_B;
    int *d_Ipiv, *d_info;
    cudaMalloc(&d_A, n*n*sizeof(float));
    cudaMalloc(&d_B, n*sizeof(float));
    cudaMalloc(&d_Ipiv, n*sizeof(int));
    cudaMalloc(&d_info, sizeof(int));

    cudaMemcpy(d_A, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n*sizeof(float), cudaMemcpyHostToDevice);

    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    int lwork;
    cusolverDnSgetrf_bufferSize(handle, n, n, d_A, n, &lwork);
    float *d_work; cudaMalloc(&d_work, lwork*sizeof(float));

    cusolverDnSgetrf(handle, n, n, d_A, n, d_work, d_Ipiv, d_info);   // Factorize A
    cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, d_A, n, d_Ipiv, d_B, n, d_info); // Solve for x

    float x[3];
    cudaMemcpy(x, d_B, n*sizeof(float), cudaMemcpyDeviceToHost);
    // x now holds the solution

    cusolverDnDestroy(handle);
    return 0;
}
```

### Step-by-step
1. Set up matrix `A` and vector `b` (from equation `Ax = b`).
2. `cusolverDnSgetrf` breaks `A` into simpler pieces (LU decomposition) — like factoring a number.
3. `cusolverDnSgetrs` uses those pieces to solve for `x`.
4. Copy `x` back — that's your answer.

### Real-world use
Structural engineers simulating stress on a bridge, or physics simulations, need to solve thousands of equations simultaneously. cuSOLVER does this fast.

---

## 3. cuSPARSE — Math on "Mostly Zero" Matrices

### What it does
Many real matrices are mostly zeros (called "sparse"). Instead of wasting time computing with all those zeros, cuSPARSE only stores and processes the non-zero values.

### Analogy
Imagine a 10,000-seat stadium, but only 50 seats are occupied. Instead of checking all 10,000 seats one by one, you just keep a short list: "Seat 42 - occupied, Seat 1005 - occupied…" That's how sparse matrices are stored and processed.

### Code Example (sparse matrix-vector multiplication)
```c
#include <cusparse.h>
#include <cuda_runtime.h>

int main() {
    // Sparse matrix in COO format: 3 non-zero values
    int rows[3]    = {0, 1, 2};
    int cols[3]    = {0, 1, 2};
    float vals[3]  = {5.0f, 3.0f, 2.0f};
    float x[3]     = {1.0f, 1.0f, 1.0f};
    float y[3]     = {0.0f, 0.0f, 0.0f};

    int *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;
    cudaMalloc(&d_rows, 3*sizeof(int));
    cudaMalloc(&d_cols, 3*sizeof(int));
    cudaMalloc(&d_vals, 3*sizeof(float));
    cudaMalloc(&d_x, 3*sizeof(float));
    cudaMalloc(&d_y, 3*sizeof(float));

    cudaMemcpy(d_rows, rows, 3*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols, 3*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, vals, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, 3*sizeof(float), cudaMemcpyHostToDevice);

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    // (Real code builds a cusparseSpMatDescr_t + cusparseDnVecDescr_t
    //  and calls cusparseSpMV — simplified here for clarity)

    cusparseDestroy(handle);
    return 0;
}
```

### Step-by-step
1. Instead of storing a full grid of numbers, you store only 3 things: which row, which column, and the value — for every non-zero entry.
2. cuSPARSE reads this compact list and performs multiplication only where there's an actual number.
3. Zeros are skipped entirely, saving huge amounts of time and memory.

### Real-world use
Recommendation systems (Netflix/Amazon) — most users haven't rated most products, so the "ratings matrix" is mostly zero. cuSPARSE handles this efficiently.

---

## 4. cuFFT — Frequency Analysis (Fourier Transform)

### What it does
Converts a signal (like sound or an image) from "time/space" view into "frequency" view — showing what patterns/frequencies make it up.

### Analogy
Think of a music equalizer. A song is just a wave of sound over time — but the equalizer shows you which frequencies (bass, mid, treble) are present. cuFFT does this math instantly, even for huge signals.

### Code Example (1D FFT on GPU)
```c
#include <cufft.h>

int main() {
    int N = 8;  // signal length
    cufftComplex data[8]; // input signal (real + imaginary parts)
    for (int i = 0; i < N; i++) { data[i].x = i; data[i].y = 0; }

    cufftComplex *d_data;
    cudaMalloc(&d_data, sizeof(cufftComplex)*N);
    cudaMemcpy(d_data, data, sizeof(cufftComplex)*N, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);   // create an FFT "plan"
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD); // run the FFT

    cudaMemcpy(data, d_data, sizeof(cufftComplex)*N, cudaMemcpyDeviceToHost);
    // data[] now contains frequency information

    cufftDestroy(plan);
    cudaFree(d_data);
    return 0;
}
```

### Step-by-step
1. Put your signal (numbers over time) into an array.
2. Copy it to GPU memory.
3. Create an FFT "plan" — this tells the GPU the size/type of transform to do.
4. Run `cufftExecC2C` — this converts your time-based signal into frequency-based data.
5. Copy back — now you know which frequencies are strongest in your signal.

### Real-world use
Noise-cancelling headphones detect unwanted frequencies and cancel them out. Wireless signal processing, MRI image reconstruction, and audio apps all use FFT constantly.

---

## 5. cuRAND — Fast Random Number Generation

### What it does
Generates huge amounts of random numbers extremely fast, directly on the GPU.

### Analogy
Imagine needing to flip a coin a billion times for an experiment. Instead of flipping one at a time, cuRAND "flips" millions of virtual coins simultaneously using GPU cores.

### Code Example (generate random numbers)
```c
#include <curand.h>

int main() {
    int n = 1000;
    float *d_randomNumbers;
    cudaMalloc(&d_randomNumbers, n*sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    curandGenerateUniform(gen, d_randomNumbers, n); // fill array with random floats (0 to 1)

    float result[1000];
    cudaMemcpy(result, d_randomNumbers, n*sizeof(float), cudaMemcpyDeviceToHost);

    curandDestroyGenerator(gen);
    cudaFree(d_randomNumbers);
    return 0;
}
```

### Step-by-step
1. Reserve GPU memory for however many random numbers you need.
2. Create a random number "generator" and give it a seed (so results are reproducible if needed).
3. Call `curandGenerateUniform` — it fills the entire array with random numbers in one fast GPU call.
4. Copy results back to use them.

### Real-world use
Banks run "Monte Carlo simulations" — testing thousands of random future scenarios to estimate financial risk. AI models also use random numbers to initialize their starting weights.

---

## 6. nvJPEG — GPU-Accelerated JPEG Decode/Encode

### What it does
Opens (decodes) and saves (encodes) JPEG images using the GPU instead of the CPU, so you can process thousands of images per second.

### Analogy
Normally, unzipping/opening an image is like unpacking one box at a time. nvJPEG lets you unpack a warehouse full of boxes (images) all at once.

### Code Example (decode a JPEG on GPU)
```c
#include <nvjpeg.h>

int main() {
    nvjpegHandle_t handle;
    nvjpegCreateSimple(&handle);

    nvjpegJpegState_t state;
    nvjpegJpegStateCreate(handle, &state);

    // Assume jpeg_data + jpeg_size are loaded from a .jpg file
    // int width, height; nvjpegGetImageInfo(...) gets dimensions first

    nvjpegImage_t output; // holds decoded RGB pixel buffers
    // cudaMalloc buffers for output.channel[0..2] here...

    nvjpegDecode(handle, state, /*jpeg_data*/nullptr, /*jpeg_size*/0,
                 NVJPEG_OUTPUT_RGB, &output, 0);
    // 'output' now contains the decoded image directly on the GPU

    nvjpegJpegStateDestroy(state);
    nvjpegDestroy(handle);
    return 0;
}
```

### Step-by-step
1. Load the raw JPEG file bytes (still compressed).
2. Give them to `nvjpegDecode` — the GPU decompresses the image directly into pixel data.
3. The decoded image stays on the GPU — ready for further GPU processing (like feeding into an AI model) without wasting time moving it back to the CPU.

### Real-world use
Training an image-recognition AI model needs millions of images loaded quickly. nvJPEG decodes them in bulk so the GPU never sits idle waiting for images.

---

## 7. NPP — Image & Signal Processing Toolbox

### What it does
A giant library of ready-made functions: resize, rotate, blur, sharpen, change colors, detect edges — for images and signals.

### Analogy
Like Photoshop's filters, but each filter runs on the GPU so it processes images instantly, even in bulk or in real-time video.

### Code Example (resizing an image)
```c
#include <npp.h>

int main() {
    // Assume d_src is a device pointer to an existing image already on the GPU
    Npp8u *d_src, *d_dst;
    // cudaMalloc + fill d_src with image pixel data beforehand...

    NppiSize srcSize = {640, 480};
    NppiRect srcRoi  = {0, 0, 640, 480};
    NppiSize dstSize = {320, 240};   // shrink to half size
    NppiRect dstRoi  = {0, 0, 320, 240};

    cudaMalloc(&d_dst, 320*240*3); // 3 channels (RGB)

    nppiResize_8u_C3R(d_src, 640*3, srcSize, srcRoi,
                       d_dst, 320*3, dstSize, dstRoi,
                       NPPI_INTER_LINEAR);
    // d_dst now holds the resized image
    return 0;
}
```

### Step-by-step
1. Have your image already loaded into GPU memory (`d_src`).
2. Decide your target size (here, shrinking to 320x240).
3. Call `nppiResize_8u_C3R` — one function call resizes the whole image on the GPU.
4. Use `d_dst` — your resized image, ready for display or further processing.

### Real-world use
Real-time video conferencing apps that blur backgrounds or resize video feeds on the fly use NPP-style GPU image processing.

---

## 8. cuTENSOR — Math on Multi-Dimensional Data (Tensors)

### What it does
While cuBLAS handles flat 2D matrices, cuTENSOR handles **tensors** — data with 3, 4, or more dimensions.

### Analogy
A matrix is like a checkerboard (rows x columns). A tensor is like a stack of checkerboards (or a Rubik's Cube) — data with extra dimensions like depth, time, or color channels. cuTENSOR does math across all these dimensions efficiently.

### Code Example (simplified tensor contraction concept)
```c
#include <cutensor.h>

int main() {
    cutensorHandle_t handle;
    cutensorCreate(&handle);

    // Define tensor A with shape [I, J, K] and tensor B with shape [J, K, L]
    // Contracting over J and K produces result C with shape [I, L]
    // (Full setup requires cutensorCreateTensorDescriptor for each tensor,
    //  then cutensorContract to run the operation on the GPU)

    cutensorDestroy(handle);
    return 0;
}
```

### Step-by-step (conceptually)
1. Describe the "shape" of your multi-dimensional data (e.g., a video: width × height × color × time).
2. Tell cuTENSOR which dimensions to combine/contract (similar to matrix multiplication, but in more dimensions).
3. cuTENSOR runs the operation across the GPU, handling all dimensions at once.

### Real-world use
Deep learning models that process video (4D: width, height, color, time) or scientific simulations (quantum chemistry) rely on cuTENSOR for these multi-dimensional calculations.

---

## 9. cuSPARSELt — Optimized Sparse Matrix Multiply for AI

### What it does
A specialized version of cuSPARSE built for "structured sparsity" — a pattern where exactly half of a matrix's values are zero in a predictable, repeating way. Newer NVIDIA GPUs have special hardware to exploit this pattern for speed.

### Analogy
Imagine a checkerboard where you already know every red square is empty. You don't need to check each square — you just skip straight to the black squares. That predictability is what cuSPARSELt exploits.

### Code Example (conceptual usage flow)
```c
#include <cusparseLt.h>

int main() {
    cusparseLtHandle_t handle;
    cusparseLtInit(&handle);

    // 1. Describe a "sparse" matrix (structured, 50% zeros) and a "dense" matrix
    // 2. Compress the sparse matrix (cusparseLtSpMMACompress)
    // 3. Run the multiplication (cusparseLtMatmul) — faster than a normal dense multiply

    cusparseLtDestroy(&handle);
    return 0;
}
```

### Step-by-step (conceptually)
1. Take an AI model's weight matrix that has been "pruned" (half its values zeroed out in a fixed pattern).
2. Compress it — cuSPARSELt stores only the necessary half.
3. Multiply it against your data — this skips half the normal work, roughly doubling speed.

### Real-world use
Making large AI models (like image classifiers or language models) run faster during inference (prediction time) by skipping unnecessary computation.

---

## 10. nvJPEG2000 — Decoding JPEG2000 Images

### What it does
Like nvJPEG, but for the JPEG2000 format — used where extremely high image quality/detail matters (not your everyday photo format).

### Analogy
Regular JPEG is like a compressed MP3 of an image — good enough, some quality lost. JPEG2000 is like a lossless, studio-quality version — used when every pixel detail matters (medical scans, satellite images).

### Code Example (conceptual decode)
```c
#include <nvjpeg2k.h>

int main() {
    nvjpeg2kHandle_t handle;
    nvjpeg2kCreateSimple(&handle);

    nvjpeg2kStream_t stream;
    nvjpeg2kStreamCreate(&stream);
    // nvjpeg2kStreamParse(handle, jpeg2000_data, data_size, 0, 0, stream);

    nvjpeg2kImage_t output;
    // Set up output buffers (per color channel) on GPU beforehand

    nvjpeg2kDecode(handle, /*decode_state*/nullptr, stream, &output, 0);
    // output now holds the decoded high-quality image on the GPU

    nvjpeg2kStreamDestroy(stream);
    nvjpeg2kDestroy(handle);
    return 0;
}
```

### Step-by-step
1. Load the raw JPEG2000 file bytes.
2. Parse the stream to understand its structure (`nvjpeg2kStreamParse`).
3. Decode it directly to GPU memory (`nvjpeg2kDecode`).
4. Use the high-quality decoded image for analysis (e.g., a doctor viewing a scan, or a satellite image being analyzed).

### Real-world use
Hospitals viewing/storing X-rays and MRIs, or satellite companies processing Earth imagery — both need lossless detail that JPEG2000 preserves.

---

## Putting It Together — A Simple Mental Model

| If you need to... | Use this library |
|---|---|
| Multiply matrices/vectors | **cuBLAS** |
| Solve equations (Ax = b) | **cuSOLVER** |
| Work with mostly-zero matrices | **cuSPARSE** |
| Analyze frequencies in a signal | **cuFFT** |
| Generate random numbers | **cuRAND** |
| Load/save JPEG images fast | **nvJPEG** |
| Resize/filter/edit images | **NPP** |
| Do math on 3D+ data (video, etc.) | **cuTENSOR** |
| Speed up sparse AI models | **cuSPARSELt** |
| Load high-quality JPEG2000 images | **nvJPEG2000** |

## A Simple Way to Remember the Pattern

Almost every CUDA library follows the same basic recipe:
1. **Allocate** GPU memory (`cudaMalloc`)
2. **Copy** your data from CPU → GPU (`cudaMemcpy`)
3. **Create a handle** (like opening a "session" with the library)
4. **Call the library function** to do the heavy math
5. **Copy the result** back from GPU → CPU
6. **Clean up** (destroy handles, free memory)

Once you recognize this pattern, every CUDA library — no matter which one — becomes much easier to read and use.