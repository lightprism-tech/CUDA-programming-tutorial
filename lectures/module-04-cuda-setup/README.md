# Module 04 — CUDA Setup & Verification

> **Estimated time:** ~45 minutes  
> **Prerequisites:** [Module 03 — C++ Pointers and Memory](../module-03-pointers-and-memory/README.md)  
> **Next module:** [Module 05 — Threads, Blocks & Grids](../module-05-threads-blocks-grids/README.md)

---

## 1. Quick Hardware Verification
Ensure you have an NVIDIA GPU:
*   **Ubuntu / WSL:** Run `lspci | grep -i nvidia` or `nvidia-smi`
*   **Windows:** Check **Device Manager** $\rightarrow$ **Display adapters**

---

## 2. NVIDIA Driver & CUDA Compatibility

Your installed NVIDIA driver determines which CUDA Toolkit version you can run.

> [!IMPORTANT]
> **Check Compatibility First:** Run `nvidia-smi` to see your current driver. Always download a CUDA Toolkit version that is compatible with your driver. If you try to run a newer CUDA version on an older driver, you will get: `the provided PTX was compiled with an unsupported toolchain`.

---

## 3. Ubuntu Installation (Recommended / Native Linux)

1. **Install NVIDIA Drivers:**
   ```bash
   sudo apt update
   ubuntu-drivers devices # Identify recommended driver
   sudo apt install nvidia-driver-<recommended-version> -y
   sudo reboot
   ```
2. **Download & Install CUDA Toolkit:**
   * Go to the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
   * Select: **Linux** $\rightarrow$ **x86_64** $\rightarrow$ **Ubuntu** $\rightarrow$ Select your version $\rightarrow$ **runfile (local)**.
   * Run the commands provided by NVIDIA.
3. **Configure Environment Variables:**
   Add these to your `~/.bashrc`:
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

---

## 4. WSL Installation (Windows + Ubuntu)

If you are running Windows but want to use Ubuntu via WSL2:

1. **Install WSL2 & Ubuntu:**
   In Windows PowerShell (as Admin):
   ```powershell
   wsl --install
   ```
2. **Install Windows Host Driver:**
   * Install the latest NVIDIA driver on Windows. **Do not install a GPU driver inside WSL.**
3. **Install CUDA inside WSL Ubuntu:**
   * Open your WSL Ubuntu terminal.
   * Go to [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
   * Select: **Linux** $\rightarrow$ **x86_64** $\rightarrow$ **WSL-Ubuntu** $\rightarrow$ **runfile (local)**.
   * Follow the commands to install.
4. **Configure Environment Variables:**
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```
5. **Access Windows Files:**
   Your Windows drives are located under `/mnt/` (e.g. `C:\Projects` is at `/mnt/c/Projects`).

---

## 5. Windows Installation

1. **Install Host Driver:** Ensure the latest GeForce/RTX driver is installed on Windows.
2. **Download & Install CUDA Toolkit:**
   * Go to the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
   * Select: **Windows** $\rightarrow$ **x86_64** $\rightarrow$ Choose version $\rightarrow$ **exe (local)**.
   * Run the installer and choose **Express Installation**. Restart your system.

---

## 6. Verify the Installation

Confirm both the driver and compiler are working:
*   **Driver & GPU Check:**
    ```bash
    nvidia-smi
    ```
*   **CUDA Compiler Check:**
    ```bash
    nvcc --version
    ```
    *(If `nvcc` is not found, verify your environment paths from step 3 or 4).*

---

## 7. Compile Your First CUDA Program

Create a file named `hello.cu`:

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU Thread %d\n", threadIdx.x);
}

int main() {
    // Launch 1 block with 5 threads
    helloFromGPU<<<1, 5>>>();
    cudaDeviceSynchronize(); // Wait for GPU to finish printing
    return 0;
}
```

### Understanding <<<Blocks, Threads>>>
*   `helloFromGPU<<<1, 5>>>` launches **1 Block** containing **5 Threads** (Total 5 parallel threads).
*   `threadIdx.x` is the thread's index within the block (0, 1, 2, 3, 4).

### Target Your GPU Architecture
Determine your GPU's Compute Capability (e.g. `8.6` for RTX 3060, `8.9` for RTX 4060). Compile with the `-arch` flag:
```bash
nvcc -arch=sm_xx hello.cu -o hello
```
*Replace `xx` with your GPU capability (e.g., `sm_86`, `sm_89`).*

Run the executable:
*   **Linux / WSL:** `./hello`
*   **Windows:** `.\hello.exe`

---

## 8. Troubleshooting Guide

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `nvcc: command not found` | PATH variable is not set | Export `/usr/local/cuda/bin` to your path in `~/.bashrc` (See step 3). |
| `the provided PTX was compiled with an unsupported toolchain` | CUDA Toolkit is newer than your NVIDIA Driver | Update your GPU driver, or install an older compatible CUDA Toolkit, or compile with `-arch=sm_xx`. |
| `invalid configuration argument` | Too many threads per block | Ensure threads per block is $\le$ 1024. |
| GPU shows 0% usage | Kernel runs too fast | Use a longer computation loop or profile with Nsight. |
| `No such file or directory` | Executing from the wrong folder | Use `pwd` and `ls` to find your file, then `cd` into that directory. |

---

## 9. Verification Checklist
Before moving to Module 5:
*   [ ] `nvidia-smi` works and lists your GPU.
*   [ ] `nvcc --version` works.
*   [ ] `hello.cu` compiles without errors.
*   [ ] Running `./hello` prints output from threads 0 to 4.

---

**Next:** [Module 05 — Threads, Blocks & Grids](../module-05-threads-blocks-grids/README.md)
