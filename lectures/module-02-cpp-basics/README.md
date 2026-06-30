# Module 02 — C++ Basics for CUDA

> **Estimated time:** ~2 hours  
> **Prerequisites:** None  
> **Next module:** [Module 03 — Pointers and Memory](../module-03-pointers-and-memory/README.md)

---

## The Main Idea

**You don't need to be a C++ expert to learn CUDA.** You just need a solid grasp of basic C++ syntax and concepts. Think of it like learning to drive: you don't need to know how to rebuild an engine to get behind the wheel!

---

## Part 1: C++ Basics You Need to Learn

### 1. Variables and Data Types
Variables act like labeled storage boxes containing different types of data:

```cpp
int age = 20;           // A whole number (integer)
float price = 10.5f;     // A decimal number (single-precision floating point)
char letter = 'A';      // A single character
bool isStudent = true;  // A boolean (true/false)
```

**Real-World Analogy:**
*   `int` = A crate for counting apples (whole units only).
*   `float` = A measuring cup for liquids (allows fractional parts).
*   `char` = A letterbox containing a single physical alphabet block.
*   `bool` = A light switch (either ON or OFF).

**Why it matters for CUDA:** GPU programs process large sets of numbers (matrix weights, coordinates, etc.). Knowing whether you are using single-precision (`float`) or double-precision (`double`) is critical because GPUs are optimized heavily for fast `float` (FP32) arithmetic.

---

### 2. Conditions (Decision Making)
Conditions allow your code to take different paths based on statements:

```cpp
if (age >= 18) {
    std::cout << "You can vote";
} else {
    std::cout << "Too young to vote";
}
```

**Real-World Analogy:** "If it is raining, open the umbrella. Otherwise, leave it closed."

**Why it matters for CUDA:** When threads in a GPU process code, conditional statements can cause threads inside the same warp to diverge (taking different paths), which can significantly slow down execution.

---

### 3. Loops (Repetition)
Loops allow you to execute a block of code multiple times:

```cpp
for (int i = 0; i < 10; i++) {
    std::cout << i << " ";  // Output: 0 1 2 3 4 5 6 7 8 9
}
```

**Real-World Analogy:** "Paint 10 fence panels. For each panel, apply paint, then move to the next."

**Why it matters for CUDA:** On a CPU, loops are used to process arrays sequentially (element by element). In CUDA, we often replace loops with parallel execution, launching millions of threads where each thread does the work of a single loop iteration.

---

### 4. Functions (Reusable Code blocks)
A function is a packaged recipe that accepts inputs, executes logic, and optionally returns an output:

```cpp
int add(int a, int b) {
    return a + b;
}

// How to use it:
int result = add(5, 3);  // result = 8
```

**Real-World Analogy:** A food processor. You put ingredients in, it processes them according to its setup, and it outputs the result.

**Why it matters for CUDA:** CUDA GPU operations are written inside special functions called **kernels** (using the `__global__` specifier) which are launched from the CPU to run in parallel on the GPU.

---

### 5. Arrays (Lists of Data)
An array is a contiguous block of memory storing elements of the same type:

```cpp
int scores[5] = {10, 20, 30, 40, 50};
std::cout << scores[0];  // Output: 10 (First element)
std::cout << scores[2];  // Output: 30 (Third element)
```

**Real-World Analogy:** A post office mailbox rack with sequentially numbered slots starting at index 0.

**Why it matters for CUDA:** Deep Learning models represent parameters (like weights and activations) as large arrays (vectors and matrices). In CUDA, you will manage, transfer, and compute over these large arrays continuously.

---

### 6. Pointers ⭐⭐⭐ (The Most Important Concept)
A pointer is a variable that stores the **memory address** of another variable instead of storing a value directly.

```cpp
int age = 25;        // A box containing the value 25
int* ptr = &age;     // A pointer pointing TO the memory address of the age box

std::cout << age;         // Output: 25 (the value)
std::cout << &age;        // Output: 0x7fff5fbff8ac (the physical memory location)
std::cout << *ptr;        // Output: 25 (dereferencing: looking at what ptr points to)
```

**Real-World Analogy:**
*   `age` = The actual treasure inside a chest.
*   `&age` = The GPS coordinates on a map indicating where the chest is buried.
*   `*ptr` = Following the GPS coordinates on the map to open the chest and access the treasure.

**Why it matters for CUDA:** Pointers are absolutely **critical** in CUDA. Because the CPU (Host) and GPU (Device) have physical separate memories, pointers are used to allocate GPU memory (`cudaMalloc`) and copy data back and forth (`cudaMemcpy`).

---

### 7. Dynamic Memory
Dynamic memory allocation lets you request memory from the operating system on-the-fly during runtime:

```cpp
int* arr = new int[100];  // Allocate space for 100 integers on the heap
// Use the array...
delete[] arr;             // Release the memory back to the OS
```

**Real-World Analogy:**
*   `new` = Renting a set of hotel rooms for a group of guests.
*   `delete[]` = Checking out of the hotel so the rooms can be rented to someone else.

**Why it matters for CUDA:** Deep Learning models dynamically load variable batch sizes and image resolutions at runtime. CUDA uses its own dynamic allocation APIs (`cudaMalloc` and `cudaFree`) on the GPU.

---

### 8. Classes (Optional for CUDA Beginners)
Classes bundle data fields and functions together into objects:

```cpp
class Car {
public:
    std::string color;
    std::string brand;
};

Car myCar;
myCar.color = "red";
```

> [!TIP]
> **Learning Tip:** You can safely skip classes, structures, and object-oriented programming when writing your first CUDA kernels. Keep your initial GPU code simple and procedural.

---

## Part 2: What You CAN Skip Initially

Do not spend time on these advanced C++ features right now:
*   ❌ **Templates** (Compile-time polymorphism)
*   ❌ **Advanced Standard Library Containers** (`std::map`, `std::unordered_map` - these do not easily port to raw GPU code)
*   ❌ **C++ Standard Multithreading** (`std::thread`, `std::mutex` - GPU concurrency uses warps, blocks, and CUDA streams instead)
*   ❌ **Complex Software Design Patterns**

---

## Part 3: Transitioning to CUDA

Once you understand basic C++, CUDA introduces specialized extensions to help you write GPU code:

*   **CUDA Kernel:** A function written in C++ that is executed by many GPU threads in parallel. It is marked with the `__global__` keyword.
*   **Thread Indexing:** GPU threads use built-in variables like `threadIdx.x` to find out which element of an array they are supposed to process.

```cpp
__global__ void add(int* a, int* b, int* c) {
    int i = threadIdx.x;  // My unique thread ID
    c[i] = a[i] + b[i];   // Perform math in parallel
}
```

---

## Part 4: Deep Learning & C++ References

When designing performance-critical software like deep learning inference engines, a solid understanding of C++ basics is required. For more details on standard C++ fundamentals, refer to the following official resources:

*   **Variables, Types and Array Layouts:** Learn about how memory layout affects performance at [cppreference - Fundamental Types](https://en.cppreference.com/w/cpp/language/types).
*   **Contiguous Sequences:** Understand `std::vector` and dynamic arrays for storing tensor buffers at [cppreference - Vector Container](https://en.cppreference.com/w/cpp/container/vector).
*   **Pointer Semantics:** Read about memory layouts and pointers at [cppreference - Pointer Types](https://en.cppreference.com/w/cpp/language/pointer).

---

➡️ **Next:** [Module 03 — Pointers and Memory](../module-03-pointers-and-memory/README.md)
