# Module 03 — C++ Pointers and Memory

> **Estimated time:** ~3 hours  
> **Prerequisites:** [Module 02 — C++ Basics](../module-02-cpp-basics/README.md)  
> **Next module:** [Module 04 — CUDA Setup](../module-04-cuda-setup/README.md)

---

## What is Memory?

When a C++ program runs, the operating system allocates a block of memory (RAM) for it. This memory is divided into different regions:

*   **Stack:** Used for automatic storage (local variables). It is fast, but limited in size.
*   **Heap:** Used for dynamic storage (allocated at runtime using `new`). It is large, but requires manual cleanup.
*   **Global/Static:** Used for variables that live for the entire duration of the program.

Every variable you declare is stored at a specific physical location in memory, known as its **memory address**. A pointer is simply a variable that stores one of these addresses.

---

## What is a Pointer?

A pointer stores the **memory address** of another variable. Instead of holding the actual data, it points to *where* the data is located.

### Two Key Operators:
- `&` (**Address-of** operator): Retrieves the memory address of a variable.
- `*` (**Dereference** operator): Accesses or modifies the value stored at the address the pointer points to.

```cpp
int age = 25;
int* ptr = &age;   // ptr holds the memory address of age

std::cout << ptr;       // Output: 0x1A4F (an example memory address)
std::cout << *ptr;      // Output: 25 (the value inside age)

*ptr = 30;         // Dereferencing to change the value at the address
std::cout << age;       // Output: 30 (age has been updated!)
```

---

## Declaring Pointers

Pointers must be declared with a type matching the variable they point to:

```cpp
int*    p;    // Pointer to an integer
double* d;    // Pointer to a double-precision float
char*   c;    // Pointer to a character
int**   pp;   // Pointer to a pointer (nested pointers)

// Always initialize your pointers. An uninitialized pointer points to a random memory location and will crash your program.
int* pSafe = nullptr;   // Safe, null initialized pointer
```

---

## Pointer Arithmetic

Pointers are aware of the **byte size** of the data type they reference. Adding 1 to a pointer increments its address by the size of the type, not by 1 byte.

```cpp
int arr[] = {10, 20, 30, 40};
int* ptr = arr;   // Points to arr[0] (address of first element)

std::cout << *ptr;       // Output: 10
std::cout << *(ptr + 1); // Output: 20 (moves 4 bytes forward because sizeof(int) = 4)
std::cout << *(ptr + 2); // Output: 30

ptr++;              // Increments address to point to the next integer (arr[1])
std::cout << *ptr;       // Output: 20
```

---

## Pointers and Arrays

In C++, **the name of an array acts as a pointer** to its first element:

```cpp
int arr[] = {1, 2, 3};

// Subscript notation and pointer notation are equivalent:
std::cout << arr[0];    // Output: 1
std::cout << *arr;      // Output: 1

std::cout << arr[2];    // Output: 3
std::cout << *(arr + 2);// Output: 3

// Iterating through an array using a pointer:
for (int* p = arr; p < arr + 3; p++) {
    std::cout << *p << " ";   // Output: 1 2 3
}
```

---

## Dynamic Memory Allocation (Heap)

Stack memory is managed automatically by the compiler but has size limitations. When you need memory that:
1. Must persist beyond the scope of the function that created it, or
2. Has a size determined at runtime (e.g., loading a variable-sized dataset)

You allocate it on the **Heap** using `new`, and you must free it using `delete`:

```cpp
// 1. Single Object Allocation
int* ptr = new int(42);    // Allocate space for one integer on the heap and set it to 42
std::cout << *ptr;         // Output: 42
delete ptr;                // Free the memory to prevent memory leaks!
ptr = nullptr;             // Reset the pointer

// 2. Array Allocation
int* arr = new int[5];     // Allocate space for 5 integers on the heap
arr[0] = 10;
delete[] arr;              // Use delete[] (with brackets) for arrays!
arr = nullptr;
```

---

## References vs. Pointers

A **reference** is an alias (another name) for an existing variable. It acts similarly to a pointer but has a safer, cleaner syntax.

```cpp
int age = 25;

int* ptr = &age;   // Pointer: can be reassigned, can be nullptr, requires *
int& ref = age;    // Reference: cannot be null, cannot be reassigned, no * needed

*ptr = 30;         // Modify via pointer
ref = 35;          // Modify via reference (much simpler syntax!)
```

### Detailed Comparison:

| Feature | Pointer | Reference |
| :--- | :--- | :--- |
| **Can be null?** | ✅ Yes (`nullptr`) | ❌ No (must refer to a valid object) |
| **Can be reassigned?** | ✅ Yes | ❌ No (bound to the initial variable) |
| **Dereference syntax?** | ✅ Yes (requires `*` operator) | ❌ No (uses normal variable syntax) |
| **Use Cases** | Dynamic memory, arrays, optional values | Function parameters, operator overloading |

---

## Passing Parameters to Functions

How parameters are passed to functions changes whether the original arguments are modified:

```cpp
void byValue(int x)    { x = 99; } // Copies the value. Original is unchanged.
void byPointer(int* x) { *x = 99; } // Passes address. Original changes.
void byRef(int& x)     { x = 99; } // Passes alias. Original changes (clean syntax).

int a = 5;
byValue(a);      std::cout << a;   // Output: 5
byPointer(&a);   std::cout << a;   // Output: 99
byRef(a);        std::cout << a;   // Output: 99
```

---

## Smart Pointers (Modern C++)

Manual memory management with `new`/`delete` is error-prone and leads to memory leaks or program crashes. Modern C++ introduced **smart pointers** (defined in `<memory>`) to automate cleanup:

```cpp
#include <memory>

// 1. unique_ptr (Single Owner)
// Automatically deletes the allocated memory when the unique_ptr goes out of scope.
std::unique_ptr<int> p1 = std::make_unique<int>(42);
std::cout << *p1; // Output: 42
// No manual 'delete' needed!

// 2. shared_ptr (Multiple Owners)
// Keeps a reference count. Memory is freed only when the last shared_ptr is destroyed.
std::shared_ptr<int> p2 = std::make_shared<int>(100);
std::shared_ptr<int> p3 = p2; // Both point to the same memory
```

---

## Pointers to Pointers (`**`)

A pointer can point to another pointer. This is useful for passing pointers to functions to modify their address, or creating dynamic 2D arrays.

```cpp
int x = 5;
int* p = &x;    // p points to x
int** pp = &p;  // pp points to p

std::cout << **pp; // Output: 5 (follows the chain twice to read x)
**pp = 99;         // Changes the value of x to 99
```

---

## `const` with Pointers

The `const` keyword restricts modification. Its position relative to the asterisk (`*`) determines what is constant:

```cpp
int x = 10, y = 20;

const int* p1 = &x;     // Pointer to Constant Data: Can't change data (*p1 = 99 is ERROR), CAN change pointer (p1 = &y is OK)
int* const p2 = &x;     // Constant Pointer: Can change data (*p2 = 99 is OK), CAN'T change pointer (p2 = &y is ERROR)
const int* const p3 = &x;// Constant Pointer to Constant Data: Both are read-only!
```

> [!TIP]
> **Reading Trick:** Read the declaration from **right to left**:
> *   `int* const p` $\rightarrow$ "p is a `const` pointer to `int`"
> *   `const int* p` $\rightarrow$ "p is a pointer to `const int`"

---

## Deep Learning & C++ References

Efficient memory management is the foundation of high-performance deep learning libraries (like PyTorch and TensorRT). To dive deeper into how C++ handles memory layouts, pointers, and performance optimizations, refer to the official C++ references:

*   **Dynamic Memory Management:** Learn about dynamic allocations and custom memory allocators at [cppreference - Dynamic Memory](https://en.cppreference.com/w/cpp/memory).
*   **Smart Pointers:** Understand modern memory ownership models at [cppreference - Smart Pointers](https://en.cppreference.com/w/cpp/memory/unique_ptr).
*   **Standard Arrays and Layouts:** Read about raw memory configurations at [cppreference - Arrays](https://en.cppreference.com/w/cpp/language/array).

---

➡️ **Next:** [Module 04 — CUDA Setup](../module-04-cuda-setup/README.md)
