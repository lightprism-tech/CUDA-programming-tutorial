# Module 10 Multi-GPU Programming


> **Estimated time:** ~5 hours  
> **Prerequisites:** [module-9 CUDA-libraries](/lectures/module-09-CUDA-Libraries/README.md)  
> **Next module:**[comming soon ]



# Multi-GPU Programming: NVLink, NCCL, and Peer Access

## The big picture

Multi-GPU programming means getting several GPUs to work together as if they were one giant GPU. The hard part is almost never the math — GPUs are fast at math. The hard part is **moving data between GPUs fast enough that they aren't sitting idle waiting for each other**.

Three pieces solve this problem, each at a different layer of the stack:

| Layer | Piece | Question it answers |
|---|---|---|
| Hardware (the wire) | **NVLink** | What physical connection do GPUs use to talk to each other? |
| Software permission (the door) | **Peer Access (P2P)** | Can GPU A reach directly into GPU B's memory? |
| Orchestration (the manager) | **NCCL** | When many GPUs need to exchange data at once, who decides the most efficient way to do it? |

Think of a company with several offices (GPUs) that all need to collaborate:
- **NVLink** is the dedicated private road built between office buildings, instead of everyone using the crowded public highway (PCIe).
- **Peer access** is the security policy that lets an employee from Office A walk straight into Office B and grab a file, instead of mailing it through the central mailroom (the CPU).
- **NCCL** is the logistics manager who decides the smartest way for all offices to send updates to each other at once, instead of everyone individually emailing everyone else.

---

## 1. The core problem, in more detail

Say you're training a large neural network across 8 GPUs. A common setup (data parallelism) works like this:

1. Split your training batch into 8 chunks, one per GPU.
2. Each GPU runs the forward and backward pass on its chunk, producing its own set of **gradients** (the numbers that say "adjust the model weights this way").
3. Before anyone updates their weights, all 8 GPUs need to **agree on the average gradient** — otherwise each GPU would train a slightly different model and they'd drift apart.
4. Step 3 requires every GPU to send its gradients to every other GPU (or some equivalent), sum them up, and get the result back.

That exchange in step 3 can involve gigabytes of data, repeated thousands of times during training. If it's slow, adding more GPUs barely helps — you've traded "waiting for compute" for "waiting for communication." This is why the three pieces below exist: to make step 3 as close to free as possible.

---

## 2. Sub-topic: NVLink (the hardware highway)

### The default path: PCIe

Every GPU sits in a PCIe slot, which is how it normally talks to the CPU and the rest of the system. PCIe is general-purpose — it's shared by GPUs, storage, network cards, everything — and its bandwidth (tens of GB/s) is modest by GPU standards.

### The upgrade: NVLink

NVLink is NVIDIA's dedicated point-to-point connection built specifically for GPU-to-GPU traffic. It skips the general-purpose bus entirely.

- **PCIe Gen4/5** (shared, general purpose): roughly 32–64 GB/s total.
- **NVLink** (dedicated, GPU-only): several hundred GB/s per GPU, and newer generations combine multiple links to exceed a terabyte per second.

Each NVLink generation has roughly doubled bandwidth over the last (Pascal → Volta → Ampere → Hopper → Blackwell), though exact numbers depend on the specific GPU model and how many links it has, so treat any specific figure as approximate and check NVIDIA's current spec sheets if you need exact numbers for a purchasing or capacity decision.

### NVSwitch: connecting *everyone* to *everyone*

A single NVLink cable connects two GPUs directly. That's fine for 2 GPUs, but with 8 GPUs in a server, you'd need a tangled mesh of direct cables to connect every pair — and it still wouldn't scale well.

**NVSwitch** solves this: it's a switch chip inside the server that every GPU plugs into, and it gives *any* GPU a full-bandwidth path to *any* other GPU, not just its physical neighbor. This is what makes an 8-GPU box (like NVIDIA's DGX systems) behave like one cohesive unit instead of 8 cards that happen to share a case.

Newer systems (like the NVL72 rack-scale design) extend this switching fabric across many servers, blurring the line between "inside one box" and "across a rack."

### Simple mental model

```
Without NVSwitch (mesh of direct links):        With NVSwitch:

  GPU0 --- GPU1                                  GPU0   GPU1
   |  \   /  |                                     \     /
   |   \ /   |                                      NVSwitch
   |   / \   |                                      /     \
  GPU2 --- GPU3                                  GPU2   GPU3

  Gets messy fast as GPU count grows.             Every GPU has a full-speed
                                                   path to every other GPU.
```

---

## 3. Sub-topic: Peer Access / P2P (the door between GPUs)

Fast wires alone aren't enough — there's a software question underneath: **is GPU A even allowed to read GPU B's memory directly?**

### The slow default

Without peer access, moving data from GPU B to GPU A goes:

```
GPU B memory  →  CPU (system RAM)  →  GPU A memory
```

That's two extra copies and a trip through the CPU, wasting time and PCIe bandwidth that could be doing other work.

### The fast path: enabling P2P

CUDA lets you enable direct GPU-to-GPU memory access with a simple call:

```c
cudaSetDevice(0);
cudaDeviceEnablePeerAccess(1, 0);  // GPU 0 can now access GPU 1's memory directly

cudaSetDevice(1);
cudaDeviceEnablePeerAccess(0, 0);  // and vice versa
```

Once enabled, a direct copy looks like this:

```c
// Copy 1MB directly from GPU 1's memory into GPU 0's memory, no CPU involved
cudaMemcpyPeer(dst_ptr_on_gpu0, 0, src_ptr_on_gpu1, 1, size_in_bytes);
```

This only works at full speed if there's a real fast path between the two GPUs — usually NVLink (if present) or PCIe as a fallback. `cudaDeviceCanAccessPeer()` lets you check in code whether two specific GPUs actually support this before relying on it.

### Beyond one server: GPUDirect RDMA

The same idea extends across the network. **GPUDirect RDMA** lets a network card pull data straight out of GPU memory and ship it to another machine, again without bouncing through CPU memory first. This is what makes multi-*server* training fast, not just multi-GPU-in-one-box training.

### Analogy recap

- No peer access: GPU B mails a package to the CPU's front desk, which then forwards it to GPU A.
- Peer access: GPU B walks it directly next door to GPU A.
- GPUDirect RDMA: GPU B hands it directly to a courier (the network card) who takes it straight to another building (another server), no front desk involved.

---

## 4. Sub-topic: NCCL (the logistics manager)

Fast wires and direct access solve the "how do two GPUs move data" problem. But when **all 8 (or 800) GPUs** need to exchange data at once, someone has to decide the smartest pattern — because "everyone sends to everyone" wastes enormous bandwidth.

**NCCL** (NVIDIA Collective Communications Library, said "Nickel") is a library of pre-optimized routines for exactly this. It's the piece that sits on top of NVLink and peer access, and it's what libraries like PyTorch's `DistributedDataParallel`, DeepSpeed, Megatron-LM, and Horovod use under the hood — you rarely call it directly, but it's doing the real work.

### The core collective operations

Each of these has a simple real-world analogy:

| Operation | What it does | Analogy |
|---|---|---|
| **Broadcast** | One GPU sends the same data to all others | One person reads an announcement to the whole room |
| **Reduce** | Combine (e.g. sum) data from all GPUs, result lands on *one* GPU | Everyone hands their number to one person, who adds them all up |
| **AllReduce** | Combine data from all GPUs, *everyone* gets the result | Everyone hands in a number, the total is written on a whiteboard everyone can see |
| **Gather** | Collect data from all GPUs onto *one* GPU | Everyone's individual reports get filed in one folder held by one person |
| **AllGather** | Collect data from all GPUs, *everyone* gets the full collection | Everyone's individual reports get copied and handed to everyone |
| **ReduceScatter** | Combine data, then split the result so each GPU gets only its slice | The total is computed, then cut into pieces and each person gets their own piece |
| **Send/Recv** | Direct point-to-point transfer between two specific GPUs | A private message between two people |

**AllReduce is the one you'll hear about most** — it's exactly what's needed for the gradient-averaging step in the training example from Section 1: every GPU contributes its gradients, and every GPU walks away with the same averaged result.

### How NCCL actually moves the data efficiently

If GPU 0 just sent its data individually to GPUs 1 through 7, that's 7 separate slow transfers, and GPU 0's outgoing link becomes a bottleneck. Instead, NCCL arranges GPUs into a pattern — commonly a **ring** or a **tree** — so that data flows in stages and every GPU's link is used in parallel instead of funneling through one GPU.

**Ring AllReduce**, the classic approach:

```
GPU0 → GPU1 → GPU2 → GPU3 → GPU0   (data flows around the ring)
```

Each GPU only ever talks to its two ring neighbors. Data is broken into chunks, and in a series of steps every GPU passes a partial sum to the next GPU while receiving one from the previous GPU. After enough steps, every GPU has the full combined result — and critically, every GPU's link was busy the whole time, instead of one GPU being a bottleneck.

### Topology awareness

Before any of this happens, NCCL automatically inspects the hardware: which GPUs share NVLink, which share NVSwitch, which are only connected via PCIe, and which are on entirely separate machines connected by InfiniBand or Ethernet. It then picks a communication pattern (ring, tree, or a hybrid) that matches the actual physical topology, so data always prefers the fastest path available.

---

## 5. Simple worked example

### Example A: Two GPUs, direct peer copy (bare CUDA)

This is the simplest possible illustration of peer access — no NCCL yet, just two GPUs handing data to each other directly.

```c
#include <cuda_runtime.h>

int main() {
    float *data_gpu0, *data_gpu1;
    size_t size = 1024 * sizeof(float);

    // Allocate memory on each GPU
    cudaSetDevice(0);
    cudaMalloc(&data_gpu0, size);

    cudaSetDevice(1);
    cudaMalloc(&data_gpu1, size);

    // Check and enable peer access both ways
    int canAccess;
    cudaDeviceCanAccessPeer(&canAccess, 0, 1);
    if (canAccess) {
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
    }

    // Direct GPU-to-GPU copy, no CPU staging
    cudaMemcpyPeer(data_gpu0, 0, data_gpu1, 1, size);

    // ... use data_gpu0, now populated with GPU 1's data ...

    cudaFree(data_gpu0);
    cudaFree(data_gpu1);
    return 0;
}
```

What happened: GPU 1's data moved directly into GPU 0's memory. If NVLink connects these two GPUs, this transfer used it. If peer access hadn't been enabled, `cudaMemcpyPeer` would still work, but CUDA would silently route it through host memory instead — much slower.

### Example B: Four GPUs, AllReduce with NCCL (simplified)

This shows the real building block behind distributed training — every GPU contributes a value, and every GPU ends up with the sum.

```c
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
    int nGPUs = 4;
    ncclComm_t comms[4];
    int devs[4] = {0, 1, 2, 3};

    // Initialize one NCCL communicator per GPU, all aware of each other
    ncclCommInitAll(comms, nGPUs, devs);

    float *sendbuf[4], *recvbuf[4];
    size_t count = 1024;

    for (int i = 0; i < nGPUs; i++) {
        cudaSetDevice(i);
        cudaMalloc(&sendbuf[i], count * sizeof(float));
        cudaMalloc(&recvbuf[i], count * sizeof(float));
        // ... fill sendbuf[i] with this GPU's gradient values ...
    }

    // Every GPU calls AllReduce; NCCL handles the ring/tree pattern internally
    ncclGroupStart();
    for (int i = 0; i < nGPUs; i++) {
        cudaSetDevice(i);
        ncclAllReduce(sendbuf[i], recvbuf[i], count, ncclFloat, ncclSum, comms[i], 0);
    }
    ncclGroupEnd();

    // recvbuf[i] on every GPU now holds the SAME summed result
    return 0;
}
```

The key line is `ncclAllReduce`. Every GPU calls it with its own local data; NCCL figures out the fastest path (NVLink if available, ring or tree pattern, using peer access under the hood) and every GPU ends up with the combined total — without your code needing to know or care about the underlying topology.

### Example C: What this looks like in PyTorch (the level most people actually use)

In practice, almost nobody calls NCCL directly — a framework does it for you:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")  # tells PyTorch to use NCCL under the hood
model = DDP(model.to(local_rank), device_ids=[local_rank])

# Normal training loop
output = model(batch)
loss = loss_fn(output, target)
loss.backward()   # <-- DDP automatically triggers NCCL AllReduce here,
                   #     averaging gradients across all GPUs before the optimizer step
optimizer.step()
```

That single line `backend="nccl"` is the whole point of this document: it tells PyTorch to use NCCL, which uses peer access, which (when available) uses NVLink — three layers, invisible to you, working together every time `loss.backward()` runs.

---

## 6. Putting it all together: one training step, end to end

1. Each of 8 GPUs computes gradients on its own slice of the batch (pure computation, no communication yet).
2. `loss.backward()` finishes, and DDP tells NCCL it's time to reconcile gradients — this triggers an **AllReduce**.
3. NCCL checks the topology it detected at startup: these 8 GPUs share **NVSwitch**, so it builds an efficient ring/tree pattern across the NVLink fabric.
4. Data moves GPU-to-GPU using **peer access**, so none of it detours through CPU memory.
5. Within milliseconds, all 8 GPUs hold the identical averaged gradient and proceed to update their local copy of the model weights identically.
6. If this were spread across multiple *servers* instead of one box, step 4 would instead use **GPUDirect RDMA** over InfiniBand/Ethernet for the cross-server portion, while still using NVLink+peer access for the portion within each server.

The result: training scales close to linearly as you add GPUs, because communication is barely a bottleneck.

---

## 7. Quick reference summary

| Concept | What it is | Where it lives |
|---|---|---|
| **PCIe** | General-purpose shared bus | Every system, default GPU connection |
| **NVLink** | Dedicated high-bandwidth GPU-to-GPU link | Hardware, mainly within a server |
| **NVSwitch** | Switch chip giving all-to-all NVLink connectivity | Hardware, multi-GPU servers |
| **Peer Access (P2P)** | Permission for one GPU to read/write another's memory directly | CUDA software layer |
| **GPUDirect RDMA** | Peer access extended across the network | CUDA + network card, multi-server |
| **NCCL** | Library that orchestrates efficient collective data movement | Software, used by PyTorch/TensorFlow/etc. |
| **AllReduce** | The specific collective op behind gradient averaging | Most common NCCL operation in training |

Together, these let a cluster of GPUs behave less like a pile of separate chips and more like one very large accelerator.