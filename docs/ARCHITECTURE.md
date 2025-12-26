# Architecture & Technical Design

This document provides in-depth technical details about the sparse matrix implementation.

## Table of Contents

1. [CSR Format Details](#csr-format-details)
2. [Memory Access Patterns](#memory-access-patterns)
3. [Kernel Design Decisions](#kernel-design-decisions)
4. [Performance Bottlenecks](#performance-bottlenecks)
5. [Optimization Strategies](#optimization-strategies)

---

## CSR Format Details

### Storage Layout

Compressed Sparse Row (CSR) stores an M×N matrix with NNZ non-zeros using three arrays:

```
Matrix A (4×5):              CSR Representation:
┌─────────────────────┐
│  1   0   0   2   0  │      data:    [1, 2, 3, 4, 5, 6, 7]
│  0   3   4   0   0  │      indices: [0, 3, 1, 2, 0, 4, 2]
│  5   0   0   0   6  │      indptr:  [0, 2, 4, 6, 7]
│  0   0   7   0   0  │
└─────────────────────┘
```

- **data** (NNZ elements): Non-zero values in row-major order
- **indices** (NNZ elements): Column index for each non-zero
- **indptr** (M+1 elements): Row i spans `data[indptr[i]:indptr[i+1]]`

### Memory Footprint

```
CSR:   NNZ × 4 (data) + NNZ × 4 (indices) + (M+1) × 4 (indptr)
     = 8×NNZ + 4M + 4 bytes

Dense: M × N × 4 bytes

Compression ratio: (M × N) / (2×NNZ + M)
```

For 50K×50K matrix at 0.1% density:
- Dense: 10 GB
- CSR: 20 MB (500× compression)

---

## Memory Access Patterns

### The Fundamental Challenge

SpMV computes `y = A × x`:

```python
for i in range(num_rows):
    for j in range(indptr[i], indptr[i+1]):
        y[i] += data[j] * x[indices[j]]
                          ↑
                          Random access!
```

The column indices `indices[j]` are essentially random, causing **scattered reads** from vector `x`.

### Coalescing Analysis

**Ideal (Dense GEMM):**
```
Thread 0: x[0]  ─┐
Thread 1: x[1]  ─┼─► Single 128-byte transaction
Thread 2: x[2]  ─┤
...             ─┘
```

**Reality (Sparse SpMV):**
```
Thread 0: x[1042]  ─► Transaction 1
Thread 1: x[7]     ─► Transaction 2
Thread 2: x[892]   ─► Transaction 3
...                   (32 separate transactions!)
```

Each warp potentially generates 32 memory transactions instead of 1-4.

---

## Kernel Design Decisions

### Scalar Kernel

```
Strategy: 1 thread → 1 row

Thread 0 ──► Row 0: [■ · · ■ ·]
Thread 1 ──► Row 1: [· ■ ■ · ·]
Thread 2 ──► Row 2: [■ · · · ■]
```

**Trade-offs:**
- [+] Minimal overhead, good for short rows
- [-] Thread divergence: threads with long rows block others
- [-] Poor occupancy if num_rows < num_SMs x threads_per_SM

### Vector Kernel

```
Strategy: 1 warp (32 threads) → 1 row

Warp 0, Lane 0-31 ──► Row 0: [■ ■ ■ ■ ■ ■ ...]
                     Lane 0: j=0, j=32, j=64...
                     Lane 1: j=1, j=33, j=65...
                     ...
                     Shuffle reduction → y[0]
```

**Trade-offs:**
- [+] Parallel reduction within row
- [+] Better for dense rows (many non-zeros)
- [-] Wastes 31 threads if row has 1-5 non-zeros

### Adaptive Kernel

```python
if row_length <= 32:
    # Scalar: only lane 0 works
else:
    # Vector: all 32 lanes participate
```

Combines both strategies based on row characteristics.

---

## Performance Bottlenecks

### Measured Bandwidth Breakdown

| Kernel     | Achieved BW | % of Peak | % of cuSPARSE |
|------------|-------------|-----------|---------------|
| Scalar     | 20.8 GB/s   | 2.8%      | 22%           |
| Vector     | 29.3 GB/s   | 4.0%      | 31%           |
| cuSPARSE   | 95.5 GB/s   | 13.1%     | 100%          |

### Why We're Losing Bandwidth

**1. Vector x Access Pattern (~50% of loss)**

Random column indices cause cache thrashing and non-coalesced access:

```
Ideal:    1 transaction serves 32 threads
Reality:  Up to 32 transactions serve 32 threads
Impact:   32× memory traffic amplification
```

**2. Row Length Variance (~25% of loss)**

```
Row lengths in our test matrix:
- Mean: 50 non-zeros
- Std:  7 non-zeros
- Range: 38-67 non-zeros

Warp divergence: threads finish at different times
```

**3. Kernel Launch Overhead (~10% of loss)**

Numba JIT compilation and Python-to-CUDA overhead.

**4. No Prefetching (~15% of loss)**

cuSPARSE uses texture cache and hardware prefetching; we don't.

---

## Optimization Strategies

### Strategy 1: Shared Memory for Vector x

Cache frequently accessed x values in shared memory:

```python
@cuda.jit
def spmv_cached_kernel(...):
    # Phase 1: Load x segment into shared memory
    shared_x = cuda.shared.array(BLOCK_SIZE, dtype=float32)
    
    # Phase 2: Process rows using cached values
    # Falls back to global memory for cache misses
```

**Expected gain:** 1.5-2× (reduces random global access)

### Strategy 2: Row Binning

Separate rows by length, launch different kernels:

```
Short rows (1-8 nnz):     Scalar kernel, high parallelism
Medium rows (9-64 nnz):   Warp-per-row kernel
Long rows (65+ nnz):      Multiple warps per row
```

**Expected gain:** 1.3-1.5× (better load balancing)

### Strategy 3: CSR5 Format

Reorganize data for coalesced access within "tiles":

```
Standard CSR:  Rows processed sequentially
CSR5:          2D tiles, coalesced within each tile
```

**Expected gain:** 2-3× (fundamentally better access pattern)
**Complexity:** High (requires format conversion)

### Strategy 4: ELL Format for Regular Matrices

When row lengths are uniform, ELL enables perfect coalescing:

```
ELL Format:  Pad all rows to max length
             Store in column-major order
             → Consecutive threads access consecutive memory
```

**Best for:** Structured matrices (FEM, stencils)
**Expected gain:** Up to 3× on suitable matrices

---

## Implementation Priorities

Based on effort vs. impact:

| Priority | Optimization           | Effort  | Expected Gain             |
|----------|------------------------|---------|---------------------------|
| 1        | Shared memory caching  | Medium  | 1.5–2×                    |
| 2        | Row binning            | Low     | 1.3×                      |
| 3        | Block size tuning      | Low     | 1.1×                      |
| 4        | ELL format             | Medium  | 2× (specific matrices)    |
| 5        | CSR5 format            | High    | 2–3×                      |

---

## Appendix: Memory Bandwidth Calculation

For SpMV `y = A × x`:

**Bytes Read:**
- `data`: NNZ × 4 bytes
- `indices`: NNZ × 4 bytes
- `indptr`: (M+1) × 4 bytes
- `x`: N × 4 bytes (pessimistic: assume full read)

**Bytes Written:**
- `y`: M × 4 bytes

**Total:** `8×NNZ + 4M + 4N + 4` bytes

**Effective Bandwidth:** `Total Bytes / Execution Time`

For 50K×50K matrix, 0.1% density (2.5M nnz):
```
Total = 8×2.5M + 4×50K + 4×50K + 4 ≈ 20.4 MB
At 0.7ms: 20.4 MB / 0.0007s = 29.1 GB/s
```

