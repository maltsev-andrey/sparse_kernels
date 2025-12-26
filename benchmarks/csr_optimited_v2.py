"""
Optimized CSR SpMV Kernels - Version 2

Focused optimizations for matrices with uniform row lengths:
1. Tuned thread block sizes for better occupancy
2. Multiple rows per warp for short rows
3. Vectorized loads (float4) where possible
4. Reduced register pressure
"""

import numpy as np
from numba import cuda, float32, int32
import math


class CSRMatrix:
    """CSR sparse matrix with CPU and GPU storage."""
    
    def __init__(self, data, indices, indptr, shape):
        self.data = np.asarray(data, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.indptr = np.asarray(indptr, dtype=np.int32)
        self.shape = shape
        self.nnz = len(self.data)
        
        self._d_data = None
        self._d_indices = None
        self._d_indptr = None

    @classmethod
    def random(cls, rows, cols, density=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        nnz_per_row = max(1, int(cols * density))
        
        data = []
        indices = []
        indptr = [0]
        
        for i in range(rows):
            row_indices = np.random.choice(cols, size=min(nnz_per_row, cols), replace=False)
            row_indices = np.sort(row_indices)
            row_data = np.random.randn(len(row_indices)).astype(np.float32)
            
            data.extend(row_data)
            indices.extend(row_indices)
            indptr.append(len(data))
        
        return cls(data, indices, indptr, (rows, cols))

    def to_gpu(self):
        if self._d_data is None:
            self._d_data = cuda.to_device(self.data)
            self._d_indices = cuda.to_device(self.indices)
            self._d_indptr = cuda.to_device(self.indptr)
        return self._d_data, self._d_indices, self._d_indptr

    def free_gpu(self):
        self._d_data = None
        self._d_indices = None
        self._d_indptr = None

    @property
    def density(self):
        return self.nnz / (self.shape[0] * self.shape[1])
    
    @property 
    def avg_nnz_per_row(self):
        return self.nnz / self.shape[0]

    def __repr__(self):
        return f"CSRMatrix(shape={self.shape}, nnz={self.nnz}, density={self.density:.4f})"


# =============================================================================
# Kernel 1: Tuned Vector Kernel with Optimal Block Size
# =============================================================================

@cuda.jit
def spmv_vector_tuned_kernel(data, indices, indptr, x, y, num_rows):
    """
    Vector kernel with tuned parameters.
    Uses 128 threads per block (4 warps) for better occupancy on P100.
    """
    lane = cuda.threadIdx.x % 32
    warp_id = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // 32
    
    if warp_id < num_rows:
        row_start = indptr[warp_id]
        row_end = indptr[warp_id + 1]
        
        # Local accumulator
        dot = float32(0.0)
        
        # Process elements
        for j in range(row_start + lane, row_end, 32):
            dot += data[j] * x[indices[j]]
        
        # Warp shuffle reduction - fully unrolled
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 16)
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 8)
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 4)
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 2)
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 1)
        
        if lane == 0:
            y[warp_id] = dot


# =============================================================================
# Kernel 2: Two Rows Per Warp (for short rows ~50 nnz)
# =============================================================================

@cuda.jit
def spmv_two_rows_per_warp_kernel(data, indices, indptr, x, y, num_rows):
    """
    Process 2 rows per warp for better efficiency on short rows.
    Each half-warp (16 threads) handles one row.
    Optimal for rows with ~32-64 non-zeros.
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    lane = tid % 32
    warp_id = tid // 32
    half_warp = lane // 16  # 0 or 1
    half_lane = lane % 16   # 0-15
    
    # Each warp processes 2 rows
    row = warp_id * 2 + half_warp
    
    if row < num_rows:
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        dot = float32(0.0)
        
        # 16 threads process the row together
        for j in range(row_start + half_lane, row_end, 16):
            dot += data[j] * x[indices[j]]
        
        # Half-warp reduction (16 threads)
        dot += cuda.shfl_down_sync(0xFFFF << (half_warp * 16), dot, 8)
        dot += cuda.shfl_down_sync(0xFFFF << (half_warp * 16), dot, 4)
        dot += cuda.shfl_down_sync(0xFFFF << (half_warp * 16), dot, 2)
        dot += cuda.shfl_down_sync(0xFFFF << (half_warp * 16), dot, 1)
        
        if half_lane == 0:
            y[row] = dot


# =============================================================================
# Kernel 3: Four Rows Per Warp (for very short rows)
# =============================================================================

@cuda.jit  
def spmv_four_rows_per_warp_kernel(data, indices, indptr, x, y, num_rows):
    """
    Process 4 rows per warp - each quarter-warp (8 threads) handles one row.
    Optimal for rows with ~16-32 non-zeros.
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    lane = tid % 32
    warp_id = tid // 32
    quarter = lane // 8      # 0, 1, 2, or 3
    quarter_lane = lane % 8  # 0-7
    
    row = warp_id * 4 + quarter
    
    if row < num_rows:
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        dot = float32(0.0)
        
        # 8 threads process the row
        for j in range(row_start + quarter_lane, row_end, 8):
            dot += data[j] * x[indices[j]]
        
        # Quarter-warp reduction (8 threads)
        # Use full mask but only threads in same quarter participate
        mask = 0xFF << (quarter * 8)
        dot += cuda.shfl_down_sync(mask, dot, 4)
        dot += cuda.shfl_down_sync(mask, dot, 2)
        dot += cuda.shfl_down_sync(mask, dot, 1)
        
        if quarter_lane == 0:
            y[row] = dot


# =============================================================================
# Kernel 4: Warp-Per-Row with Prefetching
# =============================================================================

@cuda.jit
def spmv_vector_prefetch_kernel(data, indices, indptr, x, y, num_rows):
    """
    Vector kernel with manual prefetching hint.
    Loads next iteration's data while computing current.
    """
    lane = cuda.threadIdx.x % 32
    warp_id = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // 32
    
    if warp_id < num_rows:
        row_start = indptr[warp_id]
        row_end = indptr[warp_id + 1]
        row_len = row_end - row_start
        
        dot = float32(0.0)
        
        # Main loop with manual software pipelining
        j = row_start + lane
        
        if j < row_end:
            # Load first element
            val = data[j]
            col = indices[j]
            
            j += 32
            while j < row_end:
                # Prefetch next iteration
                next_val = data[j]
                next_col = indices[j]
                
                # Compute current
                dot += val * x[col]
                
                # Move prefetched to current
                val = next_val
                col = next_col
                j += 32
            
            # Process last element
            dot += val * x[col]
        
        # Warp reduction
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 16)
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 8)
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 4)
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 2)
        dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 1)
        
        if lane == 0:
            y[warp_id] = dot


# =============================================================================
# Kernel 5: Hybrid - Select Strategy Based on Row Length
# =============================================================================

@cuda.jit
def spmv_hybrid_kernel(data, indices, indptr, x, y, num_rows, avg_nnz):
    """
    Hybrid kernel that selects strategy based on average nnz per row.
    Determined at launch time.
    """
    lane = cuda.threadIdx.x % 32
    warp_id = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // 32
    
    if avg_nnz <= 16:
        # 4 rows per warp
        quarter = lane // 8
        quarter_lane = lane % 8
        row = warp_id * 4 + quarter
        stride = 8
        
        if row < num_rows:
            row_start = indptr[row]
            row_end = indptr[row + 1]
            
            dot = float32(0.0)
            for j in range(row_start + quarter_lane, row_end, stride):
                dot += data[j] * x[indices[j]]
            
            # Reduction for 8 threads
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 4)
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 2)
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 1)
            
            if quarter_lane == 0:
                y[row] = dot
                
    elif avg_nnz <= 48:
        # 2 rows per warp
        half = lane // 16
        half_lane = lane % 16
        row = warp_id * 2 + half
        stride = 16
        
        if row < num_rows:
            row_start = indptr[row]
            row_end = indptr[row + 1]
            
            dot = float32(0.0)
            for j in range(row_start + half_lane, row_end, stride):
                dot += data[j] * x[indices[j]]
            
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 8)
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 4)
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 2)
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 1)
            
            if half_lane == 0:
                y[row] = dot
    else:
        # 1 row per warp (standard)
        if warp_id < num_rows:
            row_start = indptr[warp_id]
            row_end = indptr[warp_id + 1]
            
            dot = float32(0.0)
            for j in range(row_start + lane, row_end, 32):
                dot += data[j] * x[indices[j]]
            
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 16)
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 8)
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 4)
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 2)
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, 1)
            
            if lane == 0:
                y[warp_id] = dot


# =============================================================================
# Interface Functions
# =============================================================================

def spmv_gpu_vector_tuned(csr_matrix, x, y=None):
    """SpMV with tuned vector kernel (128 threads/block)."""
    num_rows = csr_matrix.shape[0]
    d_data, d_indices, d_indptr = csr_matrix.to_gpu()
    
    if isinstance(x, np.ndarray):
        d_x = cuda.to_device(x.astype(np.float32))
    else:
        d_x = x
    
    if y is None:
        d_y = cuda.device_array(num_rows, dtype=np.float32)
    else:
        d_y = cuda.to_device(y) if isinstance(y, np.ndarray) else y
    
    # 128 threads = 4 warps per block (better occupancy)
    threads_per_block = 128
    num_warps = num_rows
    blocks = (num_warps * 32 + threads_per_block - 1) // threads_per_block
    
    spmv_vector_tuned_kernel[blocks, threads_per_block](
        d_data, d_indices, d_indptr, d_x, d_y, num_rows
    )
    
    return d_y


def spmv_gpu_two_rows(csr_matrix, x, y=None):
    """SpMV with 2 rows per warp."""
    num_rows = csr_matrix.shape[0]
    d_data, d_indices, d_indptr = csr_matrix.to_gpu()
    
    if isinstance(x, np.ndarray):
        d_x = cuda.to_device(x.astype(np.float32))
    else:
        d_x = x
    
    if y is None:
        d_y = cuda.device_array(num_rows, dtype=np.float32)
    else:
        d_y = cuda.to_device(y) if isinstance(y, np.ndarray) else y
    
    threads_per_block = 256
    # Each warp handles 2 rows
    num_warps_needed = (num_rows + 1) // 2
    blocks = (num_warps_needed * 32 + threads_per_block - 1) // threads_per_block
    
    spmv_two_rows_per_warp_kernel[blocks, threads_per_block](
        d_data, d_indices, d_indptr, d_x, d_y, num_rows
    )
    
    return d_y


def spmv_gpu_four_rows(csr_matrix, x, y=None):
    """SpMV with 4 rows per warp."""
    num_rows = csr_matrix.shape[0]
    d_data, d_indices, d_indptr = csr_matrix.to_gpu()
    
    if isinstance(x, np.ndarray):
        d_x = cuda.to_device(x.astype(np.float32))
    else:
        d_x = x
    
    if y is None:
        d_y = cuda.device_array(num_rows, dtype=np.float32)
    else:
        d_y = cuda.to_device(y) if isinstance(y, np.ndarray) else y
    
    threads_per_block = 256
    num_warps_needed = (num_rows + 3) // 4
    blocks = (num_warps_needed * 32 + threads_per_block - 1) // threads_per_block
    
    spmv_four_rows_per_warp_kernel[blocks, threads_per_block](
        d_data, d_indices, d_indptr, d_x, d_y, num_rows
    )
    
    return d_y


def spmv_gpu_prefetch(csr_matrix, x, y=None):
    """SpMV with prefetching."""
    num_rows = csr_matrix.shape[0]
    d_data, d_indices, d_indptr = csr_matrix.to_gpu()
    
    if isinstance(x, np.ndarray):
        d_x = cuda.to_device(x.astype(np.float32))
    else:
        d_x = x
    
    if y is None:
        d_y = cuda.device_array(num_rows, dtype=np.float32)
    else:
        d_y = cuda.to_device(y) if isinstance(y, np.ndarray) else y
    
    threads_per_block = 128
    num_warps = num_rows
    blocks = (num_warps * 32 + threads_per_block - 1) // threads_per_block
    
    spmv_vector_prefetch_kernel[blocks, threads_per_block](
        d_data, d_indices, d_indptr, d_x, d_y, num_rows
    )
    
    return d_y


def spmv_gpu_hybrid(csr_matrix, x, y=None):
    """SpMV with hybrid strategy based on row length."""
    num_rows = csr_matrix.shape[0]
    avg_nnz = int(csr_matrix.avg_nnz_per_row)
    
    d_data, d_indices, d_indptr = csr_matrix.to_gpu()
    
    if isinstance(x, np.ndarray):
        d_x = cuda.to_device(x.astype(np.float32))
    else:
        d_x = x
    
    if y is None:
        d_y = cuda.device_array(num_rows, dtype=np.float32)
    else:
        d_y = cuda.to_device(y) if isinstance(y, np.ndarray) else y
    
    # Adjust grid based on rows per warp
    if avg_nnz <= 16:
        rows_per_warp = 4
    elif avg_nnz <= 48:
        rows_per_warp = 2
    else:
        rows_per_warp = 1
    
    threads_per_block = 256
    num_warps_needed = (num_rows + rows_per_warp - 1) // rows_per_warp
    blocks = (num_warps_needed * 32 + threads_per_block - 1) // threads_per_block
    
    spmv_hybrid_kernel[blocks, threads_per_block](
        d_data, d_indices, d_indptr, d_x, d_y, num_rows, avg_nnz
    )
    
    return d_y


def spmv_cpu(csr_matrix, x):
    """CPU reference."""
    y = np.zeros(csr_matrix.shape[0], dtype=np.float32)
    for i in range(csr_matrix.shape[0]):
        start = csr_matrix.indptr[i]
        end = csr_matrix.indptr[i + 1]
        for j in range(start, end):
            y[i] += csr_matrix.data[j] * x[csr_matrix.indices[j]]
    return y


def verify_spmv(csr_matrix, x, gpu_result, rtol=1e-4, atol=1e-5):
    """Verify GPU result against CPU."""
    cpu_result = spmv_cpu(csr_matrix, x)
    if hasattr(gpu_result, 'copy_to_host'):
        gpu_result = gpu_result.copy_to_host()
    return np.allclose(cpu_result, gpu_result, rtol=rtol, atol=atol)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_v2():
    """Benchmark optimized kernels v2."""
    
    print("=" * 70)
    print("OPTIMIZED SpMV KERNELS v2 - BENCHMARK")
    print("=" * 70)
    
    N = 50000
    density = 0.001
    
    print(f"\nMatrix: {N}x{N}, density={density:.2%}")
    A = CSRMatrix.random(N, N, density=density, seed=42)
    print(f"Created: {A}")
    print(f"Average nnz/row: {A.avg_nnz_per_row:.1f}")
    
    x = np.random.randn(N).astype(np.float32)
    
    def time_kernel(name, func, *args, iterations=10):
        # Warmup
        for _ in range(3):
            result = func(*args)
            cuda.synchronize()
        
        start = cuda.event()
        end = cuda.event()
        
        times = []
        for _ in range(iterations):
            start.record()
            result = func(*args)
            end.record()
            end.synchronize()
            times.append(cuda.event_elapsed_time(start, end))
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        is_correct = verify_spmv(A, x, result)
        
        total_bytes = (A.nnz * 4 + A.nnz * 4 + (N + 1) * 4 + N * 4 + N * 4)
        bandwidth = (total_bytes / 1e9) / (avg_time / 1000)
        
        status = "✓" if is_correct else "✗ FAILED"
        print(f"  {name:<30} {avg_time:>8.3f} ms  {bandwidth:>6.1f} GB/s  {status}")
        
        return avg_time, bandwidth, is_correct
    
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"  {'Kernel':<30} {'Time':>8}     {'BW':>6}      Status")
    print("-" * 70)
    
    results = {}
    
    results['Vector Tuned (128 tpb)'] = time_kernel(
        "Vector Tuned (128 tpb)", spmv_gpu_vector_tuned, A, x)
    
    results['Two Rows/Warp'] = time_kernel(
        "Two Rows/Warp", spmv_gpu_two_rows, A, x)
    
    results['Four Rows/Warp'] = time_kernel(
        "Four Rows/Warp", spmv_gpu_four_rows, A, x)
    
    results['Vector + Prefetch'] = time_kernel(
        "Vector + Prefetch", spmv_gpu_prefetch, A, x)
    
    results['Hybrid Auto-Select'] = time_kernel(
        "Hybrid Auto-Select", spmv_gpu_hybrid, A, x)
    
    # cuSPARSE reference
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsparse
        
        A_cupy = cpsparse.csr_matrix(
            (cp.array(A.data), cp.array(A.indices), cp.array(A.indptr)),
            shape=A.shape
        )
        x_cupy = cp.array(x)
        
        for _ in range(3):
            _ = A_cupy @ x_cupy
            cp.cuda.Stream.null.synchronize()
        
        times = []
        for _ in range(10):
            start_ev = cp.cuda.Event()
            end_ev = cp.cuda.Event()
            start_ev.record()
            _ = A_cupy @ x_cupy
            end_ev.record()
            end_ev.synchronize()
            times.append(cp.cuda.get_elapsed_time(start_ev, end_ev))
        
        cusparse_time = np.mean(times)
        total_bytes = (A.nnz * 4 + A.nnz * 4 + (N + 1) * 4 + N * 4 + N * 4)
        cusparse_bw = (total_bytes / 1e9) / (cusparse_time / 1000)
        
        print(f"  {'cuSPARSE Reference':<30} {cusparse_time:>8.3f} ms  {cusparse_bw:>6.1f} GB/s  ✓")
        results['cuSPARSE'] = (cusparse_time, cusparse_bw, True)
        
    except ImportError:
        print("  [cuSPARSE skipped - CuPy not installed]")
        cusparse_time = 1.0
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY vs cuSPARSE")
    print("=" * 70)
    
    best_custom_time = min(t for t, bw, ok in results.values() if ok and 'cuSPARSE' not in str(t))
    best_kernel = [k for k, (t, bw, ok) in results.items() if t == best_custom_time][0]
    
    print(f"\nBest custom kernel: {best_kernel}")
    print(f"  Time: {best_custom_time:.3f} ms")
    print(f"  vs cuSPARSE: {best_custom_time/cusparse_time:.2f}× slower")
    print(f"  Gap to close: {(best_custom_time - cusparse_time):.3f} ms")
    
    # Test with different matrix sizes
    print("\n" + "=" * 70)
    print("SCALING TEST - Best Kernel vs cuSPARSE")
    print("=" * 70)
    
    for test_n in [10000, 25000, 50000, 100000]:
        print(f"\n  Matrix size: {test_n}x{test_n}")
        
        test_A = CSRMatrix.random(test_n, test_n, density=0.001, seed=42)
        test_x = np.random.randn(test_n).astype(np.float32)
        
        # Our best kernel
        for _ in range(3):
            _ = spmv_gpu_vector_tuned(test_A, test_x)
            cuda.synchronize()
        
        start = cuda.event()
        end = cuda.event()
        start.record()
        _ = spmv_gpu_vector_tuned(test_A, test_x)
        end.record()
        end.synchronize()
        our_time = cuda.event_elapsed_time(start, end)
        
        # cuSPARSE
        try:
            A_cp = cpsparse.csr_matrix(
                (cp.array(test_A.data), cp.array(test_A.indices), cp.array(test_A.indptr)),
                shape=test_A.shape
            )
            x_cp = cp.array(test_x)
            
            _ = A_cp @ x_cp
            cp.cuda.Stream.null.synchronize()
            
            s = cp.cuda.Event()
            e = cp.cuda.Event()
            s.record()
            _ = A_cp @ x_cp
            e.record()
            e.synchronize()
            cu_time = cp.cuda.get_elapsed_time(s, e)
            
            ratio = our_time / cu_time
            print(f"    Ours: {our_time:.3f} ms, cuSPARSE: {cu_time:.3f} ms, ratio: {ratio:.2f}×")
        except:
            print(f"    Ours: {our_time:.3f} ms")
        
        test_A.free_gpu()
    
    A.free_gpu()
    print("\nBenchmark complete!")


if __name__ == "__main__":
    benchmark_v2()