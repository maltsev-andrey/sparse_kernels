"""
Optimized CSR SpMV Kernels

Improvements over basic implementation:
1. Shared memory caching for x vector segments
2. Row binning - group rows by length for better load balancing
3. Optimized memory access patterns
4. Better warp utilization
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
        
        # GPU arrays
        self._d_data = None
        self._d_indices = None
        self._d_indptr = None
        
        # Row binning data (computed on demand)
        self._row_bins = None
        self._d_row_bins = None

    @classmethod
    def random(cls, rows, cols, density=0.01, seed=None):
        """Generate random sparse matrix."""
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
        """Transfer matrix data to GPU."""
        if self._d_data is None:
            self._d_data = cuda.to_device(self.data)
            self._d_indices = cuda.to_device(self.indices)
            self._d_indptr = cuda.to_device(self.indptr)
        return self._d_data, self._d_indices, self._d_indptr

    def compute_row_bins(self):
        """
        Compute row binning for load-balanced SpMV.
        
        Bins rows by their length (nnz per row):
        - Bin 0: 1-32 nnz (scalar processing)
        - Bin 1: 33-128 nnz (1 warp)
        - Bin 2: 129-512 nnz (2-4 warps)
        - Bin 3: 513+ nnz (multiple warps)
        """
        row_lengths = np.diff(self.indptr)
        
        bins = {
            'short': [],      # <= 32: scalar
            'medium': [],     # 33-128: 1 warp
            'long': [],       # 129-512: multi-warp
            'very_long': []   # > 512: special handling
        }
        
        for i, length in enumerate(row_lengths):
            if length <= 32:
                bins['short'].append(i)
            elif length <= 128:
                bins['medium'].append(i)
            elif length <= 512:
                bins['long'].append(i)
            else:
                bins['very_long'].append(i)
        
        self._row_bins = {
            'short': np.array(bins['short'], dtype=np.int32),
            'medium': np.array(bins['medium'], dtype=np.int32),
            'long': np.array(bins['long'], dtype=np.int32),
            'very_long': np.array(bins['very_long'], dtype=np.int32)
        }
        
        return self._row_bins

    def get_row_bins_gpu(self):
        """Get row bins on GPU."""
        if self._row_bins is None:
            self.compute_row_bins()
        
        if self._d_row_bins is None:
            self._d_row_bins = {
                name: cuda.to_device(arr) if len(arr) > 0 else None
                for name, arr in self._row_bins.items()
            }
        
        return self._d_row_bins

    def free_gpu(self):
        """Free GPU memory."""
        self._d_data = None
        self._d_indices = None
        self._d_indptr = None
        self._d_row_bins = None

    @property
    def density(self):
        return self.nnz / (self.shape[0] * self.shape[1])

    def __repr__(self):
        return f"CSRMatrix(shape={self.shape}, nnz={self.nnz}, density={self.density:.4f})"


# =============================================================================
# Optimization 1: Shared Memory Caching for X Vector
# =============================================================================

# Shared memory size (elements) - adjust based on GPU shared memory
SHARED_X_SIZE = 1024


@cuda.jit
def spmv_scalar_shared_kernel(data, indices, indptr, x, y, num_rows, num_cols):
    """
    Scalar SpMV with shared memory caching for x vector.
    
    Each block loads a tile of x into shared memory.
    Works best when column indices have locality.
    """
    # Shared memory for x vector tile
    shared_x = cuda.shared.array(SHARED_X_SIZE, dtype=float32)
    
    row = cuda.grid(1)
    tid = cuda.threadIdx.x
    
    if row < num_rows:
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        dot = float32(0.0)
        
        # Process in tiles
        for tile_start in range(0, num_cols, SHARED_X_SIZE):
            # Cooperatively load x tile into shared memory
            cuda.syncthreads()
            
            for i in range(tid, SHARED_X_SIZE, cuda.blockDim.x):
                idx = tile_start + i
                if idx < num_cols:
                    shared_x[i] = x[idx]
                else:
                    shared_x[i] = float32(0.0)
            
            cuda.syncthreads()
            
            # Process row elements that fall in this tile
            for j in range(row_start, row_end):
                col = indices[j]
                if tile_start <= col < tile_start + SHARED_X_SIZE:
                    dot += data[j] * shared_x[col - tile_start]
        
        y[row] = dot


@cuda.jit
def spmv_vector_optimized_kernel(data, indices, indptr, x, y, num_rows):
    """
    Optimized vector kernel with explicit float32 and better memory access.
    
    One warp per row with optimized reduction.
    """
    lane = cuda.threadIdx.x % 32
    warp_id = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // 32
    
    if warp_id < num_rows:
        row_start = indptr[warp_id]
        row_end = indptr[warp_id + 1]
        
        # Use float32 explicitly
        dot = float32(0.0)
        
        # Coalesced access pattern within warp
        for j in range(row_start + lane, row_end, 32):
            dot += data[j] * x[indices[j]]
        
        # Warp reduction using shuffle
        for offset in (16, 8, 4, 2, 1):
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, offset)
        
        if lane == 0:
            y[warp_id] = dot


# =============================================================================
# Optimization 2: Row Binning - Process Similar Rows Together
# =============================================================================

@cuda.jit
def spmv_binned_short_kernel(data, indices, indptr, x, y, row_indices, num_bin_rows):
    """
    Process short rows (<=32 nnz) with scalar approach.
    One thread per row.
    """
    tid = cuda.grid(1)
    
    if tid < num_bin_rows:
        row = row_indices[tid]
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        dot = float32(0.0)
        for j in range(row_start, row_end):
            dot += data[j] * x[indices[j]]
        
        y[row] = dot


@cuda.jit
def spmv_binned_medium_kernel(data, indices, indptr, x, y, row_indices, num_bin_rows):
    """
    Process medium rows (33-128 nnz) with one warp per row.
    """
    lane = cuda.threadIdx.x % 32
    warp_local = cuda.threadIdx.x // 32
    warps_per_block = cuda.blockDim.x // 32
    
    warp_global = cuda.blockIdx.x * warps_per_block + warp_local
    
    if warp_global < num_bin_rows:
        row = row_indices[warp_global]
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        dot = float32(0.0)
        for j in range(row_start + lane, row_end, 32):
            dot += data[j] * x[indices[j]]
        
        # Warp reduction
        for offset in (16, 8, 4, 2, 1):
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, offset)
        
        if lane == 0:
            y[row] = dot


@cuda.jit
def spmv_binned_long_kernel(data, indices, indptr, x, y, row_indices, num_bin_rows):
    """
    Process long rows (129-512 nnz) with multiple warps per row.
    Uses 2 warps (64 threads) per row.
    """
    tid = cuda.threadIdx.x
    lane = tid % 32
    warp_in_block = tid // 32
    warps_per_row = 2
    
    # Which row this thread group handles
    rows_per_block = cuda.blockDim.x // (32 * warps_per_row)
    local_row = warp_in_block // warps_per_row
    warp_in_row = warp_in_block % warps_per_row
    
    global_row_idx = cuda.blockIdx.x * rows_per_block + local_row
    
    if global_row_idx < num_bin_rows:
        row = row_indices[global_row_idx]
        row_start = indptr[row]
        row_end = indptr[row + 1]
        row_len = row_end - row_start
        
        # Each warp handles a portion
        elements_per_warp = (row_len + warps_per_row - 1) // warps_per_row
        my_start = row_start + warp_in_row * elements_per_warp
        my_end = min(my_start + elements_per_warp, row_end)
        
        dot = float32(0.0)
        for j in range(my_start + lane, my_end, 32):
            if j < row_end:
                dot += data[j] * x[indices[j]]
        
        # Warp reduction
        for offset in (16, 8, 4, 2, 1):
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, offset)
        
        # Lane 0 of each warp atomically adds to result
        if lane == 0:
            cuda.atomic.add(y, row, dot)


# =============================================================================
# Optimization 3: Cache-Optimized Kernel
# =============================================================================

@cuda.jit
def spmv_cache_optimized_kernel(data, indices, indptr, x, y, num_rows):
    """
    Cache-optimized kernel focusing on L2 cache utilization.
    
    - Processes multiple rows per thread block to improve x vector cache reuse
    - Uses local accumulation before writing
    """
    shared_results = cuda.shared.array(256, dtype=float32)  # Block results
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    
    # Each block handles a chunk of rows
    rows_per_block = 8
    row_base = bid * rows_per_block
    
    # Initialize shared memory
    if tid < rows_per_block:
        shared_results[tid] = float32(0.0)
    cuda.syncthreads()
    
    # Process rows assigned to this block
    for local_row in range(rows_per_block):
        row = row_base + local_row
        if row >= num_rows:
            break
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        row_len = row_end - row_start
        
        # Parallel reduction within block
        dot = float32(0.0)
        for j in range(row_start + tid, row_end, block_size):
            dot += data[j] * x[indices[j]]
        
        # Store to shared memory
        shared_results[tid] = dot
        cuda.syncthreads()
        
        # Tree reduction in shared memory
        s = block_size // 2
        while s > 0:
            if tid < s:
                shared_results[tid] += shared_results[tid + s]
            cuda.syncthreads()
            s //= 2
        
        # Write result
        if tid == 0:
            y[row] = shared_results[0]
        cuda.syncthreads()


# =============================================================================
# Interface Functions
# =============================================================================

def spmv_gpu_scalar_shared(csr_matrix, x, y=None):
    """SpMV using scalar kernel with shared memory caching."""
    num_rows, num_cols = csr_matrix.shape
    
    d_data, d_indices, d_indptr = csr_matrix.to_gpu()
    
    if isinstance(x, np.ndarray):
        d_x = cuda.to_device(x.astype(np.float32))
    else:
        d_x = x
    
    if y is None:
        d_y = cuda.device_array(num_rows, dtype=np.float32)
    elif isinstance(y, np.ndarray):
        d_y = cuda.to_device(y)
    else:
        d_y = y
    
    threads_per_block = 256
    blocks = (num_rows + threads_per_block - 1) // threads_per_block
    
    spmv_scalar_shared_kernel[blocks, threads_per_block](
        d_data, d_indices, d_indptr, d_x, d_y, num_rows, num_cols
    )
    
    return d_y


def spmv_gpu_vector_optimized(csr_matrix, x, y=None):
    """SpMV using optimized vector kernel."""
    num_rows = csr_matrix.shape[0]
    
    d_data, d_indices, d_indptr = csr_matrix.to_gpu()
    
    if isinstance(x, np.ndarray):
        d_x = cuda.to_device(x.astype(np.float32))
    else:
        d_x = x
    
    if y is None:
        d_y = cuda.device_array(num_rows, dtype=np.float32)
    elif isinstance(y, np.ndarray):
        d_y = cuda.to_device(y)
    else:
        d_y = y
    
    threads_per_block = 256
    num_warps = num_rows
    blocks = (num_warps * 32 + threads_per_block - 1) // threads_per_block
    
    spmv_vector_optimized_kernel[blocks, threads_per_block](
        d_data, d_indices, d_indptr, d_x, d_y, num_rows
    )
    
    return d_y


def spmv_gpu_binned(csr_matrix, x, y=None):
    """
    SpMV using row binning for load balancing.
    
    Processes rows in bins based on their length for optimal performance.
    """
    num_rows = csr_matrix.shape[0]
    
    d_data, d_indices, d_indptr = csr_matrix.to_gpu()
    bins = csr_matrix.get_row_bins_gpu()
    
    if isinstance(x, np.ndarray):
        d_x = cuda.to_device(x.astype(np.float32))
    else:
        d_x = x
    
    if y is None:
        d_y = cuda.device_array(num_rows, dtype=np.float32)
        # Initialize to zero for atomic adds
        d_y_host = np.zeros(num_rows, dtype=np.float32)
        d_y = cuda.to_device(d_y_host)
    elif isinstance(y, np.ndarray):
        d_y = cuda.to_device(np.zeros_like(y))
    else:
        d_y = y
    
    # Process short rows (scalar)
    if bins['short'] is not None and len(csr_matrix._row_bins['short']) > 0:
        n_short = len(csr_matrix._row_bins['short'])
        threads = 256
        blocks = (n_short + threads - 1) // threads
        spmv_binned_short_kernel[blocks, threads](
            d_data, d_indices, d_indptr, d_x, d_y, bins['short'], n_short
        )
    
    # Process medium rows (1 warp per row)
    if bins['medium'] is not None and len(csr_matrix._row_bins['medium']) > 0:
        n_medium = len(csr_matrix._row_bins['medium'])
        threads = 256  # 8 warps per block
        blocks = (n_medium + 7) // 8  # 8 rows per block
        spmv_binned_medium_kernel[blocks, threads](
            d_data, d_indices, d_indptr, d_x, d_y, bins['medium'], n_medium
        )
    
    # Process long rows (2 warps per row)
    if bins['long'] is not None and len(csr_matrix._row_bins['long']) > 0:
        n_long = len(csr_matrix._row_bins['long'])
        threads = 256  # 4 rows per block (2 warps each)
        blocks = (n_long + 3) // 4
        spmv_binned_long_kernel[blocks, threads](
            d_data, d_indices, d_indptr, d_x, d_y, bins['long'], n_long
        )
    
    # Very long rows use the same kernel as long (could be optimized further)
    if bins['very_long'] is not None and len(csr_matrix._row_bins['very_long']) > 0:
        n_vlong = len(csr_matrix._row_bins['very_long'])
        threads = 256
        blocks = (n_vlong + 3) // 4
        spmv_binned_long_kernel[blocks, threads](
            d_data, d_indices, d_indptr, d_x, d_y, bins['very_long'], n_vlong
        )
    
    return d_y


def spmv_gpu_cache_optimized(csr_matrix, x, y=None):
    """SpMV using cache-optimized kernel."""
    num_rows = csr_matrix.shape[0]
    
    d_data, d_indices, d_indptr = csr_matrix.to_gpu()
    
    if isinstance(x, np.ndarray):
        d_x = cuda.to_device(x.astype(np.float32))
    else:
        d_x = x
    
    if y is None:
        d_y = cuda.device_array(num_rows, dtype=np.float32)
    elif isinstance(y, np.ndarray):
        d_y = cuda.to_device(y)
    else:
        d_y = y
    
    threads_per_block = 256
    rows_per_block = 8
    blocks = (num_rows + rows_per_block - 1) // rows_per_block
    
    spmv_cache_optimized_kernel[blocks, threads_per_block](
        d_data, d_indices, d_indptr, d_x, d_y, num_rows
    )
    
    return d_y


def spmv_cpu(csr_matrix, x):
    """CPU reference implementation."""
    y = np.zeros(csr_matrix.shape[0], dtype=np.float32)
    
    for i in range(csr_matrix.shape[0]):
        start = csr_matrix.indptr[i]
        end = csr_matrix.indptr[i + 1]
        for j in range(start, end):
            y[i] += csr_matrix.data[j] * x[csr_matrix.indices[j]]
    
    return y


def verify_spmv(csr_matrix, x, gpu_result, rtol=1e-4, atol=1e-5):
    """Verify GPU result against CPU reference."""
    cpu_result = spmv_cpu(csr_matrix, x)
    
    if hasattr(gpu_result, 'copy_to_host'):
        gpu_result = gpu_result.copy_to_host()
    
    return np.allclose(cpu_result, gpu_result, rtol=rtol, atol=atol)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_optimized_kernels():
    """Compare original vs optimized kernels."""
    import time
    
    print("=" * 70)
    print("OPTIMIZED SpMV KERNEL BENCHMARK")
    print("=" * 70)
    
    # Test matrix
    N = 50000
    density = 0.001
    
    print(f"\nMatrix: {N}x{N}, density={density:.2%}")
    A = CSRMatrix.random(N, N, density=density, seed=42)
    print(f"Created: {A}")
    
    # Compute row bins
    bins = A.compute_row_bins()
    print(f"\nRow distribution:")
    print(f"  Short (<=32 nnz):    {len(bins['short']):,} rows")
    print(f"  Medium (33-128):     {len(bins['medium']):,} rows")
    print(f"  Long (129-512):      {len(bins['long']):,} rows")
    print(f"  Very long (>512):    {len(bins['very_long']):,} rows")
    
    x = np.random.randn(N).astype(np.float32)
    
    # Warmup and timing function
    def time_kernel(name, func, *args, iterations=10):
        # Warmup
        for _ in range(3):
            result = func(*args)
            cuda.synchronize()
        
        # Create events
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
        
        # Verify
        is_correct = verify_spmv(A, x, result)
        
        # Calculate bandwidth
        total_bytes = (A.nnz * 4 + A.nnz * 4 + (N + 1) * 4 + N * 4 + N * 4)
        bandwidth = (total_bytes / 1e9) / (avg_time / 1000)
        
        print(f"\n{name}:")
        print(f"  Time: {avg_time:.3f} ± {std_time:.3f} ms")
        print(f"  Bandwidth: {bandwidth:.1f} GB/s")
        print(f"  Correct: {is_correct}")
        
        return avg_time, bandwidth, is_correct
    
    print("\n" + "-" * 70)
    print("KERNEL BENCHMARKS")
    print("-" * 70)
    
    results = {}
    
    # Original vector kernel (baseline)
    results['Vector (original)'] = time_kernel(
        "Vector Kernel (Original)", 
        spmv_gpu_vector_optimized, A, x
    )
    
    # Shared memory kernel
    results['Scalar + Shared'] = time_kernel(
        "Scalar + Shared Memory",
        spmv_gpu_scalar_shared, A, x
    )
    
    # Binned kernel
    results['Binned'] = time_kernel(
        "Row-Binned Kernel",
        spmv_gpu_binned, A, x
    )
    
    # Cache optimized
    results['Cache Optimized'] = time_kernel(
        "Cache-Optimized Kernel",
        spmv_gpu_cache_optimized, A, x
    )
    
    # cuSPARSE comparison
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsparse
        
        A_cupy = cpsparse.csr_matrix(
            (cp.array(A.data), cp.array(A.indices), cp.array(A.indptr)),
            shape=A.shape
        )
        x_cupy = cp.array(x)
        
        # Warmup
        for _ in range(3):
            _ = A_cupy @ x_cupy
            cp.cuda.Stream.null.synchronize()
        
        times = []
        for _ in range(10):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            _ = A_cupy @ x_cupy
            end.record()
            end.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, end))
        
        cusparse_time = np.mean(times)
        total_bytes = (A.nnz * 4 + A.nnz * 4 + (N + 1) * 4 + N * 4 + N * 4)
        cusparse_bw = (total_bytes / 1e9) / (cusparse_time / 1000)
        
        print(f"\ncuSPARSE Reference:")
        print(f"  Time: {cusparse_time:.3f} ms")
        print(f"  Bandwidth: {cusparse_bw:.1f} GB/s")
        
        results['cuSPARSE'] = (cusparse_time, cusparse_bw, True)
        
    except ImportError:
        print("\n[SKIP] cuSPARSE (CuPy not installed)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Kernel':<25} {'Time (ms)':<12} {'GB/s':<12} {'vs cuSPARSE':<12}")
    print("-" * 70)
    
    cusparse_time = results.get('cuSPARSE', (1, 1, True))[0]
    for name, (time_ms, bw, correct) in results.items():
        ratio = time_ms / cusparse_time if 'cuSPARSE' in results else 1.0
        status = "OK" if correct else "FAILED"
        print(f"{name:<25} {time_ms:<12.3f} {bw:<12.1f} {ratio:<12.2f}× {status}")
    
    # Cleanup
    A.free_gpu()
    
    return results


if __name__ == "__main__":
    benchmark_optimized_kernels()