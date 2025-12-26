"""
CSR (Compressed Sparse Row) Format Implementation with GPU SpMV Kernels

CSR stores sparse matrices using three arrays:
- data: non-zero values (nnz elements)
- indices: column indices for each non-zero (nnz elements)  
- indptr: row pointers, indptr[i] to indptr[i+1] spans row i (n+1 elements)

This module implements multiple SpMV kernel variants for performance comparison.
"""
import numpy as np
from numba import cuda, float32
import math

class CSRMatrix:
    """CSR sparse matrix with CPU and GPU storage."""
    def __init__(self, data, indices, indptr, shape):
        """
        Initialize CSR matrix.
        
        Args:
            data: Non-zero values array
            indices: Column indices array
            indptr: Row pointer array
            shape: (rows, cols) tuple
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.indptr = np.asarray(indptr, dtype = np.int32)
        self.shape = shape
        self.nnz = len(self.data)

        # GPU arrays (allocated on first use)
        self._d_data = None
        self._d_indices = None
        self._d_indptr = None

    @classmethod
    def from_dense(cls, dense_matrix):
        """Create CSR matrix from dense numpy array."""
        dense = np. asarray(dense_matrix, dtype=np.float32)
        rows, cols = dense.shape

        data = []
        indices = []
        indptr = [0]

        for i in range(rows):
            for j in range(cols):
                if dense[i, j] != 0:
                    data.append(dense[i, j])
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    @classmethod
    def from_scipy(cls, scipy_csr):
        """Create from scipy.sparse.csr_matrix."""
        return cls(
            scipy_csr.data,
            scipy_csr.indices,
            scipy_csr.indptr,
            scipy_csr.shape
        )
        
    @classmethod
    def random(cls, rows, cols, density=0.001, seed=None):
        """
        Generate random sparse matrix.
        
        Args:
            rows, cols: Matrix dimensions
            density: Fraction of non-zeros (0.01 = 1%)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        nnz_per_row = int(cols * density)
        nnz_per_row = max(1, nnz_per_row) # at least 1 per row

        data = []
        indices = []
        indptr = [0]

        for i in range(rows):
            # Random column indices for this row
            row_indices = np.random.choice(cols, size=min(nnz_per_row, cols), replace=False)
            row_indices = np.sort(row_indices)

            # Random values
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

    def free_gpu(self):
        """Free GPU memory."""
        self._d_data = None
        self._d_indices = None
        self._d_indptr = None

    @property
    def density(self):
        """Fraction of non-zero elements."""
        return self.nnz / (self.shape[0] * self.shape[1])

    @property
    def bytes_csr(self):
        """Memory footprint in CSR format."""
        return (self.data.nbytes + self.indices.nbytes + self.indptr.nbytes)

    @property
    def bytes_dense(self):
        """Memory footprint if stored dense."""
        return self.shape[0] * self.shape[1] * 4 # float32

    def __repr__(self):
        return (
                f"CSRMatrix(shape={self.shape}, nnz={self.nnz}, "
                f"density={self.density:.4f}, size={self.bytes_csr/1024:.1f}KB)"
        )
                
# =============================================================================
# SpMV Kernels
# =============================================================================

@cuda.jit
def spmv_csr_scalar_kernel(data, indices,indptr, x, y, num_rows):
    """
    Scalar CSR SpMV: one thread per row.
    Simple but inefficient for rows with many non-zeros.
    Good baseline for comparison.
    """
    row = cuda.grid(1)

    if row < num_rows:
        row_start = indptr[row]
        row_end = indptr[row + 1]

        dot = float32(0.0) # excplicity use float32
        for j in range(row_start, row_end):
            dot += data[j] *  x[indices[j]]

        y[row] = dot

@cuda.jit
def spmv_csr_vector_kernel(data, indices, indptr, x, y, num_rows):
    """
    Vector CSR SpMV: one warp (32 threads) per row.
    
    Better for matrices with many non-zeros per row.
    Uses warp shuffle for reduction.
    """
    # Thread position within warp
    lane = cuda.threadIdx.x % 32
    # Warp index (which row this warp processes)
    warp_id = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // 32

    if warp_id < num_rows:
        row_start = indptr[warp_id]
        row_end = indptr[warp_id + 1]

        # Each thread in warp handles subset of row elements
        dot = float32(0.0)
        for j in range(row_start + lane, row_end, 32):
            dot += data[j] * x[indices[j]]

        # Warp reduction usinf shuffle
        for offset in [16, 8, 4, 2, 1]:
            dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, offset)

        # Lane 0 writes result
        if lane == 0:
            y[warp_id] = dot

@cuda.jit
def spmv_csr_adaptive_kernel(data, indices, indptr, x, y, num_rows, threshold):
    """
    Adaptive CSR SpMV: switches between scalar and vector based on row length.
    
    Uses scalar approach for short rows, vector for long rows.
    Threshold typically set to 32 (warp size).
    """
    tid = cuda.grid(1)
    lane = cuda.threadIdx.x % 32
    warp_id = tid // 32

    if warp_id < num_rows:
        row_start = indptr[warp_id]
        row_end = indptr[warp_id + 1]
        row_len = row_end - row_start

        if row_len <= threshold:
            # Scalar: only lane 0 does work
            if lane == 0:
                dot = float32(0.0)
                for j in range(row_start, row_end):
                    dot += data[j] * x[indices[j]]
                y[warp_id] = dot
        else:
            # Vector: all lanes participate
            dot = float32(0.0)
            for j in range(row_start + lane, row_end, 32):
                dot += data[j] * x[indices[j]]

                # Warp reduction
            for offset in [16, 8, 4, 2, 1]:
                dot += cuda.shfl_down_sync(0xFFFFFFFF, dot, offset)
            
            if lane == 0:
                y[warp_id] = dot

# =============================================================================
# SpMV Interface Functions
# =============================================================================

def spmv_gpu_scalar(csr_matrix, x, y=None):
    """
    GPU SpMV using scalar kernel (one thread per row).
    
    Args:
        csr_matrix: CSRMatrix instance
        x: Input vector (numpy array or device array)
        y: Output vector (optional, allocated if None)
    
    Returns:
        y: Result vector on GPU
    """
    num_rows = csr_matrix.shape[0]

    # Transfer to GPU if needed
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

    # Launch kernel
    threads_per_block = 256
    blocks = (num_rows + threads_per_block - 1) // threads_per_block

    spmv_csr_scalar_kernel[blocks,threads_per_block] (
        d_data, d_indices, d_indptr, d_x, d_y, num_rows
    )

    return d_y

def spmv_gpu_vector(csr_matrix, x, y=None):
    """
    GPU SpMV using vector kernel (one warp per row).
    
    Args:
        csr_matrix: CSRMatrix instance
        x: Input vector (numpy array or device array)
        y: Output vector (optional, allocated if None)
    
    Returns:
        y: Result vector on GPU
    """
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

    # One warp (32 threads) per row
    threads_per_block = 256  # 8 warps per block
    num_warps = num_rows
    blocks = (num_warps * 32 + threads_per_block - 1) // threads_per_block
    
    spmv_csr_vector_kernel[blocks, threads_per_block](
        d_data, d_indices, d_indptr, d_x, d_y, num_rows
    )
    
    return d_y

def spmv_gpu_adaptive(csr_matrix, x, y=None, threshold=32):
    """
    GPU SpMV using adaptive kernel.
    
    Args:
        csr_matrix: CSRMatrix instance
        x: Input vector
        y: Output vector (optional)
        threshold: Row length threshold for scalar vs vector mode
    
    Returns:
        y: Result vector on GPU
    """
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

    spmv_csr_adaptive_kernel[blocks, threads_per_block] (
        d_data, d_indices, d_indptr, d_x, d_y, num_rows, threshold
    )

    return d_y

def spmv_cpu(csr_matrix, x):
    """CPU reference implementation using NumPy."""
    y = np.zeros(csr_matrix.shape[0], dtype=np.float32)
    
    for i in range(csr_matrix.shape[0]):
        start = csr_matrix.indptr[i]
        end = csr_matrix.indptr[i + 1]
        
        for j in range(start, end):
            y[i] += csr_matrix.data[j] * x[csr_matrix.indices[j]]
    
    return y

# =============================================================================
# Verification and Testing
# =============================================================================

def verify_spmv(csr_matrix, x, gpu_result, rtol=1e-4, atol=1e-5):
    """Verify GPU result against CPU reference."""
    cpu_result = spmv_cpu(csr_matrix, x)

    if hasattr(gpu_result, 'copy_to_host'):
        gpu_result = gpu_result.copy_to_host()

    return np.allclose(cpu_result, gpu_result, rtol=rtol, atol=atol)

if __name__ == "__main__":
    # Quick test
    print("CSR SpMV Test")
    print("=" * 50)
    
    # Create test matrix
    N = 10000
    density = 0.01
    
    print(f"Creating {N}x{N} matrix with {density:.1%} density...")
    A = CSRMatrix.random(N, N, density=density, seed=42)
    print(f"Matrix: {A}")
    
    # Create test vector
    x = np.random.randn(N).astype(np.float32)
    
    # Run kernels
    print("\nRunning SpMV kernels...")
    
    # Warmup
    _ = spmv_gpu_scalar(A, x)
    cuda.synchronize()
    
    # Scalar kernel
    d_y_scalar = spmv_gpu_scalar(A, x)
    cuda.synchronize()
    
    # Vector kernel
    d_y_vector = spmv_gpu_vector(A, x)
    cuda.synchronize()
    
    # Adaptive kernel
    d_y_adaptive = spmv_gpu_adaptive(A, x)
    cuda.synchronize()
    
    # ============ ADD DEBUG CODE HERE ============
    y_scalar = d_y_scalar.copy_to_host()
    y_cpu = spmv_cpu(A, x)

    print(f"\nDebug:")
    print(f"  CPU result range: [{y_cpu.min():.4f}, {y_cpu.max():.4f}]")
    print(f"  GPU result range: [{y_scalar.min():.4f}, {y_scalar.max():.4f}]")
    print(f"  First 5 CPU: {y_cpu[:5]}")
    print(f"  First 5 GPU: {y_scalar[:5]}")
    print(f"  Max absolute diff: {np.abs(y_cpu - y_scalar).max():.6f}")
    # ============ END DEBUG CODE ============
    
    # Verify
    ok_scalar = verify_spmv(A, x, d_y_scalar)
    ok_vector = verify_spmv(A, x, d_y_vector)
    ok_adapt  = verify_spmv(A, x, d_y_adaptive)

    print("\nVerification:")
    print(f"  Scalar kernel correct: {ok_scalar}")
    print(f"  Vector kernel correct: {ok_vector}")
    print(f"  Adaptive kernel correct: {ok_adapt}")

    if ok_scalar and ok_vector and ok_adapt:
        print("\nOK -> All tests passed!")
    else:
        print("\nERROR -> Some tests failed!")     
