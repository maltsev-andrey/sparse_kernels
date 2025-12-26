"""
Benchmarking Module for Sparse Matrix Operations

Provides accurate timing, bandwidth calculations, and comparison against cuSPARSE.
"""

import numpy as np
from numba import cuda
import time
from pathlib import Path
import sys

# Add parent to path imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from formats.csr import (
    CSRMatrix,spmv_gpu_scalar, spmv_gpu_vector,
    spmv_gpu_adaptive, spmv_cpu, verify_spmv
)

class SpMVBenchmark:
    """Benchmark suite for SpMV operations."""

    # Tesla P100 specifications
    P100_PEAK_BANDWIDTH = 732   # GB/s (HBM2)
    P100_PEAK_FLOPS = 4700              # GFLOPS (FP32)

    def __init__(self, warmup_iterations=3, benchmark_iterations=10):
        """
        Initialize benchmark suite.
        
        Args:
            warmup_iterations: Number of warmup runs (not timed)
            benchmark_iterations: Number of timed runs for averaging
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = []

    def _time_gpu_kernel(self, kernel_func, *args, **kwargs):
        """
        Accurately time a GPU kernel using CUDA events.
        
        Returns:
            Average execution time in milliseconds
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = kernel_func(*args, **kwargs)
            cuda.synchronize()

        # Create CUDA events for timing
        start_event = cuda.event()
        end_event = cuda.event()

        times = []
        for _ in range(self.benchmark_iterations):
            start_event.record()
            _ = kernel_func(*args, **kwargs)
            end_event.record()
            end_event.synchronize()

            elapsed_ms = cuda.event_elapsed_time(start_event, end_event)
            times.append(elapsed_ms)

        return np.mean(times), np.std(times)

    def _time_cpu(self, func, *args):
        """Time a CPU function"""
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = func(*args)

        times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = func(*args)
            end = time.perf_counter()
            times.append((end - start) * 1000) # Convert to ms
        
        return np.mean(times), np.std(times) 

    def calculate_bandwidth(self, csr_matrix, time_ms):
        """
        Calculate effective memory bandwidth for SpMV.
        
        SpMV memory traffic (read-only, minimal write):
        - Matrix values: nnz * 4 bytes (float32)
        - Column indices: nnz * 4 bytes (int32)
        - Row pointers: (rows + 1) * 4 bytes (int32)
        - Input vector x: cols * 4 bytes (float32) - assume full read
        - Output vector y: rows * 4 bytes (float32)
        
        Args:
            csr_matrix: CSRMatrix instance
            time_ms: Execution time in milliseconds
        
        Returns:
            Effective bandwidth in GB/s
        """
        bytes_data = csr_matrix.nnz * 4
        bytes_indices = csr_matrix.nnz * 4
        bytes_indptr =(csr_matrix.shape[0] + 1) * 4
        bytes_x = csr_matrix.shape[1] * 4
        bytes_y = csr_matrix.shape[0] * 4

        total_bytes = bytes_data + bytes_indices + bytes_indptr + bytes_x + bytes_y

        time_s = time_ms / 1000
        bandwidth_gb_s = (total_bytes / 1e9) / time_s

        return bandwidth_gb_s

    def calculate_flops(self, csr_matrix, time_ms):
        """
        Calculate GFLOPS for SpMV.
        
        SpMV: 2 FLOPs per non-zero (multiply + add)
        """
        flops = 2 * csr_matrix.nnz
        time_s = time_ms / 1000
        gflops = (flops / 1e9) / time_s
        
        return gflops

    def benchmark_spmv(self, csr_matrix, include_cusparse=True):
        """
        Run complete SpMV benchmark suite.
        
        Args:
            csr_matrix: CSRMatrix instance
            include_cusparse: Whether to benchmark cuSPARSE (requires CuPy)
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"\nBenchmarking SpMV on {csr_matrix}")
        print("=" * 60)

        # Create test vector
        x = np.random.randn(csr_matrix.shape[1]).astype(np.float32)

        results = {
            'matrix_shape': csr_matrix.shape,
            'nnz': csr_matrix.nnz,
            'density': csr_matrix.density,
            'kernels': {}
        }

        # 1. CPU Reference
        print("\n[CPU] NumPy reference...")
        cpu_time, cpu_std = self._time_cpu(spmv_cpu, csr_matrix, x)
        results['kernels']['CPU'] = {
            'time_ms': cpu_time,
            'time_std': cpu_std,
            'bandwidth_gb_s': self.calculate_bandwidth(csr_matrix, cpu_time),
            'gflops': self.calculate_flops(csr_matrix, cpu_time),
            'speedup': 1.0
        }
        print(f"    Time: {cpu_time:.3f} ± {cpu_std:.3f} ms")
        
        # 2. GPU Scalar Kernel
        print("\n[GPU] Scalar kernel (one thread per row)...")
        scalar_time, scalar_std = self._time_gpu_kernel(spmv_gpu_scalar, csr_matrix, x)
        scalar_bw = self.calculate_bandwidth(csr_matrix, scalar_time)
        results['kernels']['Scalar'] = {
            'time_ms': scalar_time,
            'time_std': scalar_std,
            'bandwidth_gb_s': scalar_bw,
            'gflops': self.calculate_flops(csr_matrix, scalar_time),
            'speedup': cpu_time / scalar_time,
            'peak_bw_pct': scalar_bw / self.P100_PEAK_BANDWIDTH * 100
        }
        print(f"    Time: {scalar_time:.3f} ± {scalar_std:.3f} ms")
        print(f"    Bandwidth: {scalar_bw:.1f} GB/s ({scalar_bw/self.P100_PEAK_BANDWIDTH*100:.1f}% peak)")
        print(f"    Speedup vs CPU: {cpu_time/scalar_time:.1f}×")
        
        # Verify correctness
        d_y = spmv_gpu_scalar(csr_matrix, x)
        cuda.synchronize()
        assert verify_spmv(csr_matrix, x, d_y), "Scalar kernel verification failed!"
        print("    OK -> Verified correct")
        
        # 3. GPU Vector Kernel
        print("\n[GPU] Vector kernel (one warp per row)...")
        vector_time, vector_std = self._time_gpu_kernel(spmv_gpu_vector, csr_matrix, x)
        vector_bw = self.calculate_bandwidth(csr_matrix, vector_time)
        results['kernels']['Vector'] = {
            'time_ms': vector_time,
            'time_std': vector_std,
            'bandwidth_gb_s': vector_bw,
            'gflops': self.calculate_flops(csr_matrix, vector_time),
            'speedup': cpu_time / vector_time,
            'peak_bw_pct': vector_bw / self.P100_PEAK_BANDWIDTH * 100
        }
        print(f"    Time: {vector_time:.3f} ± {vector_std:.3f} ms")
        print(f"    Bandwidth: {vector_bw:.1f} GB/s ({vector_bw/self.P100_PEAK_BANDWIDTH*100:.1f}% peak)")
        print(f"    Speedup vs CPU: {cpu_time/vector_time:.1f}×")
        
        d_y = spmv_gpu_vector(csr_matrix, x)
        cuda.synchronize()
        assert verify_spmv(csr_matrix, x, d_y), "Vector kernel verification failed!"
        print("    OK -> Verified correct")
        
        # 4. GPU Adaptive Kernel
        print("\n[GPU] Adaptive kernel...")
        adaptive_time, adaptive_std = self._time_gpu_kernel(spmv_gpu_adaptive, csr_matrix, x)
        adaptive_bw = self.calculate_bandwidth(csr_matrix, adaptive_time)
        results['kernels']['Adaptive'] = {
            'time_ms': adaptive_time,
            'time_std': adaptive_std,
            'bandwidth_gb_s': adaptive_bw,
            'gflops': self.calculate_flops(csr_matrix, adaptive_time),
            'speedup': cpu_time / adaptive_time,
            'peak_bw_pct': adaptive_bw / self.P100_PEAK_BANDWIDTH * 100
        }
        print(f"    Time: {adaptive_time:.3f} ± {adaptive_std:.3f} ms")
        print(f"    Bandwidth: {adaptive_bw:.1f} GB/s ({adaptive_bw/self.P100_PEAK_BANDWIDTH*100:.1f}% peak)")
        print(f"    Speedup vs CPU: {cpu_time/adaptive_time:.1f}×")
        
        d_y = spmv_gpu_adaptive(csr_matrix, x)
        cuda.synchronize()
        assert verify_spmv(csr_matrix, x, d_y), "Adaptive kernel verification failed!"
        print("    OK -> Verified correct")

        # 5. ciSPARSE (via CuPy)
        if include_cusparse:
            try:
                import cupy as cp
                import  cupyx.scipy.sparse as cpsparse

                print("\n[GPU] cuSPARSE reference...")

                # Convert to CuPy sparce matrix
                A_cupy = cpsparse.csr_matrix(
                    (cp.array(csr_matrix.data),
                      cp.array(csr_matrix.indices),
                      cp.array(csr_matrix.indptr)),
                    shape=csr_matrix.shape
                )
                x_cupy = cp.array(x)

                # Warmup
                for  _ in range(self.warmup_iterations):
                    _ = A_cupy @ x_cupy
                    cp.cuda.Stream.null.synchronize()

                # Benchmark
                times = []
                for _ in range(self.benchmark_iterations):
                    start = cp.cuda.Event()
                    end = cp.cuda.Event()

                    start.record()
                    _ = A_cupy @ x_cupy
                    end.record()
                    end.synchronize()

                    times.append(cp.cuda.get_elapsed_time(start, end))

                cusparse_time = np.mean(times)
                cusparse_std = np.std(times)
                cusparse_bw = self.calculate_bandwidth(csr_matrix, cusparse_time)

                results['kernels']['cuSPARSE'] = {
                    'time_ms': cusparse_time,
                    'time_std': cusparse_std,
                    'bandwidth_gb_s': cusparse_bw,
                    'gflops': self.calculate_flops(csr_matrix, cusparse_time),
                    'speedup': cpu_time / cusparse_time,
                    'peak_bw_pct': cusparse_bw / self.P100_PEAK_BANDWIDTH * 100
                }
                print(f"    Time: {cusparse_time:.3f} ± {cusparse_std:.3f} ms")
                print(f"    Bandwidth: {cusparse_bw:.1f} GB/s ({cusparse_bw/self.P100_PEAK_BANDWIDTH*100:.1f}% peak)")
                print(f"    Speedup vs CPU: {cpu_time/cusparse_time:.1f}×")
                
            except ImportError:
                print("\n[SKIP] cuSPARSE benchmark (CuPy not installed)")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Kernel':<12} {'Time (ms)':<12} {'GB/s':<10} {'% Peak':<10} {'Speedup':<10}")
        print("-" * 60)
        
        for name, data in results['kernels'].items():
            peak_pct = data.get('peak_bw_pct', 0)
            print(f"{name:<12} {data['time_ms']:<12.3f} {data['bandwidth_gb_s']:<10.1f} "
                  f"{peak_pct:<10.1f} {data['speedup']:<10.1f}×")
        
        self.results.append(results)
        return results

def run_scaling_benchmark(sizes=None, density=0.01, output_dir=None):
    """
    Run SpMV benchmark across multiple matrix sizes.
    
    Args:
        sizes: List of N values for NxN matrices
        density: Sparsity density
        output_dir: Directory to save results
    
    Returns:
        Dictionary with scaling results
    """
    if sizes is None:
        sizes = [1000, 5000, 10000, 25000, 50000, 100000]

    benchmark = SpMVBenchmark()

    scaling_results = {
        'sizes': sizes,
        'density': density,
        'CPU': [], 'Scalar': [], 'Vector': [], 'Adaptive': [], 'cuSPARSE': []
    }

    print("\n" + "=" * 70)
    print("SCALING BENCHMARK")
    print(f"Matrix sizes: {sizes}")
    print(f"Density: {density:.2%}")
    print("=" * 70)

    for N in sizes:
        print(f"\n{'=' * 70}")
        print(f"Matrix size: {N} x {N}")

        A = CSRMatrix.random(N, N, density=density, seed=42)
        results = benchmark.benchmark_spmv(A)

        for kernel_name in scaling_results.keys():
            if kernel_name in ['sizes', 'density']:
                continue
            if kernel_name in results['kernels']:
                scaling_results[kernel_name].append(
                    results['kernels'][kernel_name]['bandwidth_gb_s']
                )
            else:
                scaling_results[kernel_name].append(0)

        # Free GPU memory
        A.free_gpu()
        cuda.current_context().memory_manager.deallocationa.clear()

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_dir / 'scaling_results.json', 'w') as f:
            json.dump(scaling_results, f, indent=2)
        print(f"\nResults saved to {output_dir / 'scaling_results.json'}")
    return scaling_results

if __name__=="__main__":
    # Quick benchmark test
    print(f"SpMV Benchmark Test")
    print("=" * 60)

    benchmark = SpMVBenchmark()

    #Single matrix Benchmark
    A = CSRMatrix.random(50000, 50000, density=0.001, seed=42)
    results = benchmark.benchmark_spmv(A)

    print("\n\nBenchmark complete!")    