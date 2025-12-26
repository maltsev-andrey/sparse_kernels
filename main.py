#!/usr/bin/env python3
"""
Main entry point for Sparse Matrix Operations project.

Runs benchmarks and generates visualizations for portfolio presentation.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numba import cuda

from formats.csr import CSRMatrix, spmv_gpu_scalar, spmv_gpu_vector, spmv_gpu_adaptive
from benchmarks.spmv_benchmark import SpMVBenchmark, run_scaling_benchmark
from visualization.plots import (
    plot_sparsity_pattern,
    plot_row_length_distribution,
    plot_performance_comparison,
    plot_performance_scaling,
    create_summary_figure,
    plot_roofline
)


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_gpu_info():
    """Print GPU information."""
    print_header("GPU INFORMATION")
    device = cuda.get_current_device()
    print(f"Device: {device.name}")
    print(f"Compute Capability: {device.compute_capability}")
    
    # Memory info
    free_mem, total_mem = cuda.current_context().get_memory_info()
    print(f"Memory: {free_mem/1e9:.1f} GB free / {total_mem/1e9:.1f} GB total")


def run_single_benchmark(N=50000, density=0.001):
    """Run benchmark on a single matrix size."""
    print_header(f"SINGLE MATRIX BENCHMARK ({N}×{N}, {density:.2%} density)")
    
    A = CSRMatrix.random(N, N, density=density, seed=42)
    print(f"\nMatrix created: {A}")
    
    benchmark = SpMVBenchmark()
    results = benchmark.benchmark_spmv(A, include_cusparse=True)
    
    return A, results


def generate_visualizations(csr_matrix, results, output_dir):
    """Generate all visualizations."""
    print_header("GENERATING VISUALIZATIONS")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Sparsity pattern
    print("\n1. Sparsity pattern...")
    plot_sparsity_pattern(csr_matrix, output_path=output_dir / "sparsity_pattern.png")
    
    # 2. Row length distribution
    print("2. Row length distribution...")
    plot_row_length_distribution(csr_matrix, output_path=output_dir / "row_distribution.png")
    
    # 3. Performance comparison bar chart
    print("3. Performance comparison...")
    kernel_names = list(results['kernels'].keys())
    bandwidths = [results['kernels'][k]['bandwidth_gb_s'] for k in kernel_names]
    plot_performance_comparison(
        kernel_names, bandwidths,
        metric='Bandwidth (GB/s)',
        title=f"SpMV Performance ({csr_matrix.shape[0]:,}×{csr_matrix.shape[1]:,})",
        output_path=output_dir / "performance_comparison.png"
    )
    
    # 4. Summary figure
    print("4. Summary figure...")
    benchmark_data = {
        'kernels': kernel_names,
        'bandwidth': bandwidths,
        'speedup': [results['kernels'][k]['speedup'] for k in kernel_names]
    }
    create_summary_figure(csr_matrix, benchmark_data, output_path=output_dir / "summary.png")
    
    # 5. Roofline analysis
    print("5. Roofline analysis...")
    roofline_data = []
    for name, data in results['kernels'].items():
        if name == 'CPU':
            continue
        roofline_data.append({
            'name': name,
            'flops': 2 * csr_matrix.nnz,  # 2 FLOPs per non-zero
            'bytes': (csr_matrix.nnz * 4 + csr_matrix.nnz * 4 + 
                     (csr_matrix.shape[0] + 1) * 4 +
                     csr_matrix.shape[1] * 4 + csr_matrix.shape[0] * 4),
            'time_s': data['time_ms'] / 1000
        })
    
    if roofline_data:
        plot_roofline(roofline_data, output_path=output_dir / "roofline.png")
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


def run_scaling_study(output_dir):
    """Run benchmark across multiple matrix sizes."""
    print_header("SCALING STUDY")
    
    sizes = [5000, 10000, 25000, 50000, 75000, 100000]
    density = 0.001
    
    scaling_results = run_scaling_benchmark(
        sizes=sizes,
        density=density,
        output_dir=output_dir
    )
    
    # Generate scaling plot
    output_dir = Path(output_dir)
    
    # Filter out CPU and empty results
    plot_data = {k: v for k, v in scaling_results.items() 
                 if k not in ['sizes', 'density', 'CPU'] and any(v)}
    
    if plot_data:
        plot_performance_scaling(
            scaling_results['sizes'],
            plot_data,
            title=f"SpMV Performance Scaling ({density:.2%} density)",
            output_path=output_dir / "scaling.png"
        )
    
    return scaling_results


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print(" GPU-ACCELERATED SPARSE MATRIX OPERATIONS")
    print(" Portfolio Project - Andrey Maltsev")
    print("=" * 70)
    
    # GPU info
    print_gpu_info()
    
    # Output directory
    output_dir = PROJECT_ROOT / "docs" / "figures"
    
    # Run single benchmark
    A, results = run_single_benchmark(N=50000, density=0.001)
    
    # Generate visualizations
    generate_visualizations(A, results, output_dir)
    
    # Run scaling study (optional, takes longer)
    run_scaling = input("\nRun scaling study? (y/N): ").lower().strip() == 'y'
    if run_scaling:
        run_scaling_study(output_dir)
    
    print_header("COMPLETE")
    print(f"\nResults and figures saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review generated figures in docs/figures/")
    print("  2. Run: python formats/csr.py  # Quick kernel test")
    print("  3. Run: python benchmarks/spmv_benchmark.py  # Full benchmark")
    print("\nProject ready for GitHub! ")


if __name__ == "__main__":
    main()
