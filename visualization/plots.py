"""
Visualization Module for Sparse Matrix Operations

Generates publication-quality visualizations:
- Sparsity pattern plots
- Performance scaling charts
- Roofline model analysis
- Comparison bar charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from pathlib import Path


# Style configuration for consistent, professional plots
STYLE_CONFIG = {
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# NVIDIA color palette
NVIDIA_GREEN = '#76B900'
NVIDIA_DARK = '#1A1A1A'
COLORS = {
    'nvidia_green': '#76B900',
    'cpu': '#4A90A4',
    'scalar': '#E74C3C',
    'vector': '#F39C12',
    'adaptive': '#9B59B6',
    'cusparse': '#76B900',
}


def apply_style():
    """Apply consistent style to all plots."""
    plt.rcParams.update(STYLE_CONFIG)


def plot_sparsity_pattern(csr_matrix, title=None, output_path=None, 
                          max_display=2000, figsize=(10, 10)):
    """
    Visualize matrix sparsity pattern.
    
    Args:
        csr_matrix: CSRMatrix instance
        title: Plot title
        output_path: Path to save figure
        max_display: Maximum dimension to display (subsamples larger matrices)
        figsize: Figure size tuple
    """
    apply_style()
    
    rows, cols = csr_matrix.shape
    
    # Subsample if matrix is too large
    if rows > max_display or cols > max_display:
        # Create density map instead of exact pattern
        return plot_density_map(csr_matrix, title, output_path, figsize)
    
    # Build coordinate lists
    row_coords = []
    col_coords = []
    
    for i in range(rows):
        start = csr_matrix.indptr[i]
        end = csr_matrix.indptr[i + 1]
        for j in range(start, end):
            row_coords.append(i)
            col_coords.append(csr_matrix.indices[j])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(col_coords, row_coords, s=1, c=NVIDIA_GREEN, marker='s')
    
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # Invert y-axis for matrix convention
    ax.set_aspect('equal')
    
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    
    if title is None:
        title = f'Sparsity Pattern ({rows}×{cols}, {csr_matrix.nnz:,} non-zeros, {csr_matrix.density:.2%} density)'
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_density_map(csr_matrix, title=None, output_path=None, 
                     figsize=(10, 8), grid_size=100):
    """
    Visualize large matrix as density heatmap.
    
    Args:
        csr_matrix: CSRMatrix instance
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size tuple
        grid_size: Number of bins in each dimension
    """
    apply_style()
    
    rows, cols = csr_matrix.shape
    
    # Create density grid
    density_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    row_scale = grid_size / rows
    col_scale = grid_size / cols
    
    for i in range(rows):
        start = csr_matrix.indptr[i]
        end = csr_matrix.indptr[i + 1]
        grid_row = min(int(i * row_scale), grid_size - 1)
        
        for j in range(start, end):
            grid_col = min(int(csr_matrix.indices[j] * col_scale), grid_size - 1)
            density_grid[grid_row, grid_col] += 1
    
    # Normalize
    cell_area = (rows / grid_size) * (cols / grid_size)
    density_grid = density_grid / cell_area
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use log scale for better visualization
    density_grid_plot = np.ma.masked_where(density_grid == 0, density_grid)
    
    im = ax.imshow(density_grid_plot, cmap='YlGn', aspect='auto',
                   norm=LogNorm(vmin=max(density_grid_plot.min(), 1e-6), 
                               vmax=density_grid_plot.max()))
    
    cbar = plt.colorbar(im, ax=ax, label='Local Density')
    
    ax.set_xlabel('Column Block')
    ax.set_ylabel('Row Block')
    
    if title is None:
        title = f'Density Map ({rows:,}×{cols:,}, {csr_matrix.nnz:,} non-zeros, {csr_matrix.density:.4%} density)'
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_row_length_distribution(csr_matrix, title=None, output_path=None, figsize=(10, 6)):
    """
    Histogram of non-zeros per row.
    
    Important for understanding load balancing in SpMV.
    """
    apply_style()
    
    # Calculate row lengths
    row_lengths = np.diff(csr_matrix.indptr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(row_lengths, bins=50, color=NVIDIA_GREEN, edgecolor='white', alpha=0.8)
    
    ax.axvline(np.mean(row_lengths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(row_lengths):.1f}')
    ax.axvline(np.median(row_lengths), color='blue', linestyle='--',
               label=f'Median: {np.median(row_lengths):.1f}')
    
    ax.set_xlabel('Non-zeros per Row')
    ax.set_ylabel('Frequency')
    
    if title is None:
        title = f'Row Length Distribution (std={np.std(row_lengths):.1f})'
    ax.set_title(title)
    
    ax.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_performance_scaling(sizes, results_dict, title=None, 
                             output_path=None, figsize=(12, 6)):
    """
    Plot performance (GB/s or GFLOPS) vs matrix size.
    
    Args:
        sizes: List of matrix sizes (N for NxN matrices)
        results_dict: Dict mapping kernel names to performance lists
        title: Plot title
        output_path: Path to save figure
    """
    apply_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Absolute performance
    for name, values in results_dict.items():
        color = COLORS.get(name.lower(), 'gray')
        ax1.plot(sizes, values, 'o-', label=name, color=color, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Matrix Size (N for N×N)')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.set_title('Absolute Performance')
    
    # Add P100 peak bandwidth reference line
    p100_peak = 732  # GB/s
    ax1.axhline(p100_peak, color='gray', linestyle=':', alpha=0.5, label='P100 Peak (732 GB/s)')
    
    # Right: Relative to cuSPARSE (if present)
    if 'cuSPARSE' in results_dict or 'cusparse' in results_dict:
        cusparse_key = 'cuSPARSE' if 'cuSPARSE' in results_dict else 'cusparse'
        cusparse_perf = np.array(results_dict[cusparse_key])
        
        for name, values in results_dict.items():
            if name.lower() == 'cusparse':
                continue
            color = COLORS.get(name.lower(), 'gray')
            relative = np.array(values) / cusparse_perf * 100
            ax2.plot(sizes, relative, 'o-', label=name, color=color, linewidth=2, markersize=6)
        
        ax2.axhline(100, color=COLORS['cusparse'], linestyle='--', label='cuSPARSE (100%)')
        ax2.set_ylabel('Performance (% of cuSPARSE)')
    else:
        # Relative to best custom kernel
        best_key = max(results_dict.keys(), key=lambda k: np.mean(results_dict[k]))
        best_perf = np.array(results_dict[best_key])
        
        for name, values in results_dict.items():
            color = COLORS.get(name.lower(), 'gray')
            relative = np.array(values) / best_perf * 100
            ax2.plot(sizes, relative, 'o-', label=name, color=color, linewidth=2, markersize=6)
        
        ax2.set_ylabel(f'Performance (% of {best_key})')
    
    ax2.set_xlabel('Matrix Size (N for N×N)')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.set_title('Relative Performance')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    return fig, (ax1, ax2)


def plot_performance_comparison(kernel_names, performance_values, 
                                metric='GB/s', title=None, 
                                output_path=None, figsize=(10, 6)):
    """
    Bar chart comparing kernel performance.
    
    Args:
        kernel_names: List of kernel names
        performance_values: List of performance values
        metric: Performance metric label
        title: Plot title
        output_path: Path to save figure
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = [COLORS.get(name.lower(), 'gray') for name in kernel_names]
    bars = ax.bar(kernel_names, performance_values, color=colors, edgecolor='white')
    
    # Add value labels on bars
    for bar, val in zip(bars, performance_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel(metric)
    ax.set_xlabel('Implementation')
    
    if title is None:
        title = f'SpMV Performance Comparison'
    ax.set_title(title)
    
    # Add P100 peak reference if applicable
    if 'GB/s' in metric:
        ax.axhline(732, color='gray', linestyle=':', alpha=0.5)
        ax.text(len(kernel_names) - 0.5, 740, 'P100 Peak: 732 GB/s', 
                ha='right', va='bottom', fontsize=9, color='gray')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_roofline(results, p100_peak_bw=732, p100_peak_flops=4700,
                  title=None, output_path=None, figsize=(10, 8)):
    """
    Roofline model plot for SpMV analysis.
    
    Args:
        results: List of dicts with keys: 'name', 'flops', 'bytes', 'time_s'
        p100_peak_bw: Peak memory bandwidth in GB/s
        p100_peak_flops: Peak FLOPS in GFLOPS (FP32)
        title: Plot title
        output_path: Path to save figure
    
    SpMV is memory-bound: 2 FLOPs per non-zero (multiply + add),
    but requires loading matrix values, indices, and vector elements.
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Arithmetic intensity range for roofline
    ai_range = np.logspace(-3, 2, 1000)
    
    # Roofline ceiling
    memory_roof = ai_range * p100_peak_bw  # GFLOPS limited by memory
    compute_roof = np.full_like(ai_range, p100_peak_flops)  # GFLOPS limited by compute
    roofline = np.minimum(memory_roof, compute_roof)
    
    ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline')
    ax.fill_between(ai_range, roofline, alpha=0.1, color='gray')
    
    # Ridge point
    ridge_ai = p100_peak_flops / p100_peak_bw
    ax.axvline(ridge_ai, color='gray', linestyle=':', alpha=0.5)
    ax.text(ridge_ai * 1.1, p100_peak_flops * 0.8, f'Ridge Point\nAI={ridge_ai:.2f}',
            fontsize=9, color='gray')
    
    # Plot data points
    for i, r in enumerate(results):
        ai = r['flops'] / r['bytes']  # FLOP/Byte
        achieved_gflops = r['flops'] / r['time_s'] / 1e9
        
        color = COLORS.get(r['name'].lower(), f'C{i}')
        ax.scatter(ai, achieved_gflops, s=150, c=color, marker='o', 
                   label=f"{r['name']}: {achieved_gflops:.1f} GFLOPS", zorder=5)
    
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)')
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(1, p100_peak_flops * 2)
    ax.legend(loc='lower right')
    
    if title is None:
        title = 'Roofline Model Analysis - Tesla P100'
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    return fig, ax


def create_summary_figure(csr_matrix, benchmark_results, output_path=None):
    """
    Create a comprehensive summary figure combining multiple visualizations.
    
    Args:
        csr_matrix: CSRMatrix instance
        benchmark_results: Dict with benchmark data
        output_path: Path to save figure
    """
    apply_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Sparsity pattern / density map (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    # Simplified density visualization for summary
    row_lengths = np.diff(csr_matrix.indptr)
    ax1.bar(range(min(100, len(row_lengths))), row_lengths[:100], 
            color=NVIDIA_GREEN, width=1.0)
    ax1.set_xlabel('Row Index (first 100 rows)')
    ax1.set_ylabel('Non-zeros per Row')
    ax1.set_title(f'Row Length Pattern ({csr_matrix.shape[0]:,}×{csr_matrix.shape[1]:,}, {csr_matrix.density:.4%} density)')
    
    # 2. Row length histogram (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(row_lengths, bins=30, color=NVIDIA_GREEN, edgecolor='white', alpha=0.8)
    ax2.axvline(np.mean(row_lengths), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Non-zeros per Row')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Row Length Distribution')
    
    # 3. Performance comparison (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'kernels' in benchmark_results and 'bandwidth' in benchmark_results:
        bars = ax3.bar(benchmark_results['kernels'], benchmark_results['bandwidth'],
                       color=[COLORS.get(k.lower(), 'gray') for k in benchmark_results['kernels']])
        for bar, val in zip(bars, benchmark_results['bandwidth']):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    ax3.set_ylabel('Bandwidth (GB/s)')
    ax3.set_title('Kernel Performance')
    ax3.axhline(732, color='gray', linestyle=':', alpha=0.5)
    
    # 4. Speedup vs CPU (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'kernels' in benchmark_results and 'speedup' in benchmark_results:
        bars = ax4.bar(benchmark_results['kernels'], benchmark_results['speedup'],
                       color=[COLORS.get(k.lower(), 'gray') for k in benchmark_results['kernels']])
        for bar, val in zip(bars, benchmark_results['speedup']):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}×', ha='center', va='bottom', fontsize=10)
    ax4.set_ylabel('Speedup vs NumPy CPU')
    ax4.set_title('GPU Speedup')
    
    # 5. Key metrics text box (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    metrics_text = f"""
    Matrix Statistics
    ─────────────────
    Dimensions: {csr_matrix.shape[0]:,} × {csr_matrix.shape[1]:,}
    Non-zeros:  {csr_matrix.nnz:,}
    Density:    {csr_matrix.density:.4%}
    
    Memory Usage
    ─────────────────
    CSR Format: {csr_matrix.bytes_csr / 1024 / 1024:.2f} MB
    Dense:      {csr_matrix.bytes_dense / 1024 / 1024:.2f} MB
    Compression: {csr_matrix.bytes_dense / csr_matrix.bytes_csr:.1f}×
    
    Hardware: Tesla P100
    Peak BW:  732 GB/s
    """
    
    ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('SpMV Performance Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualizations with dummy data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from formats.csr import CSRMatrix
    
    print("Testing visualization module...")
    
    # Create test matrix
    A = CSRMatrix.random(5000, 5000, density=0.005, seed=42)
    print(f"Test matrix: {A}")
    
    # Test plots
    output_dir = Path(__file__).parent.parent / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_sparsity_pattern(A, output_path=output_dir / "sparsity_pattern.png")
    plot_row_length_distribution(A, output_path=output_dir / "row_distribution.png")
    
    # Dummy benchmark results
    benchmark_results = {
        'kernels': ['CPU', 'Scalar', 'Vector', 'Adaptive'],
        'bandwidth': [15, 180, 320, 350],
        'speedup': [1.0, 12.0, 21.3, 23.3]
    }
    
    create_summary_figure(A, benchmark_results, output_path=output_dir / "summary.png")
    
    print("\nOK -> Visualization tests complete!")
