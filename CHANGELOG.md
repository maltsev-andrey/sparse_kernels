# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - Phase 2: Optimization

### Planned
- Shared memory caching for vector x access
- Row binning by length for better load balancing
- Block size tuning experiments
- COO format implementation

---

## [0.1.0] - 2024-12-26 - Phase 1: Baseline Implementation

### Added

#### Core Implementation
- **CSRMatrix class** with efficient storage and GPU transfer
  - `from_dense()`, `from_scipy()`, `random()` constructors
  - Automatic GPU memory management
  - Memory footprint calculation

- **Three SpMV kernel variants:**
  - `spmv_csr_scalar_kernel` - One thread per row (baseline)
  - `spmv_csr_vector_kernel` - One warp per row with shuffle reduction
  - `spmv_csr_adaptive_kernel` - Dynamic strategy selection

#### Benchmarking
- **SpMVBenchmark class** with CUDA event timing
- Bandwidth and GFLOPS calculation
- cuSPARSE comparison via CuPy
- Automatic correctness verification

#### Visualization
- Sparsity pattern plots (scatter and density heatmap)
- Row length distribution histograms
- Performance comparison bar charts
- Roofline model analysis
- Summary figure combining all metrics

#### Documentation
- README with benchmark results and project goals
- ARCHITECTURE.md with technical design details
- Code comments explaining optimization decisions

### Performance Results (Tesla P100)

**Single Matrix (50K × 50K, 0.1% density):**

| Kernel     | Bandwidth  | % cuSPARSE |
|------------|------------|------------|
| Scalar     | 21.0 GB/s  | 22%        |
| Vector     | 28.6 GB/s  | 30%        |
| Adaptive   | 28.5 GB/s  | 30%        |

**Scaling Study (0.1% density):**

| Size  | Vector BW  | % cuSPARSE | CPU Speedup |
|-------|------------|------------|-------------|
| 5K    | 0.5 GB/s   | 20%        | 31×         |
| 25K   | 9.1 GB/s   | 22%        | 671×        |
| 50K   | 30.4 GB/s  | 31%        | 2,222×      |
| 75K   | 56.2 GB/s  | 42%        | 4,223×      |
| 100K  | 73.5 GB/s  | 48%        | 5,432×      |

**Key Finding:** Performance improves with matrix size — reaches 48% of cuSPARSE at 100K×100K.

### Known Limitations
- Low bandwidth utilization due to scattered vector access
- No shared memory optimization
- Basic row parallelization without binning
- Single precision (float32) only

---

## Version History

| Version | Date       | Phase        | Key Achievement                |
|---------|------------|--------------|--------------------------------|
| 0.1.0   | 2024-12-26 | Baseline     | 31% of cuSPARSE, 2183× vs CPU  |
| 0.2.0   | TBD        | Optimization | Target: 50% of cuSPARSE        |
| 0.3.0   | TBD        | Advanced     | Target: 70% of cuSPARSE        |

