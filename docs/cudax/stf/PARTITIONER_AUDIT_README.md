# Partitioner Audit Investigation - Summary

## Purpose

This directory contains the investigation report for understanding the current partitioner implementation in cudax STF and the 2 MiB page alignment issue in multi-GPU memory management.

## Investigation Files

### 1. `padded_blocked_partitioner_investigation.md` (Main Report)

Comprehensive investigation covering:
- **Code locations** - All relevant source files and their relationships
- **API documentation** - Public interfaces and usage patterns
- **Memory management flow** - How data is allocated and distributed
- **Problem analysis** - Detailed explanation of the 2 MiB page imbalance issue
- **Call graphs** - Control flow from user code to hardware
- **Design considerations** - Constraints, assumptions, and gaps
- **Recommendations** - Next steps and potential solutions

**Key Sections:**
- Section 2: Partitioner API and Interface
- Section 3: Multi-GPU Memory Management Flow
- Section 4: The 2 MiB Page Imbalance Problem (most critical)
- Section 8: Identified Gaps and Design Considerations
- Section 9: Touchpoints Between Partitioners and Memory Manager
- Section 10: Recommendations for Next Phase

### 2. `partitioner_memory_flow_diagram.txt` (Visual Aid)

ASCII diagram showing:
- Complete data flow from user code to GPU memory
- Concrete example with 100×100×100 array on 2 GPUs
- Comparison of logical vs. physical partitioning
- Visualization of page boundary misalignment
- Proposed solution approach

## Key Findings

### Current Architecture

1. **Partitioners** (`blocked_partition`, `tiled_partition`, etc.) operate at **logical element level**
2. **Memory manager** (`localized_array`) operates at **physical 2 MiB page level**
3. **No coordination** between logical partitioning and physical allocation

### The Problem

When buffer sizes are not multiples of 2 MiB:
- Pages may straddle logical partition boundaries
- Random sampling (10 samples per page) determines device ownership
- Results in **non-deterministic** and **imbalanced** memory distribution
- Potential for remote memory access penalties

### Example (from fdtd_mgpu.cu)

**Setup:** 100×100×100 double array (8 MB total) on 2 GPUs

**Logical partition:**
- Device 0: elements [0, 500K) = 4 MB
- Device 1: elements [500K, 1M) = 4 MB

**Physical allocation:** 4 pages × 2 MiB = 8 MB

**Problem:** Page 1 (elements 262K-524K) spans both partitions!

**Outcomes:**
- Scenario A: Page 1 → Device 0 → balanced pages (2 vs 2) but cross-device access
- Scenario B: Page 1 → Device 1 → **imbalanced** (1 vs 3 pages)

## Code Locations (Quick Reference)

### Partitioner Implementations
```
cudax/include/cuda/experimental/__stf/places/
├── blocked_partition.cuh    ← Main implementation
├── tiled_partition.cuh
└── cyclic_shape.cuh
```

### Memory Management
```
cudax/include/cuda/experimental/__stf/localization/
└── composite_slice.cuh      ← localized_array class (page allocation)
```

### Integration
```
cudax/include/cuda/experimental/__stf/internal/
└── parallel_for_scope.cuh   ← Partitioner invocation (line 688)
```

### Examples
```
cudax/examples/stf/
├── fdtd_mgpu.cu            ← Maxwell equations (referenced in ticket)
├── heat_mgpu.cu
└── jacobi.cu
```

## Touchpoints Between Components

### 1. Partitioner → Memory Manager
- `get_executor()` function pointer stored in `data_place::composite_state`
- Called during page ownership sampling in `localized_array::block_to_grid_pos()`

### 2. Shape Information
- User defines shape → Partitioner computes sub-shapes → Memory manager allocates pages

### 3. Grid Dimensions
- `exec_place_grid` provides device topology
- Passed to both partitioner and memory manager

## Design Constraints

1. **VMM Required:** Must support CUDA Virtual Memory Management
2. **2 MiB Pages:** Fixed allocation granularity (hardware limitation)
3. **Static Partitioner Interface:** Only static methods, no state
4. **Single VA Range:** Contiguous virtual address space for all devices

## Proposed Solution Direction

### Page-Aligned Blocked Partition

```cpp
// Align partition boundaries to page size
size_t page_size_elements = page_size / elem_size;
size_t part_size = ceil(ideal_size / page_size_elements) * page_size_elements;
```

**Benefits:**
- Deterministic page assignment
- No boundary pages
- Perfect memory locality

**Trade-offs:**
- Slight computational imbalance (acceptable)
- Marginally increased memory usage (padding)

## Next Steps (From Section 10)

1. **Quantify the Problem**
   - Create measurement harness
   - Log actual memory distribution
   - Test various buffer sizes

2. **Design Padded Partitioner**
   - Implement page-aligned blocked partition
   - Add deterministic boundary handling
   - Maintain backward compatibility

3. **Validation**
   - Modify fdtd_mgpu example
   - Verify balanced distribution
   - Benchmark performance

4. **Documentation**
   - API documentation
   - Usage examples
   - Migration guide

## Open Design Questions

1. Should page size be compile-time or runtime parameter?
2. How to handle heterogeneous GPU configurations?
3. Should padding be explicit in the API?
4. Impact on ghost zones and halo regions?

## References

- Ticket: "Audit partitioners"
- Example: `cudax/examples/stf/fdtd_mgpu.cu`
- Main report: `padded_blocked_partitioner_investigation.md`
- Diagram: `partitioner_memory_flow_diagram.txt`

## Acceptance Criteria Status

✅ **Report captures file locations** - Section 1 and Appendix A
✅ **Control flow diagrams** - Section 5 and partitioner_memory_flow_diagram.txt
✅ **Explanation of current uneven partitioning** - Section 4 with concrete example
✅ **Confirms touchpoints** - Section 9 with detailed mapping
✅ **Highlights constraints/assumptions** - Sections 7 and 8

---

**Date:** October 29, 2024  
**Investigation Branch:** `investigate-partitioners-mgpu-2mib-page-imbalance`
