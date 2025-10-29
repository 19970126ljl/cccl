# Partitioner Memory Management Investigation Report

## Executive Summary

This document investigates the current behavior of partitioners in cudax STF, specifically focusing on multi-GPU memory management and the 2 MiB page alignment issue that causes imbalanced memory distribution across devices.

## 1. Code Locations and Architecture

### 1.1 Core Partitioner Implementations

**Location:** `cudax/include/cuda/experimental/__stf/places/`

- **`blocked_partition.cuh`** (174 lines)
  - Main blocked partitioning strategy
  - Template class `blocked_partition_custom<which_dim>` with default alias `blocked_partition`
  - Partitions multidimensional data into contiguous blocks along a selected dimension (default: last dimension)

- **`tiled_partition.cuh`** (192 lines)
  - Tiled partitioning with configurable tile size
  - Template class `tiled_partition<tile_size>`
  - Distributes tiles cyclically across execution places

- **`cyclic_shape.cuh`** (9.3 KB)
  - Cyclic distribution strategy
  - Used for round-robin style partitioning

### 1.2 Memory Management Components

**Location:** `cudax/include/cuda/experimental/__stf/localization/composite_slice.cuh` (377 lines)

Key class: `localized_array`
- Implements multi-GPU memory allocation using CUDA Virtual Memory Management (VMM)
- Uses `CUmemGenericAllocationHandle` for physical allocations
- Maps physical pages to virtual address space with device-specific placement

**Location:** `cudax/include/cuda/experimental/__stf/places/places.cuh` (1751 lines)

Key classes:
- `data_place`: Represents where data is stored (CPU, GPU device, composite multi-GPU)
- `data_place::composite_state`: Stores grid and partitioner function for composite data places
- `exec_place` and `exec_place_grid`: Represents where execution occurs

### 1.3 Partitioner Integration Points

**Location:** `cudax/include/cuda/experimental/__stf/internal/parallel_for_scope.cuh` (1045 lines)

Key method: `parallel_for_scope::operator->*`
- Line 557: Creates composite data place: `data_place::composite(partitioner_t(), e_place.as_grid())`
- Line 688: Applies partitioner: `partitioner_t::apply(shape, pos4(i), t.grid_dims())`
- Iterates over grid places, computing sub-shape for each device

## 2. Partitioner API and Interface

### 2.1 Public Interface

All partitioners must implement:

```cpp
template <typename shape_t>
static auto apply(const shape_t& in, pos4 place_position, dim4 grid_dims) -> box<dimensions>
```

**Parameters:**
- `in`: Input shape (can be `box<dimensions>` or mdspan-like shape)
- `place_position`: Position in the execution grid (e.g., `pos4(0,0,0,0)` for first device)
- `grid_dims`: Dimensions of the execution grid (e.g., `dim4(4,1,1,1)` for 4 devices)

**Returns:** A sub-shape (box) representing the partition for this place

```cpp
static pos4 get_executor(pos4 data_coords, dim4 data_dims, dim4 grid_dims)
```

**Parameters:**
- `data_coords`: Coordinates of a data element
- `data_dims`: Overall data dimensions
- `grid_dims`: Execution grid dimensions

**Returns:** Grid position (device ID) that owns this data element

### 2.2 Blocked Partition Algorithm (Current Implementation)

**File:** `cudax/include/cuda/experimental/__stf/places/blocked_partition.cuh`

**Algorithm (lines 62-71):**
```cpp
size_t nplaces = grid_dims.x;
std::ptrdiff_t dim_beg = bounds[target_dim].first;
std::ptrdiff_t dim_end = bounds[target_dim].second;
size_t cnt = dim_end - dim_beg;
std::ptrdiff_t part_size = (cnt + nplaces - 1) / nplaces;  // Ceiling division

bounds[target_dim].first = std::min(dim_beg + part_size * place_position.x, dim_end);
bounds[target_dim].second = std::min(dim_beg + part_size * (place_position.x + 1), dim_end);
```

**Key characteristics:**
- Uses ceiling division: `part_size = (cnt + nplaces - 1) / nplaces`
- Last device may get smaller partition when `cnt % nplaces != 0`
- Partitioning is LOGICAL (based on element count) not PHYSICAL (based on page allocation)

## 3. Multi-GPU Memory Management Flow

### 3.1 Allocation Granularity and Page Size

**File:** `cudax/include/cuda/experimental/__stf/localization/composite_slice.cuh`

**Lines 88-103:** Initialization of `localized_array`

```cpp
CUmemAllocationProp prop = {};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = dev};

size_t alloc_granularity_bytes = cuda_try<cuMemGetAllocationGranularity>(&prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
// Typically 2 MiB (2097152 bytes) on modern GPUs

block_size_bytes = alloc_granularity_bytes;
vm_total_size_bytes = ((total_size_bytes + alloc_granularity_bytes - 1) / alloc_granularity_bytes) * alloc_granularity_bytes;
// Round up to multiple of page size

size_t nblocks = vm_total_size_bytes / alloc_granularity_bytes;
```

### 3.2 Device Mapping Strategy (Sampling-Based)

**Lines 120-126:** Determine device ownership for each page

```cpp
std::vector<pos4> owner;
owner.reserve(nblocks);
for (size_t i = 0; i < nblocks; i++)
{
    owner.push_back(block_to_grid_pos(i * block_size_bytes / elemsize, alloc_granularity_bytes / elemsize, delinearize));
}
```

**Lines 263-296:** `block_to_grid_pos` implementation (sampling approach)

```cpp
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, static_cast<int>(allocation_granularity - 1));

const size_t nsamples = 10;
std::array<pos4, nsamples> sampled_pos;
for (size_t sample = 0; sample < nsamples; sample++)
{
    size_t index = linearized_index + dis(gen);
    sampled_pos[sample] = index_to_grid_pos(index, delinearize);
}

// Count occurrences and select most common position
std::unordered_map<pos4, size_t, hash<pos4>> sample_cnt;
for (auto& s : sampled_pos)
{
    ++sample_cnt[s];
}

// Return the position with maximum count
```

**Key issue:** Samples 10 random indices within each 2 MiB page block and assigns the page to the device that appears most frequently. This works well when pages align with partition boundaries but fails at boundaries.

### 3.3 Physical Allocation and Mapping

**Lines 131-145:** Merge consecutive blocks with same owner

```cpp
for (size_t i = 0; i < nblocks;)
{
    pos4 p = owner[i];
    size_t j = 0;
    // Count consecutive blocks with the same position in the grid
    while ((i + j < nblocks) && (owner[i + j] == p))
    {
        j++;
    }
    meta.emplace_back(grid_pos_to_dev(p), j * alloc_granularity_bytes, i * block_size_bytes);
    i += j;
}
```

**Lines 151-188:** Create physical allocations and set up access permissions

```cpp
for (auto& item : meta)
{
    CUmemAllocationProp create_prop = {};
    create_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    create_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    create_prop.location.id = item.dev;
    
    cuda_safe_call(cuMemCreate(&item.alloc_handle, item.size, &create_prop, 0));
    cuda_safe_call(cuMemMap(base_ptr + item.offset, item.size, 0ULL, item.alloc_handle, 0ULL));
    
    // Set up peer access for all devices
}
```

## 4. The 2 MiB Page Imbalance Problem

### 4.1 Problem Description

When buffer sizes are not multiples of 2 MiB, uneven partitioning occurs due to mismatch between:
1. **Logical partitioning** (element-based, done by `blocked_partition`)
2. **Physical allocation** (page-based, done by `localized_array`)

### 4.2 Concrete Example

Consider a 2-GPU scenario with `fdtd_mgpu.cu`:

**Configuration:**
- Shape: `(100, 100, 100)` 3D array
- Element size: `sizeof(double)` = 8 bytes
- Total size: 1,000,000 elements × 8 = 8,000,000 bytes = 7.63 MiB
- Page size: 2 MiB = 2,097,152 bytes
- Number of pages: `ceil(8,000,000 / 2,097,152)` = 4 pages

**Logical partition (blocked_partition on last dimension):**
- Device 0: indices [0, 50) → elements 0-499,999 → 4,000,000 bytes
- Device 1: indices [50, 100) → elements 500,000-999,999 → 4,000,000 bytes

**Physical allocation:**
- Total VM reservation: 4 pages × 2 MiB = 8,388,608 bytes (rounded up from 8,000,000)
- Extra padding: 388,608 bytes

**Page boundaries (linearized in z-dimension):**
- Page 0: bytes [0, 2,097,152) → elements [0, 262,144)
- Page 1: bytes [2,097,152, 4,194,304) → elements [262,144, 524,288)
- Page 2: bytes [4,194,304, 6,291,456) → elements [524,288, 786,432)
- Page 3: bytes [6,291,456, 8,388,608) → elements [786,432, 1,048,576)

**Logical ownership (element 500,000 is boundary):**
- Elements [0, 500,000) → Device 0
- Elements [500,000, 1,000,000) → Device 1

**Page ownership via sampling:**
- Page 0 (elements 0-262,143): Samples hit Device 0 → Device 0 ✓
- Page 1 (elements 262,144-524,287): Samples hit Device 0 AND Device 1 → **Contested!**
- Page 2 (elements 524,288-786,431): Samples hit Device 1 → Device 1 ✓
- Page 3 (elements 786,432-1,048,575): Samples hit Device 1 AND padding → Device 1 ✓

### 4.3 Imbalance Scenarios

**Case 1: Page 1 assigned to Device 0**
- Device 0: Pages 0, 1 = 4 MiB
- Device 1: Pages 2, 3 = 4 MiB
- Result: Balanced page count BUT logical partition mismatch

**Case 2: Page 1 assigned to Device 1**
- Device 0: Page 0 = 2 MiB (undersized!)
- Device 1: Pages 1, 2, 3 = 6 MiB (oversized!)
- Result: Severe imbalance

### 4.4 Root Causes

1. **Granularity Mismatch:**
   - Logical partitioning operates at element granularity
   - Physical allocation operates at 2 MiB page granularity
   - No alignment enforcement between the two

2. **Boundary Pages:**
   - Pages that straddle partition boundaries get assigned entirely to one device
   - Sampling approach (10 random samples) may not accurately represent ownership when page contains elements from multiple partitions

3. **Padding Allocation:**
   - Rounding up to page size creates "phantom" elements beyond logical data bounds
   - Last page may contain significant padding (up to 2 MiB - 1 byte)
   - Padding ownership affects physical allocation

4. **Non-Determinism:**
   - Random sampling introduces non-determinism in page assignment
   - Lines 263-265: `std::random_device` and `std::uniform_int_distribution` used
   - Same buffer may get different device mappings across runs

## 5. Call Graphs and Control Flow

### 5.1 Partitioner Setup Flow

```
Context::parallel_for(partitioner, exec_place, shape, deps...)
    ↓
parallel_for_scope::operator->*()
    ↓ (line 557)
data_place::composite(partitioner_t(), exec_place.grid())
    ↓ (line 1365)
data_place::composite(&partitioner_t::get_executor, grid)
    ↓ (line 1348)
Stores: composite_state { grid, partitioner_func }
```

### 5.2 Execution Flow (Grid of Places)

```
parallel_for_scope::operator->*()
    ↓ (lines 684-691)
For each place i in grid:
    ↓
    sub_shape = partitioner_t::apply(shape, pos4(i), grid_dims)
    ↓
    do_parallel_for(f, place[i], sub_shape, task)
```

### 5.3 Memory Allocation Flow

```
Task needs data on composite data place
    ↓
composite_slice_cache::get()
    ↓
localized_array constructor
    ↓
cuMemGetAllocationGranularity() → page_size (typically 2 MiB)
    ↓
Calculate nblocks = ceil(total_bytes / page_size)
    ↓
For each block:
    ↓
    block_to_grid_pos() with sampling
        ↓
        Sample 10 random indices within page
        ↓
        For each sample:
            ↓
            delinearize(index) → coords
            ↓
            partitioner_func(coords, data_dims, grid_dims) → grid_pos
        ↓
        Return most frequent grid_pos
    ↓
Merge consecutive pages with same owner
    ↓
cuMemCreate() + cuMemMap() for each device
```

## 6. Example Usage: fdtd_mgpu.cu

**File:** `cudax/examples/stf/fdtd_mgpu.cu`

**Line 124:** Setup execution place
```cpp
exec_place where = exec_place::all_devices();  // Grid of all available devices
```

**Lines 172-178:** Define 3D data shapes
```cpp
auto data_shape = shape_of<slice<double, 3>>(SIZE_X, SIZE_Y, SIZE_Z);  // 100×100×100
auto lEx = ctx.logical_data(data_shape);
auto lEy = ctx.logical_data(data_shape);
auto lEz = ctx.logical_data(data_shape);
// ... more fields
```

**Lines 191-196:** Initialize with blocked partition
```cpp
ctx.parallel_for(blocked_partition(), where, data_shape, lEx.write(), lEy.write(), lEz.write())
    ->*[] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Ex, auto Ey, auto Ez) {
        Ex(i, j, k) = 0.0;
        Ey(i, j, k) = 0.0;
        Ez(i, j, k) = 0.0;
    };
```

**Behavior with 2 devices:**
- `blocked_partition()` partitions along Z dimension (last dim)
- Device 0 gets Z ∈ [0, 50), Device 1 gets Z ∈ [50, 100)
- Total bytes = 100×100×100×8 = 8,000,000 bytes
- Pages needed = 4 × 2 MiB
- Boundary page (Page 1) spans both devices' logical partitions
- **Result:** Potential memory imbalance depending on sampling

## 7. Constraints and Assumptions

### 7.1 Current Design Assumptions

1. **VMM Availability:**
   - Requires `CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED`
   - Line 83-85 in `composite_slice.cuh`

2. **Page Size Universality:**
   - Assumes all devices have same allocation granularity
   - Line 93: queries only current device's granularity
   - Could be problematic in heterogeneous GPU configurations

3. **Sampling Accuracy:**
   - Assumes 10 samples sufficiently represent page ownership
   - Works well for large pages with uniform partition
   - Breaks down at partition boundaries

4. **Peer Access:**
   - Attempts to enable peer access between all device pairs
   - Falls back gracefully if not supported (line 178-180)

### 7.2 API Constraints

1. **Partitioner Signature:**
   - Static methods only (no state)
   - Must return `box<dimensions>` from `apply()`
   - Must return `pos4` from `get_executor()`

2. **Grid Structure:**
   - Grid dimensions stored in `dim4` (4D maximum)
   - Place positions stored in `pos4`
   - Linearization happens in data structures

3. **Composite Data Place Uniqueness:**
   - Defined by `(exec_place_grid, partitioner_func)` pair
   - Line 1320-1322 in `places.cuh`
   - Cached in `composite_slice_cache`

### 7.3 Memory Management Constraints

1. **Allocation Granularity:**
   - Cannot allocate smaller than page size (typically 2 MiB)
   - Must round up total allocation to page boundary

2. **Virtual Address Space:**
   - Single contiguous VA range for entire allocation
   - Simplifies pointer arithmetic but couples logical and physical layout

3. **Peer Access Requirements:**
   - All devices must access all pages
   - May impact performance on systems without NVLink

## 8. Identified Gaps and Design Considerations

### 8.1 Critical Gaps

1. **No Padding-Aware Partitioning:**
   - Partitioners don't know about or account for page alignment
   - No mechanism to pad logical partitions to page boundaries

2. **Boundary Ambiguity:**
   - When page straddles partitions, assignment is non-deterministic
   - No conflict resolution strategy

3. **Lack of Alignment Hints:**
   - No API for partitioner to specify alignment requirements
   - No feedback from memory manager about actual allocation

4. **No Balance Guarantees:**
   - Physical memory distribution not considered in partitioning
   - No load balancing mechanism

### 8.2 Design Questions for New Padded Partitioner

1. **Alignment Strategy:**
   - Should partitioner round partition sizes to page boundaries?
   - Should it accept page_size as a parameter?

2. **Padding Placement:**
   - Distribute padding evenly across devices?
   - Place all padding on last device?
   - Minimize cross-device boundaries?

3. **API Extensions Needed:**
   - Does `apply()` need to know page_size?
   - Should it return `(box, padding_info)`?
   - Should `get_executor()` be aware of padding?

4. **Backward Compatibility:**
   - Should new partitioner be drop-in replacement?
   - New parameter for existing partitioners vs. new class?

5. **Performance Trade-offs:**
   - Slightly larger allocations (padding) vs. balanced distribution
   - Wasted memory vs. better locality

### 8.3 Potential Approaches

**Option A: Page-Aligned Blocked Partition**
```cpp
// Compute part_size as multiple of page_size
size_t page_size_elements = page_size / elem_size;
size_t part_size = ((cnt / nplaces + page_size_elements - 1) / page_size_elements) * page_size_elements;
```
Pros: Simple, predictable
Cons: May over-allocate significantly for small partitions

**Option B: Boundary-Aware Sampling**
```cpp
// When sampling, check if page is at boundary and use deterministic assignment
if (is_boundary_page(block_index))
{
    return deterministic_owner(block_index);
}
else
{
    return sample_based_owner(block_index);
}
```
Pros: Minimal allocation overhead
Cons: Still has complexity at boundaries

**Option C: Two-Phase Partitioning**
```cpp
// Phase 1: Compute ideal logical partition
// Phase 2: Adjust to page boundaries with explicit padding tracking
```
Pros: Clean separation of concerns
Cons: More complex API

## 9. Touchpoints Between Partitioners and Memory Manager

### 9.1 Direct Touchpoints

1. **`get_executor_func_t` storage** (places.cuh:57, 1336)
   - Function pointer stored in `data_place::composite_state`
   - Used by `localized_array::index_to_grid_pos()` (composite_slice.cuh:300-308)

2. **`delinearize` function** (composite_slice.cuh:68)
   - Passed to `localized_array` constructor
   - Used to convert linear index → multidimensional coordinates

3. **Sampling in `block_to_grid_pos()`** (composite_slice.cuh:257-297)
   - Calls partitioner's `get_executor` via `mapper` (stored function pointer)
   - Determines device ownership for each page

### 9.2 Indirect Touchpoints

1. **Shape information flow:**
   - `parallel_for_scope` → `partitioner::apply()` → returns sub-shape
   - `localized_array` → uses total shape for allocation size

2. **Grid dimensions:**
   - Stored in `exec_place_grid`
   - Passed to both `partitioner::apply()` and `partitioner::get_executor()`

3. **Data place affinity:**
   - Line 557 in `parallel_for_scope.cuh` creates composite data place
   - Task's data instances use this for allocation decisions

### 9.3 Extension Points for New Design

1. **Allocation granularity query:**
   - Could pass page_size to partitioner via extended API
   - Line 93 in `composite_slice.cuh` is where it's determined

2. **Custom sampling strategy:**
   - Could replace `block_to_grid_pos()` with partitioner-specific logic
   - Currently hardcoded at lines 257-297

3. **Padding metadata:**
   - Could extend `apply()` return type to include padding information
   - Memory manager could use this to optimize allocations

## 10. Recommendations for Next Phase

### 10.1 Immediate Next Steps

1. **Quantify the Problem:**
   - Create test harness that measures actual memory distribution
   - Run fdtd_mgpu with various sizes on 2+ GPUs
   - Log which pages assigned to which devices

2. **Design Padded Blocked Partitioner:**
   - Start with simple page-aligned approach (Option A)
   - Make page_size a template parameter or constructor argument
   - Ensure deterministic page assignment

3. **Prototype Implementation:**
   - Create `padded_blocked_partition` class in new file
   - Implement `apply()` with padding logic
   - Implement `get_executor()` with boundary awareness

4. **Validation:**
   - Modify fdtd_mgpu to use new partitioner
   - Verify balanced memory distribution
   - Benchmark performance impact

### 10.2 Design Principles

1. **Explicit over Implicit:**
   - Make padding explicit in partition boundaries
   - Don't rely on sampling for deterministic decisions

2. **Minimize API Changes:**
   - Keep existing partitioner interface if possible
   - Add new optional parameters rather than breaking changes

3. **Performance First:**
   - Prioritize balanced distribution over minimal allocation
   - Some memory overhead acceptable for better performance

4. **Debuggability:**
   - Add logging/visualization for memory distribution
   - Make it easy to verify correct behavior

### 10.3 Open Questions

1. Should page_size be:
   - Compile-time constant?
   - Runtime parameter passed to partitioner?
   - Queried internally by partitioner?

2. How to handle heterogeneous page sizes:
   - Different GPUs with different granularities?
   - Use LCM (Least Common Multiple)?

3. Multi-dimensional partitioning:
   - Current `blocked_partition` only partitions one dimension
   - Should padded version support 2D/3D partitioning?

4. Interaction with other features:
   - How does padding affect ghost zones/halos?
   - Impact on reduction operations?

---

## Appendix A: File Listing

**Partitioner Headers:**
- `cudax/include/cuda/experimental/__stf/places/blocked_partition.cuh`
- `cudax/include/cuda/experimental/__stf/places/tiled_partition.cuh`
- `cudax/include/cuda/experimental/__stf/places/cyclic_shape.cuh`
- `cudax/include/cuda/experimental/__stf/places/place_partition.cuh`
- `cudax/include/cuda/experimental/__stf/places/inner_shape.cuh`

**Memory Management:**
- `cudax/include/cuda/experimental/__stf/localization/composite_slice.cuh`

**Core Infrastructure:**
- `cudax/include/cuda/experimental/__stf/places/places.cuh`
- `cudax/include/cuda/experimental/__stf/internal/parallel_for_scope.cuh`
- `cudax/include/cuda/experimental/__stf/internal/context.cuh`
- `cudax/include/cuda/experimental/__stf/stream/stream_ctx.cuh`
- `cudax/include/cuda/experimental/__stf/graph/graph_ctx.cuh`

**Examples:**
- `cudax/examples/stf/fdtd_mgpu.cu` (FDTD multi-GPU)
- `cudax/examples/stf/heat_mgpu.cu` (Heat equation multi-GPU)
- `cudax/examples/stf/mandelbrot.cu`
- `cudax/examples/stf/jacobi.cu`

**Tests:**
- `cudax/test/stf/graph/epoch.cu`
- `cudax/test/stf/green_context/gc_grid.cu`
- `cudax/test/stf/stress/launch_vs_parallelfor.cu`

## Appendix B: Key Data Structures

### `pos4` (4D position)
Represents a position in a 4-dimensional space (x, y, z, t).

### `dim4` (4D dimensions)
Represents dimensions of a 4-dimensional space.

### `box<dimensions>` (N-dimensional bounding box)
Stores pairs of (begin, end) for each dimension.

### `exec_place_grid`
Grid of execution places with associated dimensions.

### `localized_array::metadata`
```cpp
struct metadata {
    CUmemGenericAllocationHandle alloc_handle;
    int dev;          // Device ID
    size_t size;      // Size in bytes
    size_t offset;    // Offset from base_ptr
};
```

## Appendix C: Terminology

- **Allocation granularity:** Minimum page size for VMM allocations (typically 2 MiB)
- **Composite data place:** Multi-device data place defined by grid + partitioner
- **Logical partition:** Element-based partition computed by partitioner
- **Physical allocation:** Page-based allocation in device memory
- **VMM:** Virtual Memory Management (CUDA Unified Virtual Addressing)
- **Linearized index:** 1D index into multidimensional array
- **Delinearization:** Converting linear index back to multidimensional coordinates
- **Affine data place:** Data place associated with an execution place
