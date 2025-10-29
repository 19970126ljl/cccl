# Partitioner Quick Reference Card

## What Are Partitioners?

Partitioners divide multi-dimensional data across multiple execution places (GPUs) in cudax STF.

## Built-in Partitioners

### `blocked_partition()`
Divides data into contiguous blocks along one dimension (default: last dimension).

```cpp
ctx.parallel_for(blocked_partition(), exec_place::all_devices(), shape, deps...)
```

**Use when:** You want simple contiguous partitioning, good for stencil codes.

### `tiled_partition<tile_size>()`
Distributes fixed-size tiles cyclically across devices.

```cpp
ctx.parallel_for(tiled_partition<1024>(), exec_place::all_devices(), shape, deps...)
```

**Use when:** You want load balancing with cyclic distribution.

### `cyclic_shape()`
Round-robin distribution of elements.

**Use when:** You have highly irregular workloads.

## How to Use a Partitioner

### Basic Pattern
```cpp
context ctx;
exec_place where = exec_place::all_devices();  // or n_devices(4), etc.
auto shape = shape_of<slice<double, 3>>(NX, NY, NZ);
auto data = ctx.logical_data(shape);

ctx.parallel_for(
    blocked_partition(),  // ← Your partitioner
    where,                // ← Execution grid (devices)
    shape,                // ← Data shape
    data.rw()             // ← Dependencies
) ->* [=] _CCCL_DEVICE (size_t i, size_t j, size_t k, auto d) {
    // Your kernel code
};
```

## Custom Partitioner Interface

To create a custom partitioner, implement:

```cpp
class my_partitioner {
public:
    // Required: Compute sub-shape for a given place
    template <typename shape_t>
    static auto apply(const shape_t& in, pos4 place_position, dim4 grid_dims) {
        // Return a box describing this place's partition
    }
    
    // Required: Determine which place owns a data element
    static pos4 get_executor(pos4 data_coords, dim4 data_dims, dim4 grid_dims) {
        // Return place position that should process this element
    }
};
```

## Common Patterns

### 1D Blocked Distribution (2 GPUs)
```cpp
// Data: [0, 1000)
// GPU 0 gets: [0, 500)
// GPU 1 gets: [500, 1000)
ctx.parallel_for(blocked_partition(), exec_place::n_devices(2), 
                 box({0, 1000}), deps...)
```

### 3D Blocked Distribution (4 GPUs, split Z-axis)
```cpp
// Data: 100×100×400
// GPU 0 gets: [:, :, 0:100)
// GPU 1 gets: [:, :, 100:200)
// GPU 2 gets: [:, :, 200:300)
// GPU 3 gets: [:, :, 300:400)
ctx.parallel_for(blocked_partition(), exec_place::n_devices(4),
                 shape_of<slice<double, 3>>(100, 100, 400), deps...)
```

### Custom Dimension Selection
```cpp
// Partition along X dimension instead of default (Z)
blocked_partition_custom<0>  // 0 = X axis
```

## Memory Implications (IMPORTANT!)

### Current Behavior
- Memory allocated in **2 MiB pages** (hardware constraint)
- Partitioners work with **logical elements**
- **Mismatch can cause imbalanced memory distribution**

### Example Problem
```cpp
// 8 MB total on 2 GPUs
shape_of<slice<double, 3>>(100, 100, 100)  // 8,000,000 bytes

// Expected: 4 MB per GPU
// Reality: May get 2 MB vs 6 MB due to page boundaries!
```

**See `padded_blocked_partitioner_investigation.md` for details.**

## Performance Tips

### DO ✅
- Use blocked partition for stencil computations (good locality)
- Partition largest dimension for better load balance
- Align buffer sizes to 2 MiB boundaries when possible

### DON'T ❌
- Don't assume perfect memory balance with non-2MiB-aligned sizes
- Don't partition tiny dimensions (increases boundary overhead)
- Don't mix partitioners for same data without understanding implications

## Debugging Partitioners

### Check What Each Device Gets
```cpp
ctx.parallel_for(blocked_partition(), where, shape, data.read())
    ->*[=] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto d) {
        if (i == 0 && j == 0 && k == 0) {
            printf("Device processing first element\n");
        }
    };
```

### Verify Partition Boundaries
```cpp
// Host-side test
auto sub_shape = blocked_partition::apply(
    box({0, 100}, {0, 100}, {0, 100}),
    pos4(0, 0, 0, 0),  // First device
    dim4(2, 1, 1, 1)   // 2 devices
);
// Print sub_shape bounds
```

## Common Issues and Solutions

### Issue: Imbalanced Execution Time
**Symptom:** One GPU finishes much faster than others.

**Possible causes:**
1. Data not divisible evenly by number of GPUs
2. Irregular computation (more work in one partition)
3. Memory imbalance (see investigation report)

**Solutions:**
- Use tiled partition for better load balancing
- Increase buffer size to be multiple of (2 MiB × num_GPUs)
- Profile to identify hotspots

### Issue: Unexpected Memory Usage
**Symptom:** One GPU running out of memory while others have plenty.

**Likely cause:** Page alignment issue (see section 4 of investigation).

**Solutions:**
- Pad buffer sizes to 2 MiB boundaries
- Use page-aligned partitioner (when available)
- Reduce buffer size to fit worst-case distribution

### Issue: Slow Inter-GPU Communication
**Symptom:** Performance doesn't scale with GPU count.

**Possible causes:**
1. Partitioner creating too many boundary interactions
2. Lack of NVLink between GPUs
3. Frequent synchronization

**Solutions:**
- Ensure partitioner minimizes cross-device dependencies
- Check that peer access is enabled
- Batch operations to reduce sync frequency

## Advanced: Partition for Multi-GPU Stencil

For codes like FDTD or heat equation with ghost zones:

```cpp
// Main computation
box interior({1, NX-1}, {1, NY-1}, {1, NZ-1});
ctx.parallel_for(blocked_partition(), where, interior, data.rw())
    ->*[=] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto d) {
        // Stencil computation uses d(i±1, j±1, k±1)
    };

// Boundary update
box boundary({0, NX}, {0, NY}, {0, 1});  // Z=0 face
ctx.parallel_for(blocked_partition(), where, boundary, data.write())
    ->*[=] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto d) {
        // Boundary condition
    };
```

## Files and Examples

### Core Implementation
- `cudax/include/cuda/experimental/__stf/places/blocked_partition.cuh`
- `cudax/include/cuda/experimental/__stf/places/tiled_partition.cuh`

### Examples
- `cudax/examples/stf/fdtd_mgpu.cu` - FDTD with blocked partition
- `cudax/examples/stf/heat_mgpu.cu` - Heat equation multi-GPU
- `cudax/examples/stf/jacobi.cu` - Jacobi iteration

### Documentation
- `padded_blocked_partitioner_investigation.md` - Deep dive into memory issues
- `partitioner_memory_flow_diagram.txt` - Visual explanation

## Further Reading

For complete details on the 2 MiB page alignment issue and proposed solutions, see:
- **Investigation Report:** `padded_blocked_partitioner_investigation.md`
- **Flow Diagram:** `partitioner_memory_flow_diagram.txt`
- **Summary:** `PARTITIONER_AUDIT_README.md`
