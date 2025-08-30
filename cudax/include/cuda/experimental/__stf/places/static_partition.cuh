//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/experimental/__stf/utility/tuple_utility.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>

namespace cuda::experimental::stf {

// Forward declaration
template <auto... spec>
class thread_hierarchy;

namespace detail {

// Gets the number of parameters for a partitioner
template <typename P>
struct param_count {
    static constexpr size_t value = std::tuple_size_v<typename P::params_t>;
};

// Calculates the starting index for a partitioner's params in the flattened tuple
template <size_t I, typename PartitionTuple>
struct tuple_param_offset {
    static constexpr size_t value =
        tuple_param_offset<I - 1, PartitionTuple>::value + param_count<std::tuple_element_t<I - 1, PartitionTuple>>::value;
};

template <typename PartitionTuple>
struct tuple_param_offset<0, PartitionTuple> {
    static constexpr size_t value = 0;
};

// Slices a tuple
template <size_t Offset, size_t... I, typename Tuple>
_CCCL_HOST_DEVICE constexpr auto slice_tuple_impl(const Tuple& t, std::index_sequence<I...>) {
    return std::make_tuple(std::get<Offset + I>(t)...);
}

template <size_t Offset, size_t Count, typename Tuple>
_CCCL_HOST_DEVICE constexpr auto slice_tuple(const Tuple& t) {
    return slice_tuple_impl<Offset>(t, std::make_index_sequence<Count>{});
}

} // namespace detail

// --- Concrete Static Partitioner Implementations (V3 Design) ---

class StaticBlock {
public:
    // Parameters needed for this partitioner's calculation
    using params_t = std::tuple<size_t, size_t, size_t>; // total_size, rank, num_partitions

    // Returns the parameters for this level
    template <typename Shape>
    _CCCL_HOST_DEVICE static constexpr params_t get_static_params(const Shape& shape, size_t rank, size_t num_partitions) {
        return std::make_tuple(shape.extent(0), rank, num_partitions);
    }

    // Calculates the sub-shape for the next level of partitioning
    template <typename Shape>
    _CCCL_HOST_DEVICE static constexpr auto get_next_shape(const Shape& shape, size_t rank, size_t num_partitions) {
        size_t total_size = shape.extent(0);
        size_t begin = (total_size * rank) / num_partitions;
        size_t end = (total_size * (rank + 1)) / num_partitions;
        return box<1>{end - begin};
    }

    // Applies the forward mapping for this level
    _CCCL_HOST_DEVICE static constexpr size_t apply_map(size_t logical_index, const params_t& params) {
        auto [total_size, rank, num_partitions] = params;
        size_t begin = (total_size * rank) / num_partitions;
        return begin + logical_index;
    }

    // Calculates the size of the partition for this level
    _CCCL_HOST_DEVICE static constexpr size_t get_size(const params_t& params) {
        auto [total_size, rank, num_partitions] = params;
        size_t begin = (total_size * rank) / num_partitions;
        size_t end = (total_size * (rank + 1)) / num_partitions;
        return end - begin;
    }
};

class StaticCyclic {
public:
    // Parameters needed for this partitioner's calculation
    using params_t = std::tuple<size_t, size_t, size_t>; // start_offset, total_elements, stride

    // Returns the parameters for this level
    template <typename Shape>
    _CCCL_HOST_DEVICE static constexpr params_t get_static_params(const Shape& shape, size_t rank, size_t stride) {
        return std::make_tuple(rank, shape.extent(0), stride);
    }

    // Calculates the sub-shape for the next level of partitioning
    template <typename Shape>
    _CCCL_HOST_DEVICE static constexpr auto get_next_shape(const Shape& shape, size_t rank, size_t stride) {
        size_t total_elements = shape.extent(0);
        if (total_elements <= rank) return box<1>{0};
        return box<1>{(total_elements - rank + stride - 1) / stride};
    }

    // Applies the forward mapping for this level
    _CCCL_HOST_DEVICE static constexpr size_t apply_map(size_t logical_index, const params_t& params) {
        auto [start_offset, total_elements, stride] = params;
        return start_offset + logical_index * stride;
    }

    // Calculates the size of the partition for this level
    _CCCL_HOST_DEVICE static constexpr size_t get_size(const params_t& params) {
        auto [start_offset, total_elements, stride] = params;
        if (total_elements <= start_offset) return 0;
        return (total_elements - start_offset + stride - 1) / stride;
    }
};


namespace static_partition {

// Recursive helper to apply the mapping logic by iterating through the partitioner list.
// It calls the innermost partitioner first, then wraps it with the next one, and so on.
template <size_t I, typename PartitionTuple, typename ParamTuple>
struct Mapper {
    _CCCL_HOST_DEVICE static constexpr size_t apply(size_t logical_index, const ParamTuple& all_params) {
        if constexpr (I == std::tuple_size_v<PartitionTuple>) {
            // Base case: we've recursed through all partitioners. Return the index.
            return logical_index;
        } else {
            // Recursive step
            // 1. Get parameters for the current level (I)
            constexpr size_t param_offset = detail::tuple_param_offset<I, PartitionTuple>::value;
            using P = std::tuple_element_t<I, PartitionTuple>;
            auto level_params = detail::slice_tuple<param_offset, std::tuple_size_v<typename P::params_t>>(all_params);

            // 2. Recursively call the mapper for the inner level (I+1)
            size_t inner_physical_index = Mapper<I + 1, PartitionTuple, ParamTuple>::apply(logical_index, all_params);

            // 3. Apply the current level's mapping function to the result from the inner level
            return P::apply_map(inner_physical_index, level_params);
        }
    }
};

// Helper to calculate the size of the innermost partition.
template <size_t I, typename PartitionTuple, typename Shape>
struct SizeCalculator {
     _CCCL_HOST_DEVICE static constexpr size_t get(const Shape& shape, const PartitionTuple&, const auto& th) {
        if constexpr (I == std::tuple_size_v<PartitionTuple>) {
            return shape.extent(0);
        } else {
            using P = std::tuple_element_t<I, PartitionTuple>;
            auto next_shape = P::get_next_shape(shape, th.rank(0), th.size(0));
            return SizeCalculator<I + 1, PartitionTuple, decltype(next_shape)>::get(next_shape, PartitionTuple{}, th.inner());
        }
     }
};

// The final, flat map object that holds all parameters for the combined calculation.
template <typename PartitionTuple, typename ParamTuple, typename Shape, typename Hierarchy>
class StaticPartitionMap {
private:
  ParamTuple params_;
  Shape shape_;
  Hierarchy th_;

public:
  _CCCL_HOST_DEVICE constexpr StaticPartitionMap(ParamTuple params, Shape shape, Hierarchy th) : params_(params), shape_(shape), th_(th) {}

  _CCCL_HOST_DEVICE size_t size() const {
    return SizeCalculator<0, PartitionTuple, Shape>::get(shape_, PartitionTuple{}, th_);
  }

  _CCCL_HOST_DEVICE size_t operator()(size_t logical_index) const {
    return Mapper<0, PartitionTuple, ParamTuple>::apply(logical_index, params_);
  }
};

// The composer's job is to gather parameters from each level.
template <size_t I, typename Hierarchy, typename Shape, typename PartitionTuple>
struct StaticPartitionComposer {
    _CCCL_HOST_DEVICE static constexpr auto compose(const Hierarchy& th, const Shape& shape, const PartitionTuple& p_tuple) {
        if constexpr (I == std::tuple_size_v<PartitionTuple>) {
            // Base Case: Return an empty tuple of all collected parameters.
            return std::make_tuple();
        } else {
            // Recursive Step:
            using P = std::tuple_element_t<I, PartitionTuple>;

            // 1. Get the parameters for the current level.
            auto current_level_params = P::get_static_params(shape, th.rank(0), th.size(0));

            // 2. Calculate the shape for the next level.
            auto next_shape = P::get_next_shape(shape, th.rank(0), th.size(0));

            // 3. Recurse and prepend the current level's parameters.
            auto inner_params = StaticPartitionComposer<I + 1, decltype(th.inner()), decltype(next_shape), PartitionTuple>::compose(th.inner(), next_shape, p_tuple);
            return std::tuple_cat(current_level_params, inner_params);
        }
    }
};

} // namespace static_partition
} // namespace cuda::experimental::stf
