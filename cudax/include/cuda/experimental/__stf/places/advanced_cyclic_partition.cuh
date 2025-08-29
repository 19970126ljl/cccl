//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Definition of the `advanced_cyclic_partition` strategy for non-contiguous tile-aware shapes
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/places/cyclic_shape.cuh>
#include <cuda/experimental/__stf/places/places.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>

namespace cuda::experimental::stf
{

namespace reserved
{

template <::std::ptrdiff_t which_dim, typename base_shape_t>
class advanced_cyclic_shape
{
private:
  base_shape_t base_shape_;
  size_t place_id_;
  size_t nplaces_;
  
public:
  _CCCL_HOST_DEVICE advanced_cyclic_shape(
    const base_shape_t& base_shape,
    size_t place_id,
    size_t nplaces)
    : base_shape_(base_shape), place_id_(place_id), nplaces_(nplaces)
  {
  }

  _CCCL_HOST_DEVICE size_t size() const
  {
    size_t total_size = base_shape_.size();
    // Cyclic distribution: some places get ceil(total/nplaces), others get floor
    size_t base_per_place = total_size / nplaces_;
    size_t remainder = total_size % nplaces_;
    
    if (place_id_ < remainder) {
      return base_per_place + 1;
    } else {
      return base_per_place;
    }
  }

  class iterator
  {
  private:
    typename base_shape_t::iterator current_iter_;
    typename base_shape_t::iterator end_iter_;
    size_t place_id_;
    size_t nplaces_;
    bool at_end_;

  public:
    _CCCL_HOST_DEVICE iterator(
      typename base_shape_t::iterator start_iter,
      typename base_shape_t::iterator end_iter,
      size_t place_id,
      size_t nplaces,
      bool at_end = false)
      : current_iter_(start_iter), end_iter_(end_iter), 
        place_id_(place_id), nplaces_(nplaces), at_end_(at_end)
    {
      if (!at_end_) {
        // Skip to the first element for this place (place_id_ steps from start)
        for (size_t skip = 0; skip < place_id_; ++skip) {
          if (!(current_iter_ != end_iter_)) {
            at_end_ = true;
            return;
          }
          ++current_iter_;
        }
        
        // Check if we've already reached the end after initial positioning
        if (!(current_iter_ != end_iter_)) {
          at_end_ = true;
        }
      }
    }

    _CCCL_HOST_DEVICE auto& operator*() const
    {
      return *current_iter_;
    }

    _CCCL_HOST_DEVICE iterator& operator++()
    {
      if (!at_end_) {
        // Move to next element for this place: advance by nplaces_ positions
        for (size_t skip = 0; skip < nplaces_; ++skip) {
          ++current_iter_;
          if (!(current_iter_ != end_iter_)) {
            at_end_ = true;
            return *this;
          }
        }
      }
      return *this;
    }

    _CCCL_HOST_DEVICE bool operator!=(const iterator& other) const
    {
      // Simple comparison based on end status
      return at_end_ != other.at_end_;
    }
  };

  _CCCL_HOST_DEVICE iterator begin() const
  {
    return iterator(base_shape_.begin(), base_shape_.end(), place_id_, nplaces_, false);
  }

  _CCCL_HOST_DEVICE iterator end() const
  {
    return iterator(base_shape_.end(), base_shape_.end(), place_id_, nplaces_, true);
  }

  // Interface compatibility with box for further partitioning
  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_begin(size_t dim) const
  {
    return 0;
  }

  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_end(size_t dim) const
  {
    return size();
  }

  // Interface compatibility with mdspan for further partitioning
  static constexpr size_t rank() 
  {
    return 1;
  }

  _CCCL_HOST_DEVICE ::std::ptrdiff_t extent(size_t dim) const
  {
    return size();
  }
};

} // namespace reserved

template <::std::ptrdiff_t which_dim = -1>
class advanced_cyclic_partition
{
public:
  advanced_cyclic_partition() = default;

  template <typename base_shape_t>
  _CCCL_HOST_DEVICE static auto apply(const base_shape_t& in, pos4 place_position, dim4 grid_dims)
  {
    return reserved::advanced_cyclic_shape<which_dim, base_shape_t>(
      in, place_position.x, grid_dims.x);
  }

  _CCCL_HOST_DEVICE static pos4 get_executor(pos4 data_coords, dim4 data_dims, dim4 grid_dims)
  {
    // Use cyclic distribution: element index % nplaces
    size_t rank = data_dims.get_rank();
    size_t target_dim = (which_dim == -1) ? rank - 1 : size_t(which_dim);
    if (target_dim >= rank) {
      target_dim = rank - 1;
    }

    size_t nplaces = grid_dims.x;
    size_t c = data_coords.get(target_dim);
    
    return pos4(c % nplaces);
  }
};

} // namespace cuda::experimental::stf