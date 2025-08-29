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
 * @brief Definition of the `block_cyclic_partition` strategy
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

template <::std::ptrdiff_t which_dim, size_t tile_size, size_t dimensions>
class block_cyclic_shape
{
private:
  ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions> bounds_;
  size_t target_dim_;
  size_t place_id_;
  size_t nplaces_;
  size_t total_tiles_;
  size_t tiles_for_this_place_;

public:
  _CCCL_HOST_DEVICE block_cyclic_shape(
    const ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions>& bounds,
    size_t target_dim,
    size_t place_id,
    size_t nplaces)
    : bounds_(bounds), target_dim_(target_dim), place_id_(place_id), nplaces_(nplaces)
  {
    static_assert(tile_size > 0, "tile_size must be greater than 0");
    
    // Calculate total number of tiles in target dimension
    size_t dim_extent = bounds_[target_dim_].second - bounds_[target_dim_].first;
    total_tiles_ = (dim_extent + tile_size - 1) / tile_size;
    
    // Calculate how many tiles this place gets (cyclic distribution)
    tiles_for_this_place_ = 0;
    for (size_t tile_id = place_id_; tile_id < total_tiles_; tile_id += nplaces_) {
      tiles_for_this_place_++;
    }
  }

  _CCCL_HOST_DEVICE size_t size() const
  {
    if (tiles_for_this_place_ == 0) {
      return 0;
    }
    
    // Calculate total elements across all dimensions except target
    size_t other_dims_size = 1;
    for (size_t d = 0; d < dimensions; d++) {
      if (d != target_dim_) {
        other_dims_size *= (bounds_[d].second - bounds_[d].first);
      }
    }
    
    // Calculate elements in target dimension for this place
    size_t target_dim_elements = 0;
    size_t dim_begin = bounds_[target_dim_].first;
    size_t dim_end = bounds_[target_dim_].second;
    
    for (size_t tile_id = place_id_; tile_id < total_tiles_; tile_id += nplaces_) {
      size_t tile_start = dim_begin + tile_id * tile_size;
      size_t tile_end = ::std::min(dim_begin + (tile_id + 1) * tile_size, (size_t)dim_end);
      target_dim_elements += (tile_end - tile_start);
    }
    
    return other_dims_size * target_dim_elements;
  }

  class iterator
  {
  private:
    const block_cyclic_shape* parent_;
    ::std::array<::std::ptrdiff_t, dimensions> current_coords_;
    size_t current_tile_index_;
    size_t current_tile_offset_;
    bool at_end_;

    _CCCL_HOST_DEVICE void advance_to_next_valid_position()
    {
      if (at_end_) return;
      
      // Move to next position within current tile
      current_tile_offset_++;
      
      // Calculate current tile ID
      size_t current_tile_id = parent_->place_id_ + current_tile_index_ * parent_->nplaces_;
      
      if (current_tile_id >= parent_->total_tiles_) {
        at_end_ = true;
        return;
      }
      
      size_t tile_start = parent_->bounds_[parent_->target_dim_].first + current_tile_id * tile_size;
      size_t tile_end = ::std::min(
        parent_->bounds_[parent_->target_dim_].first + (current_tile_id + 1) * tile_size,
        (size_t)parent_->bounds_[parent_->target_dim_].second);
      
      // Check if we've reached the end of current tile
      if (current_tile_offset_ >= (tile_end - tile_start)) {
        current_tile_index_++;
        current_tile_offset_ = 0;
        
        // Check if we have more tiles
        size_t next_tile_id = parent_->place_id_ + current_tile_index_ * parent_->nplaces_;
        if (next_tile_id >= parent_->total_tiles_) {
          at_end_ = true;
          return;
        }
      }
      
      update_coordinates();
    }
    
    _CCCL_HOST_DEVICE void update_coordinates()
    {
      if (at_end_) return;
      
      // Calculate current tile ID and position within tile
      size_t current_tile_id = parent_->place_id_ + current_tile_index_ * parent_->nplaces_;
      size_t tile_start = parent_->bounds_[parent_->target_dim_].first + current_tile_id * tile_size;
      
      // Set target dimension coordinate
      current_coords_[parent_->target_dim_] = tile_start + current_tile_offset_;
      
      // Set other dimensions to their beginning values (will be incremented by outer loops)
      for (size_t d = 0; d < dimensions; d++) {
        if (d != parent_->target_dim_) {
          current_coords_[d] = parent_->bounds_[d].first;
        }
      }
    }

  public:
    _CCCL_HOST_DEVICE iterator(const block_cyclic_shape* parent, bool at_end = false)
      : parent_(parent), current_tile_index_(0), current_tile_offset_(0), at_end_(at_end)
    {
      if (!at_end_ && parent_->tiles_for_this_place_ > 0) {
        // Initialize to first position
        for (size_t d = 0; d < dimensions; d++) {
          current_coords_[d] = parent_->bounds_[d].first;
        }
        update_coordinates();
      } else {
        at_end_ = true;
      }
    }

    _CCCL_HOST_DEVICE auto& operator*() const
    {
      if constexpr (dimensions == 1)
      {
        return current_coords_[0];
      }
      else
      {
        return current_coords_;
      }
    }

    _CCCL_HOST_DEVICE iterator& operator++()
    {
      advance_to_next_valid_position();
      return *this;
    }

    _CCCL_HOST_DEVICE bool operator!=(const iterator& other) const
    {
      return at_end_ != other.at_end_;
    }
  };

  _CCCL_HOST_DEVICE iterator begin() const
  {
    return iterator(this, false);
  }

  _CCCL_HOST_DEVICE iterator end() const
  {
    return iterator(this, true);
  }

  // Interface compatibility with box for cyclic_partition::apply
  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_begin(size_t dim) const
  {
    return bounds_[dim].first;
  }

  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_end(size_t dim) const
  {
    return bounds_[dim].second;
  }

  // Interface compatibility with mdspan for cyclic_partition::apply
  static constexpr size_t rank() 
  {
    return dimensions;
  }

  _CCCL_HOST_DEVICE ::std::ptrdiff_t extent(size_t dim) const
  {
    return bounds_[dim].second - bounds_[dim].first;
  }
};

} // namespace reserved

template <::std::ptrdiff_t which_dim = -1, size_t tile_size = 256>
class block_cyclic_partition_custom
{
public:
  block_cyclic_partition_custom() = default;

  template <size_t dimensions>
  _CCCL_HOST_DEVICE static auto apply(const box<dimensions>& in, pos4 place_position, dim4 grid_dims)
  {
    static_assert(tile_size > 0, "tile_size must be greater than 0");
    
    ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions> bounds;
    size_t target_dim = (which_dim == -1) ? dimensions - 1 : size_t(which_dim);
    if (target_dim > dimensions - 1)
    {
      target_dim = dimensions - 1;
    }

    // Initialize bounds for all dimensions
    for (size_t d = 0; d < dimensions; d++)
    {
      bounds[d].first = in.get_begin(d);
      bounds[d].second = in.get_end(d);
    }

    return reserved::block_cyclic_shape<which_dim, tile_size, dimensions>(
      bounds, target_dim, place_position.x, grid_dims.x);
  }

  template <typename mdspan_shape_t>
  _CCCL_HOST_DEVICE static auto apply(const mdspan_shape_t& in, pos4 place_position, dim4 grid_dims)
  {
    static_assert(tile_size > 0, "tile_size must be greater than 0");
    
    constexpr size_t dimensions = mdspan_shape_t::rank();

    ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions> bounds;
    for (size_t d = 0; d < dimensions; d++)
    {
      bounds[d].first  = 0;
      bounds[d].second = in.extent(d);
    }

    // Determine target dimension
    size_t target_dim = (which_dim == -1) ? dimensions - 1 : size_t(which_dim);
    if (target_dim > dimensions - 1)
    {
      target_dim = dimensions - 1;
    }

    return reserved::block_cyclic_shape<which_dim, tile_size, dimensions>(
      bounds, target_dim, place_position.x, grid_dims.x);
  }

  _CCCL_HOST_DEVICE static pos4 get_executor(pos4 data_coords, dim4 data_dims, dim4 grid_dims)
  {
    static_assert(tile_size > 0, "tile_size must be greater than 0");
    
    // Find the target dimension
    size_t rank       = data_dims.get_rank();
    size_t target_dim = (which_dim == -1) ? rank - 1 : size_t(which_dim);
    if (target_dim >= rank)
    {
      target_dim = rank - 1;
    }

    size_t nplaces = grid_dims.x;
    
    // Get the coordinate in the selected dimension
    size_t c = data_coords.get(target_dim);
    
    // Calculate which tile this coordinate belongs to
    size_t tile_id = c / tile_size;
    
    // Return the executor using cyclic distribution
    return pos4(tile_id % nplaces);
  }
};

using block_cyclic_partition = block_cyclic_partition_custom<>;

} // namespace cuda::experimental::stf