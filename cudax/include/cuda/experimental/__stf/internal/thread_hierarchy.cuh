//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/cooperative_group_system.cuh>
#include <cuda/experimental/__stf/internal/interpreted_execution_policy.cuh>
#include <cuda/experimental/__stf/internal/slice.cuh>
#include <cuda/experimental/__stf/places/blocked_partition.cuh>
#include <cuda/experimental/__stf/places/cyclic_shape.cuh>
#include <cuda/experimental/__stf/places/static_partition.cuh>

namespace cuda::experimental::stf
{

template <auto... spec>
class thread_hierarchy
{
  static constexpr size_t depth = [](auto x, auto y, auto...) {
    static_assert(::std::is_same_v<decltype(x), bool>, "You must use bool for the odd arguments of thread_hierarchy.");
    static_assert(::std::is_same_v<decltype(y), size_t>,
                  "You must use size_t for the even arguments of thread_hierarchy.");
    return sizeof...(spec) / 2;
  }(spec..., false, size_t(0));

  template <auto... subspec>
  struct inner_t;
  template <auto x, auto y, auto... subspec>
  struct inner_t<x, y, subspec...>
  {
    using type = thread_hierarchy<subspec...>;
  };

public:
  _CCCL_HOST_DEVICE thread_hierarchy()
  {
    size_t data[] = {spec...};
    for(size_t i = 0; i < depth; ++i) {
        level_sizes[i] = data[1 + 2 * i];
    }
  }

  template <auto...>
  friend class thread_hierarchy;

  template <bool outer_sync, size_t outer_width>
  _CCCL_HOST_DEVICE thread_hierarchy(const thread_hierarchy<outer_sync, outer_width, spec...>& outer)
      : devid(outer.devid)
      , launch_config(outer.launch_config)
      , cg_system(outer.cg_system)
      , device_tmp(outer.device_tmp)
      , system_tmp(outer.system_tmp)
  {
    for (size_t i : each(0, depth))
    {
      level_sizes[i] = outer.level_sizes[i + 1];
      mem_sizes[i]   = outer.mem_sizes[i + 1];
    }
  }

  thread_hierarchy(int devid, interpreted_execution_policy<spec...>& p)
      : devid(devid)
  {
    launch_config = p.get_config();
    cg_system = p.cg_system;
    size_t i = 0;
    for (auto& l : p.get_levels())
    {
      level_sizes[i] = l.width();
      assert(l.width() > 0);
      mem_sizes[i] = l.get_mem();
      i++;
    }
  }

  static inline constexpr size_t static_width(size_t level)
  {
    size_t data[] = {spec...};
    return data[1 + 2 * level];
  }

  _CCCL_HOST_DEVICE const ::std::array<size_t, 3>& get_config() const
  {
    return launch_config;
  }

  _CCCL_HOST_DEVICE size_t rank([[maybe_unused]] int level, [[maybe_unused]] int root_level) const
  {
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (int tid = threadIdx.x; int bid = blockIdx.x;
       const int nblocks  = launch_config[1];
       const int nthreads = launch_config[2];
       int global_id = tid + bid * nthreads + devid * nblocks * nthreads;
       size_t level_effective_size      = 1;
       size_t root_level_effective_size = 1;
       if constexpr (depth > 0) {
         for (size_t l = level + 1; l < depth; l++)
         {
           level_effective_size *= level_sizes[l];
         }
         for (size_t l = root_level + 1; l < depth; l++)
         {
           root_level_effective_size *= level_sizes[l];
         }
       }
       return (global_id % root_level_effective_size)
       / level_effective_size;),
      (return 0;))
    _CCCL_UNREACHABLE();
  }

  _CCCL_HOST_DEVICE size_t size([[maybe_unused]] int level, [[maybe_unused]] int root_level) const
  {
    if constexpr (depth == 0)
    {
      return 1;
    }
    assert(root_level < level);
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (return 1;),
      (
        size_t s = 1; for (int l = root_level; l < level; l++) { s *= level_sizes[l + 1]; }
        return s;))
    _CCCL_UNREACHABLE();
  }

  _CCCL_HOST_DEVICE size_t size(int level = int(depth) - 1) const
  {
    return size(level, -1);
  }

  _CCCL_HOST_DEVICE size_t rank(int level = int(depth) - 1) const
  {
    return rank(level, -1);
  }

  _CCCL_HOST_DEVICE void sync([[maybe_unused]] int level = 0)
  {
    assert(level >= 0);
    assert(level < depth);
    NV_IF_TARGET(
      NV_IS_DEVICE,
      (
        assert(may_sync(level));
        size_t target_size = 1;
        for (int l = level; l < depth; l++) { target_size *= level_sizes[l]; }
        size_t block_scope_size = launch_config[2];
        if (target_size == block_scope_size) {
          cooperative_groups::this_thread_block().sync();
          return;
        }
        size_t device_scope_size = block_scope_size * launch_config[1];
        if (target_size == device_scope_size) {
          cooperative_groups::this_grid().sync();
          return;
        }
        size_t ndevs             = launch_config[0];
        size_t system_scope_size = device_scope_size * ndevs;
        if (target_size == system_scope_size) {
          cg_system.sync(devid, ndevs);
          return;
        }
        assert(0);))
  }

  template <typename T, typename... Others>
  auto remove_first_tuple_element(const ::std::tuple<T, Others...>& t)
  {
    return ::std::make_tuple(::std::get<Others>(t)...);
  }

  template <typename shape_t, typename P, typename... sub_partitions>
  _CCCL_HOST_DEVICE auto apply_partition(const shape_t& s, const ::std::tuple<P, sub_partitions...>& t) const
  {
    auto s0         = P::apply(s, pos4(rank(0)), dim4(size(0)));
    auto sans_first = make_tuple_indexwise<sizeof...(sub_partitions)>([&](auto index) {
      return ::std::get<index + 1>(t);
    });
    if constexpr (sizeof...(sub_partitions))
    {
      return inner().apply_partition(s0, sans_first);
    }
    else
    {
      return s0;
    }
  }

  template <typename shape_t>
  _CCCL_HOST_DEVICE auto apply_partition(const shape_t& s) const
  {
    if constexpr (depth == 1)
    {
      return cyclic_partition::apply(s, pos4(rank(0)), dim4(size(0)));
    }
    else
    {
      auto s0 = blocked_partition::apply(s, pos4(rank(0)), dim4(size(0)));
      return inner().apply_partition(s0);
    }
  }

  _CCCL_HOST_DEVICE auto inner() const
  {
    return typename inner_t<spec...>::type(*this);
  }

  template <typename T>
  _CCCL_HOST_DEVICE slice<T> storage(int level)
  {
    assert(level >= 0);
    assert(level < depth);
    size_t target_size = 1;
    for (int l = level; l < depth; l++)
    {
      target_size *= level_sizes[l];
    }
    NV_IF_TARGET(
      NV_IS_DEVICE,
      (
        size_t nelems = mem_sizes[level] / sizeof(T);
        size_t block_scope_size = launch_config[2];
        if (target_size == block_scope_size) {
          extern __shared__ T dyn_buffer[];
          return make_slice(&dyn_buffer[0], nelems);
        }
        size_t device_scope_size = block_scope_size * launch_config[1];
        if (target_size == device_scope_size) {
          return make_slice(static_cast<T*>(device_tmp), nelems);
        }
        size_t ndevs             = launch_config[0];
        size_t system_scope_size = device_scope_size * ndevs;
        if (target_size == system_scope_size) {
          return make_slice(static_cast<T*>(system_tmp), nelems);
        }))
    assert(!"Unsupported configuration : memory must be a scope boundaries");
    return make_slice(static_cast<T*>(nullptr), 0);
  }

  void set_device_tmp(void* addr)
  {
    device_tmp = addr;
  }

  void set_system_tmp(void* addr)
  {
    system_tmp = addr;
  }

  void set_devid(int d)
  {
    devid = d;
  }

private:
  int devid = -1;
  ::std::array<size_t, 3> launch_config = {};
  reserved::cooperative_group_system cg_system;
  ::std::array<size_t, depth> level_sizes = {};
  ::std::array<size_t, depth> mem_sizes = {};
  void* device_tmp = nullptr;
  void* system_tmp = nullptr;

  template <size_t Idx>
  _CCCL_HOST_DEVICE bool may_sync_impl(int) const
  {
    return false;
  }

  template <size_t Idx, bool sync, size_t w, auto... remaining>
  _CCCL_HOST_DEVICE bool may_sync_impl(int level) const
  {
    static_assert(Idx < sizeof...(spec) / 2);
    return (level == Idx) ? sync : may_sync_impl<Idx + 1, remaining...>(level);
  }

  _CCCL_HOST_DEVICE bool may_sync(int level) const
  {
    return may_sync_impl<0, spec...>(level);
  }

public:

public:
  template <typename... Partitioners, typename Shape>
  _CCCL_HOST_DEVICE auto apply_static_partition(const Shape& shape) const
  {
    using PartitionerTuple = std::tuple<Partitioners...>;
    
    // 1. The composer gathers a flat tuple of all parameters from all levels.
    auto all_params = static_partition::StaticPartitionComposer<0, decltype(*this), Shape, PartitionerTuple>::compose(*this, shape, PartitionerTuple{});

    // 2. A single, flat map object is created from these parameters.
    return static_partition::StaticPartitionMap<PartitionerTuple, decltype(all_params), Shape, decltype(*this)>(all_params, shape, *this);
  }
};

} // end namespace cuda::experimental::stf