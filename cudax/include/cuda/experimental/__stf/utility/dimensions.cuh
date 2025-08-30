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

#include <cuda/experimental/__stf/utility/cuda_attributes.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

namespace cuda::experimental::stf
{

class pos4
{
public:
  constexpr pos4() = default;

  template <typename Integral>
  _CCCL_HOST_DEVICE constexpr explicit pos4(Integral x, Integral y = 0, Integral z = 0, Integral t = 0)
      : x(static_cast<int>(x))
      , y(static_cast<int>(y))
      , z(static_cast<int>(z))
      , t(static_cast<int>(t))
  {}

  _CCCL_HOST_DEVICE constexpr int get(size_t axis_id) const
  {
    switch (axis_id)
    {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        assert(axis_id == 3);
        return t;
    }
  }

  _CCCL_HOST_DEVICE constexpr int operator()(int axis_id) const
  {
    return get(axis_id);
  }

  _CCCL_HOST_DEVICE constexpr bool operator<(const pos4& rhs) const
  {
    if (x != rhs.x)
    {
      return x < rhs.x;
    }
    if (y != rhs.y)
    {
      return y < rhs.y;
    }
    if (z != rhs.z)
    {
      return z < rhs.z;
    }
    return t < rhs.t;
  }

  _CCCL_HOST_DEVICE constexpr bool operator==(const pos4& rhs) const
  {
    return x == rhs.x && y == rhs.y && z == rhs.z && t == rhs.t;
  }

  ::std::string to_string() const
  {
    return ::std::string("pos4(" + ::std::to_string(x) + "," + ::std::to_string(y) + "," + ::std::to_string(z) + ","
                         + ::std::to_string(t) + ")");
  }

  int x = 0;
  int y = 0;
  int z = 0;
  int t = 0;
};

class dim4 : public pos4
{
public:
  dim4() = default;

  _CCCL_HOST_DEVICE constexpr explicit dim4(int x, int y = 1, int z = 1, int t = 1)
      : pos4(x, y, z, t)
  {}

  _CCCL_HOST_DEVICE constexpr size_t size() const
  {
    const ::std::ptrdiff_t result = ::std::ptrdiff_t(x) * y * z * t;
    assert(result >= 0);
    return result;
  }

  _CCCL_HOST_DEVICE static constexpr dim4 min(const dim4& a, const dim4& b)
  {
    return dim4(::std::min(a.x, b.x), ::std::min(a.y, b.y), ::std::min(a.z, b.z), ::std::min(a.t, b.t));
  }

  _CCCL_HOST_DEVICE constexpr size_t get_index(const pos4& p) const
  {
    assert(p.get(0) <= x);
    assert(p.get(1) <= y);
    assert(p.get(2) <= z);
    assert(p.get(3) <= t);
    size_t index = p.get(0) + x * (p.get(1) + y * (p.get(2) + p.get(3) * z));
    return index;
  }

  _CCCL_HOST_DEVICE constexpr size_t get_rank() const
  {
    if (t > 1)
    {
      return 3;
    }
    if (z > 1)
    {
      return 2;
    }
    if (y > 1)
    {
      return 1;
    }

    return 0;
  }
};

template <size_t dimensions>
class box
{
public:
  template <typename Int1, typename Int2>
  _CCCL_HOST_DEVICE box(const ::std::array<::std::pair<Int1, Int2>, dimensions>& s)
      : s(s)
  {}

  template <typename Int>
  _CCCL_HOST_DEVICE box(const ::std::array<Int, dimensions>& sizes)
  {
    for (size_t ind : each(0, dimensions))
    {
      s[ind].first  = 0;
      s[ind].second = sizes[ind];
      if constexpr (::std::is_signed_v<Int>)
      {
        _CCCL_ASSERT(sizes[ind] >= 0, "Invalid shape.");
      }
    }
  }

  template <typename... Int>
  _CCCL_HOST_DEVICE box(Int... args)
  {
    static_assert(sizeof...(Int) == dimensions, "Number of dimensions must match");
    each_in_pack(
      [&](auto i, const auto& e) {
        if constexpr (::std::is_arithmetic_v<::std::remove_reference_t<decltype(e)>>)
        {
          s[i].first  = 0;
          s[i].second = e;
        }
        else
        {
          s[i].first  = e.first;
          s[i].second = e.second;
        }
      },
      args...);
  }

  template <typename... E>
  _CCCL_HOST_DEVICE box(::std::initializer_list<E>... args)
  {
    static_assert(sizeof...(E) == dimensions, "Number of dimensions must match");
    each_in_pack(
      [&](auto i, auto&& e) {
        _CCCL_ASSERT((e.size() == 1 || e.size() == 2), "Invalid arguments for box.");
        if (e.size() > 1)
        {
          s[i].first  = *e.begin();
          s[i].second = e.begin()[1];
        }
        else
        {
          s[i].first  = 0;
          s[i].second = *e.begin();
        }
      },
      args...);
  }

  _CCCL_HOST_DEVICE void print()
  {
    printf("EXPLICIT SHAPE\n");
    for (size_t ind = 0; ind < dimensions; ind++)
    {
      assert(s[ind].first <= s[ind].second);
      printf("    %ld -> %ld\n", s[ind].first, s[ind].second);
    }
  }

  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_extent(size_t dim) const
  {
    return s[dim].second - s[dim].first;
  }

  _CCCL_HOST_DEVICE ::std::ptrdiff_t extent(size_t dim) const
  {
    return get_extent(dim);
  }

  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_begin(size_t dim) const
  {
    return s[dim].first;
  }

  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_end(size_t dim) const
  {
    return s[dim].second;
  }

  _CCCL_HOST_DEVICE ::std::ptrdiff_t size() const
  {
    if constexpr (dimensions == 1)
    {
      return s[0].second - s[0].first;
    }
    else
    {
      size_t res = 1;
      for (size_t d = 0; d < dimensions; d++)
      {
        res *= get_extent(d);
      }
      return res;
    }
  }

  _CCCL_HOST_DEVICE constexpr size_t get_rank() const
  {
    return dimensions;
  }

  class iterator
  {
  private:
    box iterated;
    ::std::array<::std::ptrdiff_t, dimensions> current;

  public:
    _CCCL_HOST_DEVICE iterator(const box& b, bool at_end = false)
        : iterated(b)
    {
      if (at_end)
      {
        for (size_t i = 0; i < dimensions; ++i)
        {
          current[i] = iterated.get_end(i);
        }
      }
      else
      {
        for (size_t i = 0; i < dimensions; ++i)
        {
          current[i] = iterated.get_begin(i);
        }
      }
    }

    _CCCL_HOST_DEVICE auto& operator*()
    {
      if constexpr (dimensions == 1UL)
      {
        return current[0];
      }
      else
      {
        return current;
      }
    }

    _CCCL_HOST_DEVICE iterator& operator++()
    {
      if constexpr (dimensions == 1UL)
      {
        current[0]++;
      }
      else
      {
        for (size_t i : each(0, dimensions))
        {
          _CCCL_ASSERT(current[i] < iterated.get_end(i), "Attempt to increment past the end.");
          if (++current[i] < iterated.get_end(i))
          {
            for (size_t j : each(0, i))
            {
              current[j] = iterated.get_begin(j);
            }
            break;
          }
        }
      }
      return *this;
    }

    _CCCL_HOST_DEVICE bool operator==(const iterator& rhs) const
    {
      _CCCL_ASSERT(iterated == rhs.iterated, "Cannot compare iterators in different boxes.");
      for (auto i : each(0, dimensions))
      {
        if (current[i] != rhs.current[i])
        {
          return false;
        }
      }
      return true;
    }

    _CCCL_HOST_DEVICE bool operator!=(const iterator& other) const
    {
      return !(*this == other);
    }
  };

  _CCCL_HOST_DEVICE iterator begin()
  {
    return iterator(*this);
  }

  _CCCL_HOST_DEVICE iterator end()
  {
    return iterator(*this, true);
  }

  _CCCL_HOST_DEVICE bool operator==(const box& rhs) const
  {
    for (size_t i : each(0, dimensions))
    {
      if (get_begin(i) != rhs.get_begin(i) || get_end(i) != rhs.get_end(i))
      {
        return false;
      }
    }
    return true;
  }

  _CCCL_HOST_DEVICE bool operator!=(const box& rhs) const
  {
    return !(*this == rhs);
  }

  using coords_t = array_tuple<size_t, dimensions>;

  _CCCL_HOST_DEVICE coords_t index_to_coords(size_t index) const
  {
    CUDASTF_NO_DEVICE_STACK
    return make_tuple_indexwise<dimensions>([&](auto i) {
      const ::std::ptrdiff_t begin_i  = get_begin(i);
      const ::std::ptrdiff_t extent_i = get_extent(i);
      auto result                     = begin_i + (index % extent_i);
      index /= extent_i;
      return result;
    });
    CUDASTF_NO_DEVICE_STACK
  }

private:
  ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions> s;
};

// Deduction guides
template <typename... Int>
box(Int...) -> box<sizeof...(Int)>;
template <typename... E>
box(::std::initializer_list<E>...) -> box<sizeof...(E)>;
template <typename E, size_t dimensions>
box(::std::array<E, dimensions>) -> box<dimensions>;

#ifdef UNITTESTED_FILE
// Unit tests here
#endif // UNITTESTED_FILE

template <>
struct hash<pos4>
{
  ::std::size_t operator()(pos4 const& s) const noexcept
  {
    return hash_all(s.x, s.y, s.z, s.t);
  }
};

template <>
struct hash<dim4> : hash<pos4>
{};

} // end namespace cuda::experimental::stf