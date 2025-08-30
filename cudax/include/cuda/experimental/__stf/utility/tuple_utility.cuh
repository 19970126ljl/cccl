//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include <tuple>
#include <utility>

namespace cuda::experimental::stf {
namespace reserved {

template <typename Tuple, std::size_t... I>
_CCCL_HOST_DEVICE auto tuple_pop_front_impl(const Tuple& t, std::index_sequence<I...>) {
    return std::make_tuple(std::get<I + 1>(t)...);
}

} // namespace reserved

template <typename Tuple>
_CCCL_HOST_DEVICE auto tuple_pop_front(const Tuple& t) {
    return reserved::tuple_pop_front_impl(t, std::make_index_sequence<std::tuple_size_v<Tuple> - 1>{});
}

} // namespace cuda::experimental::stf
