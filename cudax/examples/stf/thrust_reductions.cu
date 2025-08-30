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
 * @brief Multiple examples of using Thrust reductions with CUDASTF
 */

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <cuda/experimental/stf.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

// Custom functor for transform_reduce
struct square_functor
{
  __host__ __device__ double operator()(double x) const
  {
    return x * x;
  }
};

int main()
{
  context ctx;

  const size_t N = 1024 * 1024;

  // Initialize test data
  std::vector<double> X(N);
  for (size_t i = 0; i < N; i++)
  {
    X[i] = sin((double) i) * 0.1; // Small values to avoid overflow
  }

  // Results storage
  double sum_result = 0.0;
  double sum_squares_result = 0.0;
  double min_result = 0.0;
  double max_result = 0.0;

  // Create logical data
  auto lX = ctx.logical_data(&X[0], {N});
  auto lsum = ctx.logical_data(&sum_result, {1});
  auto lsum_squares = ctx.logical_data(&sum_squares_result, {1});
  auto lmin = ctx.logical_data(&min_result, {1});
  auto lmax = ctx.logical_data(&max_result, {1});

  auto where = exec_place::device(0);

  // Task 1: Simple sum reduction
  ctx.task(where, lX.read(), lsum.write())->*[](cudaStream_t stream, auto x, auto sum_slice) {
    thrust::device_ptr<const double> d_input = thrust::device_pointer_cast(x.data_handle());
    thrust::device_ptr<double> d_output = thrust::device_pointer_cast(sum_slice.data_handle());
    
    double result = thrust::reduce(
      thrust::cuda::par_nosync.on(stream),
      d_input, 
      d_input + x.size(), 
      0.0,
      thrust::plus<double>()
    );
    
    *d_output = result;
  };

  // Task 2: Sum of squares using transform_reduce
  ctx.task(where, lX.read(), lsum_squares.write())->*[](cudaStream_t stream, auto x, auto sum_squares_slice) {
    thrust::device_ptr<const double> d_input = thrust::device_pointer_cast(x.data_handle());
    thrust::device_ptr<double> d_output = thrust::device_pointer_cast(sum_squares_slice.data_handle());
    
    double result = thrust::transform_reduce(
      thrust::cuda::par_nosync.on(stream),
      d_input, 
      d_input + x.size(),
      square_functor(),                     // Transform operation
      0.0,                                  // Initial value
      thrust::plus<double>()                // Reduce operation
    );
    
    *d_output = result;
  };

  // Task 3: Find minimum value
  ctx.task(where, lX.read(), lmin.write())->*[](cudaStream_t stream, auto x, auto min_slice) {
    thrust::device_ptr<const double> d_input = thrust::device_pointer_cast(x.data_handle());
    thrust::device_ptr<double> d_output = thrust::device_pointer_cast(min_slice.data_handle());
    
    auto min_iter = thrust::min_element(
      thrust::cuda::par_nosync.on(stream),
      d_input, 
      d_input + x.size()
    );
    
    *d_output = *min_iter;
  };

  // Task 4: Find maximum value
  ctx.task(where, lX.read(), lmax.write())->*[](cudaStream_t stream, auto x, auto max_slice) {
    thrust::device_ptr<const double> d_input = thrust::device_pointer_cast(x.data_handle());
    thrust::device_ptr<double> d_output = thrust::device_pointer_cast(max_slice.data_handle());
    
    auto max_iter = thrust::max_element(
      thrust::cuda::par_nosync.on(stream),
      d_input, 
      d_input + x.size()
    );
    
    *d_output = *max_iter;
  };

  ctx.finalize();

  // Compute reference results on CPU for verification
  double ref_sum = 0.0;
  double ref_sum_squares = 0.0;
  double ref_min = X[0];
  double ref_max = X[0];

  for (size_t i = 0; i < N; i++)
  {
    ref_sum += X[i];
    ref_sum_squares += X[i] * X[i];
    ref_min = std::min(ref_min, X[i]);
    ref_max = std::max(ref_max, X[i]);
  }

  // Print results and verify
  std::cout << "=== Thrust + CUDASTF Reduction Results ===" << std::endl;
  std::cout << "Array size: " << N << std::endl;
  std::cout << std::endl;

  std::cout << "Sum:" << std::endl;
  std::cout << "  Reference: " << ref_sum << std::endl;
  std::cout << "  Thrust:    " << sum_result << std::endl;
  std::cout << "  Difference: " << fabs(sum_result - ref_sum) << std::endl;
  EXPECT(fabs(sum_result - ref_sum) < 1e-6);

  std::cout << std::endl;
  std::cout << "Sum of squares:" << std::endl;
  std::cout << "  Reference: " << ref_sum_squares << std::endl;
  std::cout << "  Thrust:    " << sum_squares_result << std::endl;
  std::cout << "  Difference: " << fabs(sum_squares_result - ref_sum_squares) << std::endl;
  EXPECT(fabs(sum_squares_result - ref_sum_squares) < 1e-6);

  std::cout << std::endl;
  std::cout << "Minimum:" << std::endl;
  std::cout << "  Reference: " << ref_min << std::endl;
  std::cout << "  Thrust:    " << min_result << std::endl;
  std::cout << "  Difference: " << fabs(min_result - ref_min) << std::endl;
  EXPECT(fabs(min_result - ref_min) < 1e-10);

  std::cout << std::endl;
  std::cout << "Maximum:" << std::endl;
  std::cout << "  Reference: " << ref_max << std::endl;
  std::cout << "  Thrust:    " << max_result << std::endl;
  std::cout << "  Difference: " << fabs(max_result - ref_max) << std::endl;
  EXPECT(fabs(max_result - ref_max) < 1e-10);

  std::cout << std::endl;
  std::cout << "All tests passed!" << std::endl;

  return 0;
}
