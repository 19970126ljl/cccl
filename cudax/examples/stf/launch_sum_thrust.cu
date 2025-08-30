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
 * @brief A reduction kernel written using CUDASTF and Thrust
 */

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

double X0(int i)
{
  return sin((double) i);
}

int main()
{
  context ctx;

  const size_t N = 128 * 1024 * 1024;

  std::vector<double> X(N);
  double sum = 0.0;

  double ref_sum = 0.0;

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = sin((double) ind);
    ref_sum += X[ind];
  }

  auto lX   = ctx.logical_data(&X[0], {N});
  auto lsum = ctx.logical_data(&sum, {1});

  // Use a single device for this example
  auto where = exec_place::device(0);

  // Use CUDASTF task to perform Thrust reduction
  ctx.task(where, lX.read(), lsum.write())->*[](cudaStream_t stream, auto x, auto sum_slice) {
    // Create device pointers from the data handles
    thrust::device_ptr<const double> d_input = thrust::device_pointer_cast(x.data_handle());
    thrust::device_ptr<double> d_output = thrust::device_pointer_cast(sum_slice.data_handle());
    
    // Perform reduction using Thrust with the provided stream
    double result = thrust::reduce(
      thrust::cuda::par_nosync.on(stream),  // Use the stream provided by CUDASTF
      d_input, 
      d_input + x.size(), 
      0.0,                                  // Initial value
      thrust::plus<double>()                // Binary operation
    );
    
    // Store the result
    *d_output = result;
  };

  ctx.finalize();

  EXPECT(fabs(sum - ref_sum) < 0.0001);
  
  printf("Reference sum: %f\n", ref_sum);
  printf("Thrust sum: %f\n", sum);
  printf("Difference: %f\n", fabs(sum - ref_sum));
  
  return 0;
}
