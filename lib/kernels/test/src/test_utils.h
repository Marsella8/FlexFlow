#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/device.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/datatype.h"
#include "op-attrs/datatype_value.dtg.h"
#include <doctest/doctest.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace FlexFlow {

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator);

GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator);

GenericTensorAccessorW create_zero_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator);

TensorShape
    make_tensor_shape_from_legion_dims(LegionOrdered<size_t> const &dims,
                                       DataType DT);

bool contains_non_zero(GenericTensorAccessorR const &accessor);

void fill_with_zeros(GenericTensorAccessorW const &accessor);

GenericTensorAccessorW
    copy_accessor_w_to_cpu_if_necessary(GenericTensorAccessorW const &accessor,
                                        Allocator &allocator);

GenericTensorAccessorR
    copy_accessor_r_to_cpu_if_necessary(GenericTensorAccessorR const &accessor,
                                        Allocator &allocator);

void print_2d_tensor_accessor_contents(GenericTensorAccessorR const &accessor);

bool accessors_are_equal(GenericTensorAccessorR const &accessor_a,
                         GenericTensorAccessorR const &accessor_b);

GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                DataTypeValue val);

GenericTensorAccessorR create_filled_accessor_r(TensorShape const &shape,
                                                Allocator &allocator,
                                                DataTypeValue val);

GenericTensorAccessorW create_iota_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator,
                                                     bool cpu_fill = false);

void fill_tensor_accessor_w(GenericTensorAccessorW accessor,
                            float val,
                            bool cpu_fill = false);

TensorShape make_float_tensor_shape_from_legion_dims(FFOrdered<size_t> dims);

TensorShape make_double_tensor_shape_from_legion_dims(FFOrdered<size_t> dims);

template <typename T>
std::vector<T> load_data_to_host_from_device(GenericTensorAccessorR accessor) {
  int volume = accessor.shape.get_volume();

  std::vector<T> local_data(volume);
  checkCUDA(cudaMemcpy(local_data.data(),
                       accessor.ptr,
                       local_data.size() * sizeof(T),
                       cudaMemcpyDeviceToHost));
  return local_data;
}

template <typename T>
bool contains_non_zero(std::vector<T> &data) {
  return !all_of(
      data.begin(), data.end(), [](T const &val) { return val == 0; });
}

template <typename T, typename Func>
std::vector<T> repeat(std::size_t n, Func &&func) {
  std::vector<T> result;
  // result.reserve(n); // Sometimes we don't have default constructor for T
  for (std::size_t i = 0; i < n; ++i) {
    result.push_back(func());
  }
  return result;
}

} //FlwxFlow


// Specialize doctest's StringMaker for std::vector<float>
template <>
struct doctest::StringMaker<std::vector<float>> {
  static doctest::String convert(std::vector<float> const &vec) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i != vec.size() - 1) {
        oss << ", ";
      }
    }
    return doctest::String(("[" + oss.str() + "]").c_str());
  }
};

#endif
