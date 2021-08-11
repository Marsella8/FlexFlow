/* Copyright 2020 Facebook
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow/ops/split.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;

__host__
OpMeta* Split::init_task(const Task* task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime* runtime)
{
  return NULL;
}

void calc_block_size(coord_t& num_blks,
                     coord_t& blk_size,
                     const Domain& domain,
                     int axis)
{
  num_blks = 1;
  blk_size = 1;
  for (int d = 0; d < domain.get_dim(); d++) {
    if (d <= axis)
      blk_size *= (domain.hi()[d] - domain.lo()[d] + 1);
    else
      num_blks *= (domain.hi()[d] - domain.lo()[d] + 1);
  }
}

void Split::forward_kernel(float **out_ptrs,
                           float const *in_ptr,
                           coord_t const *out_blk_sizes,
                           coord_t in_blk_size,
                           coord_t num_blks,
                           int numOutputs,
                           cudaStream_t stream)
{
  for (int i = 0; i < numOutputs; i++) {
    copy_with_stride<<<GET_BLOCKS(out_blk_sizes[i]*num_blks), CUDA_NUM_THREADS, 0, stream>>>(
        out_ptrs[i], in_ptr, num_blks, out_blk_sizes[i], in_blk_size);
    in_ptr += out_blk_sizes[i];
  }
}

void Split::forward_task(const Task *task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime *runtime)
{
  const Split* split = (Split*) task->args;
  assert(regions.size() == split->numOutputs + 1);
  assert(task->regions.size() == split->numOutputs + 1);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  float* out_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_domain, split->axis);
  for (int i = 0; i < split->numOutputs; i++) {
    Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+1].region.get_index_space());
    out_ptr[i] = helperGetTensorPointerWO<float>(
      regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    coord_t out_num_blks;
    calc_block_size(out_num_blks, out_blk_size[i], out_domain, split->axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_domain.get_dim(); j++)
      if (j != split->axis) {
        assert(out_domain.hi()[j] == in_domain.hi()[j]);
        assert(out_domain.lo()[j] == in_domain.lo()[j]);
      }
    total_volume += out_domain.get_volume();
  }
  assert(total_volume == in_domain.get_volume());

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel(out_ptr, in_ptr, out_blk_size, in_blk_size, num_blks, split->numOutputs, stream);
}

void Split::backward_task(const Task *task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx, Runtime *runtime)
{
  const Split* split = (Split*) task->args;
  assert(regions.size() == split->numOutputs + 1);
  assert(task->regions.size() == split->numOutputs + 1);
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  const float* out_grad_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_grad_domain, split->axis);
  for (int i = 0; i < split->numOutputs; i++) {
    Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+1].region.get_index_space());
    out_grad_ptr[i] = helperGetTensorPointerRO<float>(
      regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    coord_t out_num_blks;
    calc_block_size(out_num_blks, out_blk_size[i], out_grad_domain, split->axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_grad_domain.get_dim(); j++)
      if (j != split->axis) {
        assert(out_grad_domain.hi()[j] == in_grad_domain.hi()[j]);
        assert(out_grad_domain.lo()[j] == in_grad_domain.lo()[j]);
      }
    total_volume += out_grad_domain.get_volume();
  }
  assert(total_volume == in_grad_domain.get_volume());
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  for (int i = 0; i < split->numOutputs; i++) {
    add_with_stride<<<GET_BLOCKS(out_blk_size[i]*num_blks), CUDA_NUM_THREADS, 0, stream>>>(
        in_grad_ptr, out_grad_ptr[i], num_blks, in_blk_size, out_blk_size[i]);
    in_grad_ptr += out_blk_size[i];
  }
  //checkCUDA(cudaDeviceSynchronize());
}

bool Split::measure_operator_cost(Simulator* sim,
                                  const ParallelConfig& pc,
                                  CostMetrics& cost_metrics) const
{
  //TODO: implement measure_forward
  TensorBase sub_output[MAX_NUM_OUTPUTS], sub_input;
  for (int i = 0; i < numOutputs; i++)
    if (!outputs[i]->get_output_sub_tensor(pc, sub_output[i], OP_SPLIT))
      return false;
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, OP_SPLIT))
    return false;
  Domain in_domain = sub_input.get_domain();
  sim->free_all();
  float* output_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_domain, axis);
  for (int i = 0; i < numOutputs; i++) {
    Domain out_domain = sub_output[i].get_domain();
    output_ptr[i] = (float*)sim->allocate(sub_output[i].get_volume(), DT_FLOAT);
    coord_t out_num_blks;
    calc_block_size(out_num_blks, out_blk_size[i], out_domain, axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_domain.get_dim(); j++)
      if (j != axis) {
        assert(out_domain.hi()[j] == in_domain.hi()[j]);
        assert(out_domain.lo()[j] == in_domain.lo()[j]);
      }
    total_volume += out_domain.get_volume();
  }
  assert(total_volume == in_domain.get_volume());

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(output_ptr, input_ptr, out_blk_size, in_blk_size,
                   num_blks, numOutputs, stream);
  };
  // Assume backward has the same cost as forward
  backward = forward;

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);
  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Split] name(%s) num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
           name, sub_input.get_volume(),
           cost_metrics.forward_time,
           cost_metrics.backward_time);
  } else {
    printf("[Measure Split] name(%s) num_elements(%zu) forward_time(%.4lf)\n",
           name, sub_input.get_volume(),
           cost_metrics.forward_time);
  }
  return true;
}

}; // namespace FlexFlow