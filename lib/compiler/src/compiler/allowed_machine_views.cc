#include "compiler/allowed_machine_views.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/multi_dimensional_stride.dtg.h"
#include "pcg/task_space_operator.h"
#include "utils/containers/all_of.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/extend.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_all_permutations.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/replicate.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip.h"
#include "utils/overload.h"

namespace FlexFlow {

bool is_valid_machine_view(MachineView const &mv,
                           TaskSpaceOperator const &task,
                           MachineSpecification const &ms) {
  MachineSpaceCoordinate maximum_device_coords = get_machine_space_coordinate(
      task, mv, get_maximum_fragment_coordinate(task), ms);
  return is_valid_machine_space_coordinates(ms, maximum_device_coords);
}

/* Generates a set of candidate `MachineView`s
 * The returned set includes all valid machine views, and might contain
 invalid ones. This function should never be used externally (see
 * `get_allowed_partial_machine_view_mappings` instead). There is no
 guarantee that a non-empty returned set contains a valid machine view (i.e.
 its possible for all
 * `MachineView`s to be invalid)
 */
static std::unordered_set<MachineView>
    get_candidate_machine_views(MachineSpecification const &machine_spec,
                                TaskSpaceOperator const &task,
                                DeviceType const &device_type) {

  auto candidate_strides =
      [](std::vector<num_points_t> const &tensor_dims,
         int total_devices) -> std::unordered_multiset<MultiDimensionalStride> {
    int min_num_devices_with_full_stride_volume =
        product(transform(tensor_dims, [](num_points_t const &num_devices) {
          return num_devices.unwrapped - 1;
        }));
    int max_stride_upper_bound =
        std::ceil(total_devices / min_num_devices_with_full_stride_volume);

    std::vector<stride_t> single_stride_range =
        transform(range(1, max_stride_upper_bound + 1),
                  [](int stride) { return stride_t(stride); });
    std::unordered_multiset<std::vector<stride_t>> raw_stride_vectors =
        cartesian_product(replicate(tensor_dims.size(), single_stride_range));
    std::unordered_multiset<MultiDimensionalStride> strides =
        transform(raw_stride_vectors, [](auto const &stride_vec) {
          return MultiDimensionalStride{stride_vec};
        });
    return strides;
  };

  auto candidate_starts = [](MachineSpecification const &ms,
                             DeviceType const &device_type) {
    std::unordered_set<MachineSpaceCoordinate> result;
    for (int i : range(ms.num_nodes)) {
      for (int j : range(get_num_devices_per_node(ms, device_type))) {
        result.insert(MachineSpaceCoordinate{i, j, device_type});
      }
    }
    return result;
  };

  auto candidate_projections = [](TaskSpaceOperator const &task) {
    std::unordered_set<MachineSpecificationDimension> options = {
        MachineSpecificationDimension::INTER_NODE,
        MachineSpecificationDimension::INTRA_NODE};
    return get_all_permutations_with_repetition(options, num_dims(task));
  };

  std::vector<num_points_t> tensor_dims = task.degrees;
  int total_devices = get_num_devices(machine_spec, device_type);

  std::unordered_set<MachineView> machine_views;

  for (MultiDimensionalStride const &strides :
       candidate_strides(tensor_dims, total_devices)) {
    for (MachineSpaceCoordinate start :
         candidate_starts(machine_spec, device_type)) {
      for (std::vector<MachineSpecificationDimension> const &proj :
           candidate_projections(task)) {
        machine_views.insert(MachineView{strides.raw_strides, proj, start});
      }
    }
  }
  return machine_views;
}

std::unordered_set<MachineView>
    get_allowed_machine_views(MachineSpecification const &machine_spec,
                              TaskSpaceOperator const &task,
                              DeviceType device_type) {

  std::unordered_set<MachineView> views =
      get_candidate_machine_views(machine_spec, task, device_type);
  return filter(views, [&](MachineView const &mv) {
    return is_valid_machine_view(mv, task, machine_spec);
  });
}

} // namespace FlexFlow
