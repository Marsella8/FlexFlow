MachineSpaceCoordinate
    get_machine_space_coordinates(MachineView const &mv,
                                  MachineViewCoordinate const &coordinates,
                                  MachineSpecification const &ms,
                                  MachineViewProjection const &projection) {

  auto inter_projection =
      filter_values(projection.machine_view_dim_to_machine_spec_dim,
                    [](MachineSpecificationDimension const &dim) {
                      return dim == MachineSpecificationDimension::INTER_NODE;
                    });
  auto intra_projection =
      filter_values(projection.machine_view_dim_to_machine_spec_dim,
                    [](MachineSpecificationDimension const &dim) {
                      return dim == MachineSpecificationDimension::INTRA_NODE;
                    });

  MachineViewCoordinate transformed_coordinates = MachineViewCoordinate{
      transform(zip(coordinates.raw_coord, mv.rect.get_sides()),
                [&](auto const &pair) {
                  return pair.first * pair.second.stride.unwrapped;
                })};
  transformed_coordinates = MachineViewCoordinate{
      transform(zip(transformed_coordinates.raw_coord, mv.start.raw_coord),
                [&](auto const &pair) { return pair.first + pair.second; })};

  auto get_coordinate = [&](auto const &sub_projection) {
    std::vector<machine_view_dim_idx_t> relevant_dimensions =
        sorted(keys(sub_projection));
    std::vector<side_size_t> relevant_side_sizes =
        transform(relevant_dimensions, [&](auto const &idx) {
          return get_side_size(get_side_at_idx(mv, idx));
        });
    std::vector<int> coefficients =
        scanl(relevant_side_sizes,
              1,
              [](size_t const &result, side_size_t const &side_size) {
                return result * side_size.unwrapped;
              });
    std::vector<int> filtered_coord;
    for (int i = 0; i < transformed_coordinates.raw_coord.size(); ++i) {
      if (contains(relevant_dimensions, machine_view_dim_idx_t{i})) {
        filtered_coord.push_back(transformed_coordinates.raw_coord[i]);
      }
    }
    return sum(
        transform(zip(coefficients, filtered_coord),
                  [](auto const pair) { return pair.first * pair.second; }));
  };
  int inter_coordinate = get_coordinate(inter_projection);
  int intra_coordinate = get_coordinate(intra_projection);
  return MachineSpaceCoordinate{
      inter_coordinate, intra_coordinate, mv.device_type};
}

device_id_t get_device_id(MachineView const &mv,
                          MachineViewCoordinate const &coordinates,
                          MachineSpecification const &ms,
                          MachineViewProjection const &projection) {
  MachineSpaceCoordinate coord =
      get_machine_space_coordinates(mv, coordinates, ms, projection);
  return get_device_id(ms, coord);
}

std::unordered_set<device_id_t>
    get_device_ids(MachineView const &mv,
                   MachineSpecification const &ms,
                   MachineViewProjection const &projection) {

  return transform(get_devices_coordinates(mv),
                   [&](MachineViewCoordinate const &c) {
                     return get_device_id(mv, c, ms, projection);
                   });
}

size_t num_dims(MachineView const &mv) {
  return get_num_dims(mv.rect);
}

std::vector<num_points_t> get_num_devices_per_dim(MachineView const &mv) {
  return transform(mv.rect.get_sides(), [](StridedRectangleSide const &side) {
    return side.num_points;
  });
}

std::vector<side_size_t> get_side_size_per_dim(MachineView const &mv) {
  return transform(mv.rect.get_sides(), get_side_size);
}

size_t num_devices(MachineView const &mv) {
  return get_num_points(mv.rect).unwrapped;
}

StridedRectangleSide get_side_at_idx(MachineView const &mv,
                                     machine_view_dim_idx_t const &idx) {
  return mv.rect.at(idx.unwrapped);
}

static StridedRectangle make_1d_rect(int start, int stop, stride_t stride) {
  assert(stop > start);
  assert(stride > stride_t(0));
  StridedRectangleSide side =
      strided_side_from_size_and_stride(side_size_t{stop - start}, stride);
  StridedRectangle rect =
      StridedRectangle{std::vector<StridedRectangleSide>{side}};
  return rect;
}

MachineView make_1d_machine_view(DeviceType device_type,
                                 int start,
                                 int stop,
                                 stride_t stride) {
  StridedRectangle rect = make_1d_rect(start, stop, stride);
  MachineViewCoordinate start_coordinate = MachineViewCoordinate{{start}};
  return MachineView{start_coordinate, rect, device_type};
}

static StridedRectangle
    make_1d_rect(int start, num_points_t num_points, stride_t stride) {
  return make_1d_rect(
      start, start + num_points.unwrapped * stride.unwrapped, stride);
}

MachineView make_1d_machine_view(DeviceType device_type,
                                 int start,
                                 num_points_t num_points,
                                 stride_t stride) {
  StridedRectangle rect = make_1d_rect(start, num_points, stride);
  MachineViewCoordinate start_coordinate = MachineViewCoordinate{{start}};
  return MachineView{start_coordinate, rect, device_type};
}

static StridedRectangle
    make_1d_rect(int start, side_size_t interval_size, stride_t stride) {
  return make_1d_rect(start, start + interval_size.unwrapped, stride);
}

MachineView make_1d_machine_view(DeviceType device_type,
                                 int start,
                                 side_size_t interval_size,
                                 stride_t stride) {
  StridedRectangle rect = make_1d_rect(start, interval_size, stride);
  MachineViewCoordinate start_coordinate = MachineViewCoordinate{{start}};
  return MachineView{start_coordinate, rect, device_type};
}
