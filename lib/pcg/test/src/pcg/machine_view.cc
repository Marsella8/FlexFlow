#include "pcg/machine_view.h"
#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"
#include "test/utils/doctest.h"
#include "utils/containers/transform.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("MachineView - utility functions") {
    MachineView mv = MachineView{
        MachineViewCoordinates{{0, 0, 0}},
        StridedRectangle{{StridedRectangleSide(num_points_t{7}, stride_t{5}),
                          StridedRectangleSide(num_points_t{10}, stride_t{2}),
                          StridedRectangleSide(num_points_t{1}, stride_t{4})}},
        DeviceType::GPU};

    SUBCASE("num_dims") {
      CHECK(num_dims(mv) == 3);
    }
    SUBCASE("num_devices") {
      CHECK(num_devices(mv) == 7 * 10 * 1);
    }

    SUBCASE("get_side_size_per_dim") {
      std::vector<side_size_t> expected = {
          side_size_t(1 * 4), side_size_t(7 * 5), side_size_t(10 * 2)};
      std::vector<side_size_t> result = get_side_size_per_dim(mv);
      CHECK(expected == result);
    }
    SUBCASE("get_num_devices_per_dim") {
      std::vector<num_points_t> expected = {
          num_points_t(1), num_points_t(7), num_points_t(10)};
      std::vector<num_points_t> result = get_num_devices_per_dim(mv);
      CHECK(expected == result);
    }
  }

  TEST_CASE("get_devices_coordinates") {

    SUBCASE("2D MachineView") {

      MachineView mv =
          MachineView{MachineViewCoordinates{{0, 0}},
                      StridedRectangle{{
                          StridedRectangleSide(num_points_t(2), stride_t{3}),
                          StridedRectangleSide(num_points_t(2), stride_t{2}),
                      }},
                      DeviceType::GPU};
      SUBCASE("get_devices_coordinates") {
        std::unordered_set<MachineViewCoordinates> expected = {
            {MachineViewCoordinates{{0, 0}},
             MachineViewCoordinates{{0, 1}},
             MachineViewCoordinates{{1, 0}},
             MachineViewCoordinates{{1, 1}}}};
        std::unordered_set<MachineViewCoordinates> result =
            get_devices_coordinates(mv);
        CHECK(expected == result);
      }
    }
    SUBCASE("3D MachineView") {
      // 3D MachineView describes a 3 x 2 x 4 area, and 1*2*2=4 devices.
      // (Pre offset) the devices are at coordinates (0, 0, 0), (0, 0, 2), (0,
      // 1, 0), (0, 1, 2) Thus (pre offset) we have as device ids:
      //  0  = 0*1 + 0*3 + 0*(2*3)
      //  12  = 0*1 + 0*3 + 2*(2*3)
      //  3  = 0*1 + 1*3 + 0*(2*3)
      //  15  = 0*1 + 1*3 + 1*(2*3)
      // Where the coefficients are obtained by doing `scanl(area_coefficients,
      // 1, product) = {1,3,6}` and ignoring the last term. We do, however, have
      // 1 as a starting device, meaning all device-id are offset by 1. We thus
      // have 1, 13, 4, 16 as device-ids

      MachineView mv =
          MachineView{MachineViewCoordinates{{0, 1, 2}},
                      StridedRectangle{{
                          StridedRectangleSide(num_points_t(1), stride_t{3}),
                          StridedRectangleSide(num_points_t(2), stride_t{1}),
                          StridedRectangleSide(num_points_t(2), stride_t{2}),
                      }},
                      DeviceType::GPU};

      SUBCASE("get_devices_coordinates") {
        std::unordered_set<MachineViewCoordinates> expected = {
            {MachineViewCoordinates{{0, 0, 0}},
             MachineViewCoordinates{{0, 0, 1}},
             MachineViewCoordinates{{0, 1, 0}},
             MachineViewCoordinates{{0, 1, 1}}}};
        std::unordered_set<MachineViewCoordinates> result =
            get_devices_coordinates(mv);
        CHECK(expected == result);
      }
    }
  }

  TEST_CASE("get_maximum_device_coordinates") {
    SUBCASE("2D MachineView") {

      MachineView mv =
          MachineView{MachineViewCoordinates{{0, 0}},
                      StridedRectangle{{
                          StridedRectangleSide(num_points_t(2), stride_t{3}),
                          StridedRectangleSide(num_points_t(2), stride_t{2}),
                      }},
                      DeviceType::GPU};

      SUBCASE("get_maximum_device_coordinates") {
        CHECK(get_maximum_device_coordinates(mv) ==
              MachineViewCoordinates{{1, 1}});
      }
    }

    SUBCASE("3D MachineView") {

      MachineView mv =
          MachineView{MachineViewCoordinates{{0, 1, 2}},
                      StridedRectangle{{
                          StridedRectangleSide(num_points_t(1), stride_t{3}),
                          StridedRectangleSide(num_points_t(2), stride_t{1}),
                          StridedRectangleSide(num_points_t(2), stride_t{2}),
                      }},
                      DeviceType::GPU};

      SUBCASE("get_maximum_device_coordinates") {
        CHECK(get_maximum_device_coordinates(mv) ==
              MachineViewCoordinates{{0, 1, 1}});
      }
    }
  }

  TEST_CASE("make_1d_machine_view") {

    MachineViewCoordinates start = MachineViewCoordinates{{1}};
    MachineView mv = MachineView{
        start,
        StridedRectangle{{StridedRectangleSide{num_points_t{7}, stride_t{5}}}},
        DeviceType::GPU};

    SUBCASE("make_1d_machine_view(int start, int stop, stride_t "
            "stride,DeviceType device_type)") {
      MachineView result =
          make_1d_machine_view(1, 1 + 7 * 5, stride_t{5}, DeviceType::GPU);
      MachineView correct = mv;
      CHECK(result == correct);
    }

    SUBCASE("make_1d_machine_view(gpu_id_t start, num_points_t num_points, "
            "stride_t stride,DeviceType device_type)") {
      MachineView result = make_1d_machine_view(
          1, num_points_t{7}, stride_t{5}, DeviceType::GPU);
      MachineView correct = mv;
      CHECK(result == correct);
    }

    SUBCASE("make_1d_machine_view(gpu_id_t start, side_size_t side_size, "
            "stride_t stride,DeviceType device_type)") {
      MachineView result = make_1d_machine_view(
          1, side_size_t{7 * 5}, stride_t{5}, DeviceType::GPU);
      MachineView correct = mv;
      CHECK(result == correct);
    }
  }

  TEST_CASE("get_device_id") {
    SUBCASE("1D case") {
      MachineView mv =
          make_1d_machine_view(1, num_points_t{3}, stride_t{2}); // 1 3 5
      MachineSpecification ms = MachineSpecification{
          1, 0, 6, 0, 0}; // Single node with 6 GPUs (0,1,2,3,4,5)
      MachineViewProjection projection = MachineViewProjection{
          {{machine_view_dim_idx_t{0}, MachineSpecificationDimension::INTRA}}};

      SUBCASE("Device 0") {
        MachineViewCoordinates device = MachineViewCoordinates{{0}};
        device_id_t correct = device_id_from_index(1, DeviceType::GPU);
        device_id_t result = get_device_id(mv, device, ms, projection);
        CHECK(correct == result);
      }

      SUBCASE("Device 1") {
        MachineViewCoordinates device = MachineViewCoordinates{{1}};
        device_id_t correct = device_id_from_index(3, DeviceType::GPU);
        device_id_t result = get_device_id(mv, device, ms, projection);
        CHECK(correct == result);
      }

      SUBCASE("Device 2") {
        MachineViewCoordinates device = MachineViewCoordinates{{2}};
        device_id_t correct = device_id_from_index(5, DeviceType::GPU);
        device_id_t result = get_device_id(mv, device, ms, projection);
        CHECK(correct == result);
      }
    }
    SUBCASE("2D case") {
      MachineView mv =
          MachineView{MachineViewCoordinates{{1, 2}},
                      StridedRectangle{
                          {StridedRectangleSide(num_points_t(2), stride_t{1}),
                           StridedRectangleSide(num_points_t(2), stride_t{2})}},
                      DeviceType::GPU};
      MachineSpecification ms =
          MachineSpecification{3, 0, 5, 0, 0}; // 3 nodes with 5 GPUs each
      MachineViewProjection projection = MachineViewProjection{
          {{machine_view_dim_idx_t{0}, MachineSpecificationDimension::INTER},
           {machine_view_dim_idx_t{1}, MachineSpecificationDimension::INTRA}}};

      SUBCASE("Device (0,0)") {
        MachineViewCoordinates device = MachineViewCoordinates{{0, 0}};
        device_id_t correct = device_id_from_index(7, DeviceType::GPU);
        device_id_t result = get_device_id(mv, device, ms, projection);
        CHECK(correct == result);
      }

      SUBCASE("Device (0,1)") {
        MachineViewCoordinates device = MachineViewCoordinates{{0, 1}};
        device_id_t correct = device_id_from_index(9, DeviceType::GPU);
        device_id_t result = get_device_id(mv, device, ms, projection);
        CHECK(correct == result);
      }
      SUBCASE("Device (1,0)") {
        MachineViewCoordinates device = MachineViewCoordinates{{1, 0}};
        device_id_t correct = device_id_from_index(12, DeviceType::GPU);
        device_id_t result = get_device_id(mv, device, ms, projection);
        CHECK(correct == result);
      }
      SUBCASE("Device (1,1)") {
        MachineViewCoordinates device = MachineViewCoordinates{{1, 1}};
        device_id_t correct = device_id_from_index(14, DeviceType::GPU);
        device_id_t result = get_device_id(mv, device, ms, projection);
        CHECK(correct == result);
      }
    }

    SUBCASE("3D case") {
      MachineView mv =
          MachineView{MachineViewCoordinates{{0, 2, 0}},
                      StridedRectangle{
                          {StridedRectangleSide(num_points_t(2), stride_t{1}),
                           StridedRectangleSide(num_points_t(2), stride_t{2}),
                           StridedRectangleSide(num_points_t(2), stride_t{1})}},
                      DeviceType::GPU};
      MachineSpecification ms =
          MachineSpecification{2, 0, 8, 0, 0}; // 3 nodes with 5 GPUs each
      MachineViewProjection projection = MachineViewProjection{
          {{machine_view_dim_idx_t{0}, MachineSpecificationDimension::INTER},
           {machine_view_dim_idx_t{1}, MachineSpecificationDimension::INTRA},
           {machine_view_dim_idx_t{2}, MachineSpecificationDimension::INTRA}}};

      SUBCASE("Device (0,0,1)") {
        MachineViewCoordinates device = MachineViewCoordinates{{0, 1, 0}};
        device_id_t correct = device_id_from_index(3, DeviceType::GPU);
        device_id_t result = get_device_id(mv, device, ms, projection);
        CHECK(correct == result);
      }

      SUBCASE("Device (1, 1, 0)") {
        MachineViewCoordinates device = MachineViewCoordinates{{1, 0, 1}};
        device_id_t correct = device_id_from_index(14, DeviceType::GPU);
        device_id_t result = get_device_id(mv, device, ms, projection);
        CHECK(correct == result);
      }
      SUBCASE("All devices") {
        std::unordered_set<device_id_t> result =
            get_device_ids(mv, ms, projection);
        std::unordered_set<int> devices = {2, 3, 10, 11, 6, 7, 14, 15};
        std::unordered_set<device_id_t> correct =
            transform(devices, [&](int idx) {
              return device_id_from_index(idx, DeviceType::GPU);
            });

        CHECK(result == correct);
      }
    }
  }
}
