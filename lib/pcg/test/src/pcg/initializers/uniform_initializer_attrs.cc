#include "test/utils/doctest.h"
#include "test/utils/rapidcheck.h"
#include "pcg/initializers/uniform_initializer_attrs.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Arbitrary<UniformInitializerAttrs>") {
    rc::dc_check("arbitrary generates valid", [](UniformInitializerAttrs const &attrs) {
      RC_ASSERT(attrs.max_val >= attrs.min_val);
    });
  }
}
