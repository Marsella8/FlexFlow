#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_labels.h"

namespace FlexFlow {

// TODO(@lockshaw) eventually move this over to tests

struct Visitor {
  std::string operator()(Node const &, int);
  float operator()(OpenDataflowValue const &, int);
};

template LabelledOpenDataflowGraphView<std::string, float> rewrite_labels<int, int, Visitor>(LabelledOpenDataflowGraphView<int, int> const &, Visitor);

} // namespace FlexFlow
