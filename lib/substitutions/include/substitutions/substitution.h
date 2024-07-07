#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H

#include "sub_parallel_computation_graph.dtg.h"
#include "substitutions/substitution.dtg.h"
#include "substitutions/unlabelled/unlabelled_dataflow_graph_pattern_match.dtg.h"

namespace FlexFlow {

bool is_valid_substitution(Substitution const &);

SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &,
                       Substitution const &,
                       UnlabelledDataflowGraphPatternMatch const &);

} // namespace FlexFlow

#endif
