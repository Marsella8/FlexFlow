// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/attention/multihead_attention_parallel_inputs.struct.toml
/* proj-data
{
  "generated_from": "7c434445707968123a361c038a337da2"
}
*/

#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.dtg.h"

#include "op-attrs/datatype.dtg.h"
#include "op-attrs/parallel_tensor_shape/discard_copy_degree.dtg.h"
#include "op-attrs/parallel_tensor_shape/sum_degree.dtg.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include <cstddef>
#include <sstream>

namespace FlexFlow {
MultiHeadAttentionParallelInputs::MultiHeadAttentionParallelInputs(
    ::FlexFlow::ShardParallelDim const &batch_dim,
    ::FlexFlow::ShardParallelDim const &sequence_dim,
    ::FlexFlow::ShardParallelDim const &query_dim,
    ::FlexFlow::ShardParallelDim const &key_dim,
    ::FlexFlow::ShardParallelDim const &value_dim,
    ::FlexFlow::DiscardCopyDegree const &discard_copy_degree,
    ::FlexFlow::DataType const &datatype)
    : batch_dim(batch_dim), sequence_dim(sequence_dim), query_dim(query_dim),
      key_dim(key_dim), value_dim(value_dim),
      discard_copy_degree(discard_copy_degree), datatype(datatype) {}
bool MultiHeadAttentionParallelInputs::operator==(
    MultiHeadAttentionParallelInputs const &other) const {
  return std::tie(this->batch_dim,
                  this->sequence_dim,
                  this->query_dim,
                  this->key_dim,
                  this->value_dim,
                  this->discard_copy_degree,
                  this->datatype) == std::tie(other.batch_dim,
                                              other.sequence_dim,
                                              other.query_dim,
                                              other.key_dim,
                                              other.value_dim,
                                              other.discard_copy_degree,
                                              other.datatype);
}
bool MultiHeadAttentionParallelInputs::operator!=(
    MultiHeadAttentionParallelInputs const &other) const {
  return std::tie(this->batch_dim,
                  this->sequence_dim,
                  this->query_dim,
                  this->key_dim,
                  this->value_dim,
                  this->discard_copy_degree,
                  this->datatype) != std::tie(other.batch_dim,
                                              other.sequence_dim,
                                              other.query_dim,
                                              other.key_dim,
                                              other.value_dim,
                                              other.discard_copy_degree,
                                              other.datatype);
}
bool MultiHeadAttentionParallelInputs::operator<(
    MultiHeadAttentionParallelInputs const &other) const {
  return std::tie(this->batch_dim,
                  this->sequence_dim,
                  this->query_dim,
                  this->key_dim,
                  this->value_dim,
                  this->discard_copy_degree,
                  this->datatype) < std::tie(other.batch_dim,
                                             other.sequence_dim,
                                             other.query_dim,
                                             other.key_dim,
                                             other.value_dim,
                                             other.discard_copy_degree,
                                             other.datatype);
}
bool MultiHeadAttentionParallelInputs::operator>(
    MultiHeadAttentionParallelInputs const &other) const {
  return std::tie(this->batch_dim,
                  this->sequence_dim,
                  this->query_dim,
                  this->key_dim,
                  this->value_dim,
                  this->discard_copy_degree,
                  this->datatype) > std::tie(other.batch_dim,
                                             other.sequence_dim,
                                             other.query_dim,
                                             other.key_dim,
                                             other.value_dim,
                                             other.discard_copy_degree,
                                             other.datatype);
}
bool MultiHeadAttentionParallelInputs::operator<=(
    MultiHeadAttentionParallelInputs const &other) const {
  return std::tie(this->batch_dim,
                  this->sequence_dim,
                  this->query_dim,
                  this->key_dim,
                  this->value_dim,
                  this->discard_copy_degree,
                  this->datatype) <= std::tie(other.batch_dim,
                                              other.sequence_dim,
                                              other.query_dim,
                                              other.key_dim,
                                              other.value_dim,
                                              other.discard_copy_degree,
                                              other.datatype);
}
bool MultiHeadAttentionParallelInputs::operator>=(
    MultiHeadAttentionParallelInputs const &other) const {
  return std::tie(this->batch_dim,
                  this->sequence_dim,
                  this->query_dim,
                  this->key_dim,
                  this->value_dim,
                  this->discard_copy_degree,
                  this->datatype) >= std::tie(other.batch_dim,
                                              other.sequence_dim,
                                              other.query_dim,
                                              other.key_dim,
                                              other.value_dim,
                                              other.discard_copy_degree,
                                              other.datatype);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::MultiHeadAttentionParallelInputs>::operator()(
    FlexFlow::MultiHeadAttentionParallelInputs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::ShardParallelDim>{}(x.batch_dim) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::ShardParallelDim>{}(x.sequence_dim) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::ShardParallelDim>{}(x.query_dim) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::ShardParallelDim>{}(x.key_dim) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::ShardParallelDim>{}(x.value_dim) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::DiscardCopyDegree>{}(x.discard_copy_degree) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::DataType>{}(x.datatype) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::MultiHeadAttentionParallelInputs
    adl_serializer<FlexFlow::MultiHeadAttentionParallelInputs>::from_json(
        json const &j) {
  return {
      j.at("batch_dim").template get<::FlexFlow::ShardParallelDim>(),
      j.at("sequence_dim").template get<::FlexFlow::ShardParallelDim>(),
      j.at("query_dim").template get<::FlexFlow::ShardParallelDim>(),
      j.at("key_dim").template get<::FlexFlow::ShardParallelDim>(),
      j.at("value_dim").template get<::FlexFlow::ShardParallelDim>(),
      j.at("discard_copy_degree").template get<::FlexFlow::DiscardCopyDegree>(),
      j.at("datatype").template get<::FlexFlow::DataType>()};
}
void adl_serializer<FlexFlow::MultiHeadAttentionParallelInputs>::to_json(
    json &j, FlexFlow::MultiHeadAttentionParallelInputs const &v) {
  j["__type"] = "MultiHeadAttentionParallelInputs";
  j["batch_dim"] = v.batch_dim;
  j["sequence_dim"] = v.sequence_dim;
  j["query_dim"] = v.query_dim;
  j["key_dim"] = v.key_dim;
  j["value_dim"] = v.value_dim;
  j["discard_copy_degree"] = v.discard_copy_degree;
  j["datatype"] = v.datatype;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::MultiHeadAttentionParallelInputs>
    Arbitrary<FlexFlow::MultiHeadAttentionParallelInputs>::arbitrary() {
  return gen::construct<FlexFlow::MultiHeadAttentionParallelInputs>(
      gen::arbitrary<::FlexFlow::ShardParallelDim>(),
      gen::arbitrary<::FlexFlow::ShardParallelDim>(),
      gen::arbitrary<::FlexFlow::ShardParallelDim>(),
      gen::arbitrary<::FlexFlow::ShardParallelDim>(),
      gen::arbitrary<::FlexFlow::ShardParallelDim>(),
      gen::arbitrary<::FlexFlow::DiscardCopyDegree>(),
      gen::arbitrary<::FlexFlow::DataType>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(MultiHeadAttentionParallelInputs const &x) {
  std::ostringstream oss;
  oss << "<MultiHeadAttentionParallelInputs";
  oss << " batch_dim=" << x.batch_dim;
  oss << " sequence_dim=" << x.sequence_dim;
  oss << " query_dim=" << x.query_dim;
  oss << " key_dim=" << x.key_dim;
  oss << " value_dim=" << x.value_dim;
  oss << " discard_copy_degree=" << x.discard_copy_degree;
  oss << " datatype=" << x.datatype;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s,
                         MultiHeadAttentionParallelInputs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
