// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/element_unary_attrs.struct.toml

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_UNARY_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_UNARY_ATTRS_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/operator_type.h"
#include <functional>
#include <ostream>
#include <sstream>
#include <tuple>

namespace FlexFlow {
struct ElementUnaryAttrs {
  ElementUnaryAttrs() = delete;
  ElementUnaryAttrs(::FlexFlow::OperatorType const &op_type);

  bool operator==(ElementUnaryAttrs const &) const;
  bool operator!=(ElementUnaryAttrs const &) const;
  bool operator<(ElementUnaryAttrs const &) const;
  bool operator>(ElementUnaryAttrs const &) const;
  bool operator<=(ElementUnaryAttrs const &) const;
  bool operator>=(ElementUnaryAttrs const &) const;
  ::FlexFlow::OperatorType op_type;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ElementUnaryAttrs> {
  size_t operator()(FlexFlow::ElementUnaryAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::ElementUnaryAttrs> {
  static FlexFlow::ElementUnaryAttrs from_json(json const &);
  static void to_json(json &, FlexFlow::ElementUnaryAttrs const &);
};
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(ElementUnaryAttrs const &);
std::ostream &operator<<(std::ostream &, ElementUnaryAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_UNARY_ATTRS_H
