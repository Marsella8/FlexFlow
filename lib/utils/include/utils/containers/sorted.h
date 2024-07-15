#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SORTED_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SORTED_H

#include "utils/type_traits_core.h"
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <map>

namespace FlexFlow {

template <typename C>
struct sort_value_type :
  type_identity<typename C::value_type> {};

template <typename K, typename V>
struct sort_value_type<std::unordered_map<K, V>> :
  type_identity<std::pair<K, V>> {};

template <typename K, typename V>
struct sort_value_type<std::map<K, V>> :
  type_identity<std::pair<K, V>> {};

template <typename T>
using sort_value_type_t = typename sort_value_type<T>::type;

template <typename C, typename F, typename Elem = sort_value_type_t<C>>
void inplace_sorted_by(C &c, F const &f) {
  CHECK_SUPPORTS_ITERATOR_TAG(std::random_access_iterator_tag, C);

  auto custom_comparator = [&](Elem const &lhs, Elem const &rhs) -> bool {
    return f(lhs, rhs);
  };
  std::sort(c.begin(), c.end(), custom_comparator);
}

template <typename C, typename Elem = sort_value_type_t<C>>
std::vector<Elem> sorted(C const &c) {
  std::vector<Elem> result(c.begin(), c.end());
  inplace_sorted_by(result, [](Elem const &l, Elem const &r) { return l < r; });
  return result;
}

template <typename C, typename F, typename Elem = sort_value_type_t<C>>
std::vector<Elem> sorted_by(C const &c, F const &f) {
  std::vector<Elem> result(c.begin(), c.end());
  inplace_sorted_by(result, f);
  return result;
}

} // namespace FlexFlow

#endif
