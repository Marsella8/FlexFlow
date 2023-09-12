#ifndef _FLEXFLOW_LIB_UTILS_PREPROCESSOR_EXTRA_INCLUDE_UTILS_PREPROCESSOR_EXTRA_STRINGIZE_H
#define _FLEXFLOW_LIB_UTILS_PREPROCESSOR_EXTRA_INCLUDE_UTILS_PREPROCESSOR_EXTRA_STRINGIZE_H

#include "wrap_arg.h"
#include <boost/preprocessor/stringize.hpp>

#define STRINGIZE(...) \
  BOOST_PP_STRINGIZE(UNWRAP_ARG(WRAP_ARG(__VA_ARGS__)))

#endif
