// SPDX-License-Identifier: MIT
 
#include <pybind11/pybind11.h>
#include "rocm_ops.hpp"
#include "aiter_enum.h"

PYBIND11_MODULE(module_aiter_enum, m)
{
    AITER_ENUM_PYBIND;
}
