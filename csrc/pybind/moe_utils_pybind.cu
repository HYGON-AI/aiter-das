/* SPDX-License-Identifier: MIT
   */
#include "rocm_ops.hpp"
#include "moe_utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      MOE_UTILS_PYBIND;
}
