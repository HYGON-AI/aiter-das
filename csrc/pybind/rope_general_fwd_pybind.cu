// SPDX-License-Identifier: MIT
 
#include "rocm_ops.hpp"
#include "rope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    ROPE_GENERAL_FWD_PYBIND;
}
