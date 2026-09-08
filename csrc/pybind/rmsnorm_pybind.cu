// SPDX-License-Identifier: MIT
 
#include "rocm_ops.hpp"
#include "rmsnorm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    RMSNORM_PYBIND;
}