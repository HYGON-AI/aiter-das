// SPDX-License-Identifier: MIT
 
#include "rocm_ops.hpp"
#include "attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    ATTENTION_PYBIND;
}