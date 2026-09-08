// SPDX-License-Identifier: MIT
 
#include "rocm_ops.hpp"
#include "attention_asm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    ATTENTION_ASM_PYBIND;
}
