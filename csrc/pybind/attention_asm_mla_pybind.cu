// SPDX-License-Identifier: MIT
 
#include "rocm_ops.hpp"
#include "attention_asm_mla.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    ATTENTION_ASM_MLA_PYBIND;
}
