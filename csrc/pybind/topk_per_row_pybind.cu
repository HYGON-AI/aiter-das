// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "topk_per_row.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TOPK_PER_ROW_PYBIND;
}
