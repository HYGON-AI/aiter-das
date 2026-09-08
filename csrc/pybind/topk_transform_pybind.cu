#include "topk_transform.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TOPK_TRANSFORM_PYBIND;
}
