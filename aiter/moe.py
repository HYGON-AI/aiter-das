# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import logging
import os
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    value = value.strip().lower()
    return value not in ("", "0", "false", "off", "no")


_AITER_LOG_MOE_PARAM = _env_flag_enabled("AITER_LOG_MORE") or _env_flag_enabled("AITER_LOG_OP_PARAM")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class MoeSolutionType:
    MOE_C = "moe_c"
    ASM = "asm"
    TRITON = "triton"
    CK = "ck"


class MoeQuantType:
    """Quantization types supported by get_aiter_moe_config / aiter_moe."""
    W16A16 = "w16a16"
    W4A16 = "w4a16"
    WFP4A16 = "fp4_w4a16"
    WFP4A8 = "fp4_w4a8"
    W4A8 = "w4a8"
    W8A8 = "int8_w8a8"
    FP8_W8A8 = "fp8_w8a8"
    INT8_W8A16 = "int8_w8a16"
    FP8_W8A16 = "fp8_w8a16"


@dataclass
class AiterMoeConfig:
    """Config returned by :func:`get_aiter_moe_config`.

    Attributes:
        quant_type: The quantization type this config was obtained for.
        solution_type: Which backend to use (MoeSolutionType constant), or
            None if no solution was found.
        config: Backend-specific config dict (opaque to the caller).
        need_shuffle: Whether the backend requires weight shuffling via
            :func:`aiter_moe_shfl_weight` before calling :func:`aiter_moe`.
        dtype: Compute dtype the config was selected for; moe_c channelwise
            lookup keys CSV rows by it, so it must equal the runtime
            ``hidden_states.dtype`` for float activations. Never reuse a
            cached config across dtypes.
    """
    quant_type: Optional[str] = None
    solution_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    need_shuffle: bool = False
    need_shuffle_scale: bool = False
    dtype: Optional[torch.dtype] = None


def _pick_closest_config(configs: Dict[int, Any], m: int) -> Dict[str, Any]:
    return configs[min(configs.keys(), key=lambda x: abs(x - m))]


def _resolve_moe_c_config_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return dtype
    logger.warning(
        "dtype=%s is not a float compute dtype; normalizing to "
        "torch.bfloat16 for the moe_c selection key",
        dtype,
    )
    return torch.bfloat16


def _pad_tensor_dim(x: torch.Tensor, dim: int, target: Optional[int]) -> torch.Tensor:
    if target is None:
        return x
    dim = dim if dim >= 0 else x.dim() + dim
    if x.shape[dim] >= target:
        return x
    padded_shape = list(x.shape)
    padded_shape[dim] = target
    padded = x.new_zeros(padded_shape)
    slices = [slice(None)] * x.dim()
    slices[dim] = slice(0, x.shape[dim])
    padded[tuple(slices)] = x
    return padded


def _pad_fp8_w8a8_moec_k(k: int) -> int:
    return k if k % 64 == 0 else ((k + 63) // 64) * 64


def _pad_w16a16_moec_k(k: int) -> int:
    return k if k % 64 == 0 else ((k + 63) // 64) * 64


def _w16a16_moec_weight_layout(
    w1: Optional[torch.Tensor],
    w2: Optional[torch.Tensor],
) -> str:
    if w1 is None or w2 is None:
        return "canonical"
    if w1.dim() != 3 or w2.dim() != 3 or w1.shape[0] != w2.shape[0]:
        raise ValueError(
            "W16A16 MOE_C shuffle expects 3D w1/w2 tensors with the same "
            "expert dimension"
        )
    canonical = w1.shape[1] == 2 * w2.shape[2] and w1.shape[2] == w2.shape[1]
    transposed = w1.shape[2] == 2 * w2.shape[1] and w1.shape[1] == w2.shape[2]
    if canonical:
        return "canonical"
    if transposed:
        return "transposed"
    raise ValueError(
        "W16A16 MOE_C shuffle expects weights in either "
        "(E, 2*N, K)/(E, K, N) or (E, K, 2*N)/(E, N, K) layout, "
        f"got w1={tuple(w1.shape)}, w2={tuple(w2.shape)}"
    )


def _shuffle_w16a16_moec_weight(
    x: torch.Tensor,
    pack_fn,
    pad_dim: Optional[int] = None,
    padded_k: Optional[int] = None,
    transpose_output: bool = False,
) -> torch.Tensor:
    if pad_dim is not None:
        x = _pad_tensor_dim(x, pad_dim, padded_k)
    out_shape = (
        (x.shape[0], x.shape[2], x.shape[1])
        if transpose_output
        else tuple(x.shape)
    )
    return torch.stack([pack_fn(x[i]) for i in range(x.shape[0])]).view(*out_shape)


def _try_get_moe_c_config(
    quant_type: str,
    m: int,
    e: int,
    n: int,
    block_size: int,
    k: Optional[int] = None,
    top_k: Optional[int] = None,
    indtype: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Look up a tuned moe_c config.

    ``indtype`` is the channelwise CSV key (fp8_w8a8/int8_w8a8/int8_w4a8
    only; other families ignore it) and must match the execute-time
    ``str(compute_dtype)`` lookup.
    """
    try:
        from .fused_moe_c import get_moe_configs_marlin

        if quant_type == MoeQuantType.W4A16:
            configs = get_moe_configs_marlin(
                E=e,
                N=n,
                dtype="int4_w4a16",
                is_bottom=False,
                use_moe_wna16_cuda=True,
            )
        elif quant_type == MoeQuantType.WFP4A16:
            configs = get_moe_configs_marlin(
                E=e,
                N=n,
                dtype="fp4_w4a16",
                is_bottom=False,
                use_moe_wna16_cuda=True,
            )
        elif quant_type == MoeQuantType.W8A8:
            configs = get_moe_configs_marlin(
                E=e,
                N=n,
                dtype="int8_w8a8",
                is_bottom=False,
                use_moe_wna16_cuda=True,
                K=k,
                top_k=top_k,
                indtype=indtype,
            )
        elif quant_type == MoeQuantType.FP8_W8A8:
            if k is not None and (block_size is None or block_size == 0):
                padded_k = _pad_fp8_w8a8_moec_k(k)
                configs = get_moe_configs_marlin(
                    E=e,
                    N=n,
                    dtype="fp8_w8a8",
                    is_bottom=False,
                    use_moe_wna16_cuda=True,
                    K=padded_k,
                    top_k=top_k,
                    indtype=indtype,
                )
                if configs is None:
                    return None
                config = dict(_pick_closest_config(configs, m))
                if padded_k != k:
                    config = dict(config)
                    config.update({"ORIGINAL_K": k, "PADDED_K": padded_k})
                return config
            configs = get_moe_configs_marlin(
                E=e,
                N=n,
                dtype="fp8_w8a8",
                is_bottom=False,
                use_moe_wna16_cuda=True,
                top_k=top_k,
                indtype=indtype,
            )
        elif quant_type == MoeQuantType.W4A8:
            configs = get_moe_configs_marlin(
                E=e,
                N=n,
                dtype="int8_w4a8",
                block_n = block_size,
                block_k = block_size,
                is_bottom=False,
                use_moe_wna16_cuda=True,
                K=k,
                top_k=top_k,
                indtype=indtype,
            )
        elif quant_type == MoeQuantType.WFP4A8:
            configs = get_moe_configs_marlin(
                E=e,
                N=n,
                dtype="fp4_w4a8",
                block_n=0 if block_size else None,
                block_k=block_size,
                is_bottom=False,
                use_moe_wna16_cuda=True,
                K=k,
            )
        elif quant_type == MoeQuantType.INT8_W8A16:
            configs = get_moe_configs_marlin(
                E=e,
                N=n,
                dtype="int8_w8a16",
                block_n=0,
                block_k=block_size if block_size else 0,
                is_bottom=False,
                use_moe_wna16_cuda=True,
            )
        elif quant_type == MoeQuantType.W16A16:
            if k is None:
                return None
            def get_w16a16_pair_config(gemm1_k: int, gemm2_n: int):
                gemm1_configs = get_moe_configs_marlin(
                    E=e,
                    N=2 * n,
                    dtype="w16a16",
                    is_bottom=False,
                    use_moe_wna16_cuda=True,
                    K=gemm1_k,
                )
                gemm2_configs = get_moe_configs_marlin(
                    E=e,
                    N=gemm2_n,
                    dtype="w16a16",
                    is_bottom=True,
                    use_moe_wna16_cuda=True,
                    K=n,
                )
                if gemm1_configs is None or gemm2_configs is None:
                    return None
                return {
                    "GEMM1_CONFIG": _pick_closest_config(gemm1_configs, m),
                    "GEMM2_CONFIG": _pick_closest_config(gemm2_configs, m),
                }

            if k % 32 == 0:
                return get_w16a16_pair_config(k, k)

            padded_k = _pad_w16a16_moec_k(k)
            configs = get_w16a16_pair_config(padded_k, padded_k)
            if configs is None:
                return None
            configs.update({"ORIGINAL_K": k, "PADDED_K": padded_k})
            return configs
        else:
            return None

        if configs is None:
            return None
        return _pick_closest_config(configs, m)
    except Exception as exc:
        logger.warning(
            "moe_c config lookup failed for %s: %s",
            quant_type,
            exc,
            exc_info=True,
        )
        logger.debug("moe_c config lookup failed for %s: %s", quant_type, exc)
        return None


def _try_get_asm_config(
    quant_type: str,
    m: int,
    e: int,
    n: int,
    k: int,
    top_k: int,
    block_size: Optional[int],
    use_shuffle: int = 0,
) -> Optional[Dict[str, Any]]:
    try:
        from .fused_moe_asm_wna16 import get_moe_asm_solution, MoeQuantType as AsmMoeQuantType
        from .jit.utils.chip_info import get_gfx

        arch = get_gfx()
        asm_token = min(m, 65536)  # 与chunksize取小
        config_model_dim = k
        config_use_shuffle = use_shuffle
        padded_k = None
        if quant_type in (MoeQuantType.W8A8, MoeQuantType.FP8_W8A8):
            padded_k = k if k % 64 == 0 else ((k + 63) // 64) * 64
            if padded_k != k:
                config_model_dim = padded_k
                config_use_shuffle = 1

        def maybe_record_asm_padding(config: Dict[str, Any]) -> Dict[str, Any]:
            if padded_k is not None and padded_k != k:
                config = dict(config)
                config.update({
                    "ORIGINAL_K": k,
                    "PADDED_K": padded_k,
                    "USE_SHUFFLE": config_use_shuffle,
                })
            return config

        if quant_type == MoeQuantType.W4A16:
            from .fused_moe_asm_wna16 import decode_sol_w4a16, decode_sol_w4a16_gw32
            if block_size == 32:
                if top_k > 8 or n != 256 or k != 7168:
                    return None
                if arch == "gfx92a":
                    return None
                return decode_sol_w4a16_gw32()

            solution = get_moe_asm_solution(
                arch=arch,
                token=asm_token,
                inter_dim=n,
                model_dim=k,
                expert=e,
                topk=top_k,
                quant_type=AsmMoeQuantType.INT4_W4A16,
                use_shuffle=use_shuffle,
            )
            if solution == "default":
                return None
            return decode_sol_w4a16(solution)

        if quant_type == MoeQuantType.W8A8:
            from .fused_moe_asm_wna16 import decode_sol_0
            asm_quant_type = AsmMoeQuantType.INT8_W8A8_C if (block_size == 0 or block_size is None) else AsmMoeQuantType.INT8_W8A8
            solution = get_moe_asm_solution(
                arch=arch,
                token=asm_token,
                inter_dim=n,
                model_dim=config_model_dim,
                expert=e,
                topk=top_k,
                quant_type=asm_quant_type,
                use_shuffle=config_use_shuffle,
            )
            if solution == "default":
                return None
            return maybe_record_asm_padding(decode_sol_0(solution, config_use_shuffle))

        if quant_type == MoeQuantType.FP8_W8A8:
            from .fused_moe_asm_wna16 import decode_sol_0
            asm_quant_type = AsmMoeQuantType.F8_W8A8_C if (block_size == 0 or block_size is None) else AsmMoeQuantType.F8_W8A8
            solution = get_moe_asm_solution(
                arch=arch,
                token=asm_token,
                inter_dim=n,
                model_dim=config_model_dim,
                expert=e,
                topk=top_k,
                quant_type=asm_quant_type,
                use_shuffle=config_use_shuffle,
            )
            if solution == "default":
                return None
            return maybe_record_asm_padding(decode_sol_0(solution, config_use_shuffle))

        if quant_type == MoeQuantType.W4A8 and block_size == 64:
            from .fused_moe_asm_wna16 import decode_sol_0
            solution = get_moe_asm_solution(
                arch=arch, token=asm_token, inter_dim=n, model_dim=k,
                expert=e, topk=top_k, quant_type=AsmMoeQuantType.INT4_W4A8,
                use_shuffle=use_shuffle,
            )
            return None if solution == "default" else decode_sol_0(solution)

        if quant_type == MoeQuantType.W16A16:
            from .fused_moe_asm_wna16 import decode_sol_0

            solution = get_moe_asm_solution(
                arch=arch,
                token=asm_token,
                inter_dim=n,
                model_dim=k,
                expert=e,
                topk=top_k,
                quant_type=AsmMoeQuantType.NO_QUANT,
                use_shuffle=use_shuffle,
            )
            if solution == "default":
                return None
            return decode_sol_0(solution)

        return None
    except Exception as exc:
        logger.debug("ASM config lookup failed for %s: %s", quant_type, exc)
        return None


def _try_get_triton_config(
    quant_type: str,
    m: int,
    e: int,
    n: int,
    block_size: int,
    k: int,
    top_k: int,
) -> Optional[Dict[str, Any]]:
    try:
        from boltops.fused_moe.triton.moe_config_utils import get_moe_configs as triton_get_moe_configs

        if quant_type == MoeQuantType.W16A16:
            return {}  # Non-quantized; no tuned config lookup needed

        dtype_name = {
            MoeQuantType.W4A16: "int4_w4a16",
            MoeQuantType.WFP4A16: "fp4_w4a16",
            MoeQuantType.WFP4A8: "fp4_w4a8",
            MoeQuantType.W4A8: "int4_w4a8",
            MoeQuantType.W8A8: "int8_w8a8",
            MoeQuantType.FP8_W8A8: "fp8_w8a8",
            MoeQuantType.INT8_W8A16: "int8_w8a16",
            MoeQuantType.FP8_W8A16: "fp8_w8a16",
        }.get(quant_type)
        if dtype_name is None:
            return None

        configs = triton_get_moe_configs(
            E=e,
            N=n,
            dtype=dtype_name,
            block_n=block_size if block_size and quant_type in (MoeQuantType.W8A8, MoeQuantType.FP8_W8A8) else 0,
            block_k=block_size if block_size else 0,
            is_bottom=False,
            K=k,
            top_k=top_k,
        )
        if configs is None:
            return None
        return _pick_closest_config(configs, m)
    except Exception as exc:
        logger.debug("Triton config lookup failed for %s: %s", quant_type, exc)
        return None


def _try_get_triton_default_config(
    quant_type: str,
    m: int,
    e: int,
    n: int,
    block_size: int,
    k: int,
    top_k: int,
) -> Optional[Dict[str, Any]]:
    # Only enable quant types represented by aiter_moe's Triton execution flags.
    dtype_names = {
        MoeQuantType.W16A16: None,
        MoeQuantType.W4A16: "int4_w4a16",
        MoeQuantType.W4A8: "int4_w4a8",
        MoeQuantType.W8A8: "int8_w8a8",
        MoeQuantType.FP8_W8A8: "fp8_w8a8",
        MoeQuantType.INT8_W8A16: "int8_w8a16",
    }
    if quant_type not in dtype_names:
        return None
    try:
        if quant_type == MoeQuantType.FP8_W8A8:
            from .utility import dtypes

            # The uint8 FP8 fallback is not an executable Triton FP8 dtype.
            if dtypes.fp8 == torch.uint8:
                return None
        from boltops.fused_moe.triton.moe_config_utils import get_default_config

        block_n = block_size if quant_type in (MoeQuantType.W8A8, MoeQuantType.FP8_W8A8) else 0
        config = get_default_config(
            m, e, n, k, top_k, dtype_names[quant_type],
            is_marlin=False,
            block_shape=[block_n, block_size] if block_size else None,
        )
        if not config:
            return None
        logger.info(
            "No tuned MoE config found; using default untuned Triton config for MoE execution: "
            "quant_type=%s, M=%s, E=%s, N=%s, K=%s, top_k=%s, block_size=%s, config=%s",
            quant_type, m, e, n, k, top_k, block_size, config,
        )
        return config
    except Exception as exc:
        logger.debug("Triton default config lookup failed for %s: %s", quant_type, exc)
        return None


def _try_get_ck_config(
    quant_type: str,
    m: int,
    e: int,
    n: int,
    k: int,
    top_k: int,
    block_shape: Optional[List[int]],
) -> Optional[Dict[str, Any]]:
    try:
        from .fused_moe_ck import get_moe_ck_solution_id, MoeQuantType as CkMoeQuantType
        from .jit.utils.chip_info import get_gfx
        
        if quant_type == MoeQuantType.W16A16:
            ck_quant_type = CkMoeQuantType.NO_QUANT
        elif quant_type == MoeQuantType.W8A8 or quant_type == MoeQuantType.FP8_W8A8:
            ck_quant_type = CkMoeQuantType.INT8_W8A8
        else:
            return None

        arch = get_gfx()
        q_size_n = block_shape[0] if block_shape is not None else 0
        q_size_k = block_shape[1] if block_shape is not None else 0
        solution_id = get_moe_ck_solution_id(
            arch,
            ck_quant_type,
            m,
            n,
            k,
            e,
            top_k,
            q_size_n,
            q_size_k,
        )
        return {"solution_id": solution_id}
    except Exception as exc:
        logger.debug("CK config lookup failed for %s: %s", quant_type, exc)
        return None


def get_aiter_moe_config(
    M: int,     # Number of tokens (input sequence length)
    E: int,     # Number of experts
    N1: int,    # GEMM1 output dimension: gated = (intermediate_size * 2), non-gated = intermediate_size
    N2: int,    # GEMM2 output dimension, typically equal to hidden_size
    K: int,     # GEMM1 input dimension, typically equal to hidden_size;  for GEMM2, K typically equal to (moe_intermediate_size / TP)
    top_k: int,
    block_size: int,
    dtype: torch.dtype,
    quant_type: str,
    activation: str = "silu",  # "silu"/"situ"/"gelu"/"relu2"/"swigluoai"/"swiglustep/gelu_tanh"...
    gated: Optional[bool] = None,  # True=GLU-gated (N1=2*inter), False=non-gated (N1=inter); None=auto from activation
    spec_sol_type: Optional[MoeSolutionType] = None,     # If specified, only try this backend; otherwise try all backends in priority order.
    use_shuffle: int = 0,
) -> Tuple[bool, AiterMoeConfig]:
    """Get the best backend config for a MOE problem.

    Currently supported quant types:
    - ``MoeQuantType.W16A16`` (non-quantized)
    - ``MoeQuantType.W4A16``
    - ``MoeQuantType.W8A8`` (int8)
    - ``MoeQuantType.FP8_W8A8`` (fp8)
    - ``MoeQuantType.W4A8``
    - ``MoeQuantType.INT8_W8A16`` (int8 weight, fp16/bf16 activation)

    Backend priority:
    - ``w16a16``: asm > triton > moe_c
    - ``w4a16``: moe_c > asm > triton
    - ``w8a8``: asm > moe_c > triton > ck
    - ``fp8_w8a8``: asm > moe_c > triton > ck
    - ``w4a8``: moe_c
    - ``int8_w8a16``: moe_c > triton (ASM kernel not available)
    - ``fp8_w8a16``: not yet implemented (raises NotImplementedError)

    If all candidates lack a tuned config, try an untuned Triton default.
    An explicit backend request stays restricted to that backend; explicit
    Triton requests can also use its default config.

    For non-gated MOE (e.g. Nemotron with ReLU² activation), pass
    ``gated=False`` (or let it auto-detect from ``activation="relu2"``)
    and set ``N1 = intermediate_size`` (not ``2 * intermediate_size``).
    """
    # Determine gating: explicit > auto-detect from activation
    if gated is None:
        gated = activation in ("silu", "situ", "gelu", "swigluoai", "swiglustep", "gelu_tanh")

    # For gated (GLU): N1 = 2 * intermediate_size, n = N1 // 2
    # For non-gated:   N1 = intermediate_size,     n = N1
    n = N1 // 2 if gated else N1
    block_shape = [0, block_size] if block_size else None
    config_dtype = _resolve_moe_c_config_dtype(dtype)

    # If a specific solution type is requested, only try that one
    if spec_sol_type is not None:
        # Only try the specified backend
        if spec_sol_type == MoeSolutionType.MOE_C:
            config = _try_get_moe_c_config(
                quant_type, M, E, n, block_size, K, top_k,
                indtype=str(config_dtype),
            )
        elif spec_sol_type == MoeSolutionType.ASM:
            config = _try_get_asm_config(quant_type, M, E, n, K, top_k, block_size, use_shuffle)
        elif spec_sol_type == MoeSolutionType.TRITON:
            config = _try_get_triton_config(quant_type, M, E, n, block_size, K, top_k)
            if config is None:
                config = _try_get_triton_default_config(quant_type, M, E, n, block_size, K, top_k)
        elif spec_sol_type == MoeSolutionType.CK:
            config = _try_get_ck_config(quant_type, M, E, n, K, top_k, block_shape)
        else:
            raise ValueError(f"Unsupported spec_sol_type: {spec_sol_type}")

        if config is not None:
            return True, AiterMoeConfig(
                quant_type=quant_type,
                solution_type=spec_sol_type,
                need_shuffle=((spec_sol_type == MoeSolutionType.MOE_C and (quant_type not in (MoeQuantType.W4A16, MoeQuantType.WFP4A16))) or (spec_sol_type == MoeSolutionType.ASM and bool(config.get("USE_SHUFFLE", use_shuffle)))),
                need_shuffle_scale=((spec_sol_type == MoeSolutionType.MOE_C) and (quant_type in (MoeQuantType.W4A16, MoeQuantType.WFP4A16))),
                config=config,
                dtype=config_dtype,
            )
        else:
            return False, AiterMoeConfig(quant_type=quant_type)


    if quant_type == MoeQuantType.W4A16:
        if dtype == torch.float16:
            candidates = [
                (MoeSolutionType.MOE_C, lambda: _try_get_moe_c_config(quant_type, M, E, n, block_size, K)),
                (MoeSolutionType.TRITON, lambda: _try_get_triton_config(quant_type, M, E, n, block_size, K, top_k)),
            ]
        elif dtype == torch.bfloat16:
            candidates = [
                (MoeSolutionType.ASM, lambda: _try_get_asm_config(quant_type, M, E, n, K, top_k, block_size, use_shuffle)),
                (MoeSolutionType.TRITON, lambda: _try_get_triton_config(quant_type, M, E, n, block_size, K, top_k)),
                (MoeSolutionType.MOE_C, lambda: _try_get_moe_c_config(quant_type, M, E, n, block_size)),
                
            ]
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    elif quant_type == MoeQuantType.WFP4A16:
        candidates = [
            (MoeSolutionType.MOE_C, lambda: _try_get_moe_c_config(quant_type, M, E, n, block_size)),
        ]
    elif quant_type in (MoeQuantType.W8A8, MoeQuantType.FP8_W8A8):
        if block_size is None or block_size == 0: # Channel wise
            candidates = [
                (MoeSolutionType.ASM, lambda: _try_get_asm_config(quant_type, M, E, n, K, top_k, block_size, use_shuffle)),
                (MoeSolutionType.MOE_C, lambda: _try_get_moe_c_config(quant_type, M, E, n, block_size, K, top_k, indtype=str(config_dtype))),
                (MoeSolutionType.TRITON, lambda: _try_get_triton_config(quant_type, M, E, n, block_size, K, top_k)),
                # (MoeSolutionType.CK, lambda: _try_get_ck_config(quant_type, M, E, n, K, top_k, block_shape)),
            ]
        else: # Block wise choose ASM
            candidates = [
                (MoeSolutionType.ASM, lambda: _try_get_asm_config(quant_type, M, E, n, K, top_k, block_size, use_shuffle)),
                (MoeSolutionType.TRITON, lambda: _try_get_triton_config(quant_type, M, E, n, block_size, K, top_k)),
            ]

    elif quant_type == MoeQuantType.W4A8:
        candidates = [
            (MoeSolutionType.MOE_C, lambda: _try_get_moe_c_config(quant_type, M, E, n, block_size, K, top_k, indtype=str(config_dtype))),
            (MoeSolutionType.TRITON, lambda: _try_get_triton_config(quant_type, M, E, n, block_size, K, top_k)),
            # (MoeSolutionType.ASM, lambda: _try_get_asm_config(quant_type, M, E, n, K, top_k)),
        ]
    elif quant_type == MoeQuantType.WFP4A8:
        candidates = [
            (MoeSolutionType.MOE_C, lambda: _try_get_moe_c_config(quant_type, M, E, n, block_size, K)),
        ]
    elif quant_type == MoeQuantType.INT8_W8A16:
        # ASM backend currently has no W8A16 kernel/CSV; skip ASM and use moe_c -> triton.
        candidates = [
            (MoeSolutionType.MOE_C, lambda: _try_get_moe_c_config(quant_type, M, E, n, block_size)),
            (MoeSolutionType.TRITON, lambda: _try_get_triton_config(quant_type, M, E, n, block_size, K, top_k)),
        ]
    elif quant_type == MoeQuantType.FP8_W8A16:
        # No backend currently implements FP8 weight + 16-bit activation MoE.
        raise NotImplementedError(
            "MoeQuantType.FP8_W8A16 is not yet supported by any aiter MOE backend (asm/moe_c/triton)."
        )
    elif quant_type == MoeQuantType.W16A16:
        candidates = [
            (MoeSolutionType.ASM, lambda: _try_get_asm_config(quant_type, M, E, n, K, top_k, None, use_shuffle)),
            (MoeSolutionType.TRITON, lambda: _try_get_triton_config(quant_type, M, E, n, block_size, K, top_k)),
            (MoeSolutionType.MOE_C, lambda: _try_get_moe_c_config(quant_type, M, E, n, block_size, K)),
            # (MoeSolutionType.CK, lambda: _try_get_ck_config(quant_type, M, E, n, K, top_k, block_shape)),
        ]
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")

    for solution_type, get_config in candidates:
        config = get_config()
        if config is not None:
            return True, AiterMoeConfig(
                quant_type=quant_type,
                solution_type=solution_type,
                need_shuffle=(((solution_type == MoeSolutionType.MOE_C) and (quant_type not in (MoeQuantType.W4A16, MoeQuantType.WFP4A16))) or (solution_type == MoeSolutionType.ASM and bool(config.get("USE_SHUFFLE", use_shuffle)))),
                need_shuffle_scale=((solution_type == MoeSolutionType.MOE_C) and (quant_type in (MoeQuantType.W4A16, MoeQuantType.WFP4A16))),
                config=config,
                dtype=config_dtype,
            )

    config = _try_get_triton_default_config(quant_type, M, E, n, block_size, K, top_k)
    if config is not None:
        return True, AiterMoeConfig(
            quant_type=quant_type,
            solution_type=MoeSolutionType.TRITON,
            config=config,
            dtype=config_dtype,
        )
    return False, AiterMoeConfig(quant_type=quant_type)


def aiter_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    moe_config: AiterMoeConfig,
    inplace: Optional[bool] = False,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    routed_scaling_factor: Optional[float] = 1.0,
    use_weight_shuffle: bool = False,
    output_dtype: Optional[torch.dtype] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
) -> torch.Tensor:
    """Execute MOE using the backend and quant type described by *moe_config*."""
    if moe_config.solution_type is None or moe_config.quant_type is None:
        raise ValueError(
            "moe_config has no valid solution_type/quant_type. "
            "Call get_aiter_moe_config first and check the status."
        )
    
    if output_dtype is None and moe_config.solution_type != MoeSolutionType.MOE_C:
        output_dtype = hidden_states.dtype
    
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    if _AITER_LOG_MOE_PARAM:
        tokens = hidden_states.shape[0] if hidden_states.dim() > 0 else None
        hidden_size = hidden_states.shape[-1] if hidden_states.dim() > 0 else None
        intermediate_size = None
        if w1.dim() >= 2:
            gated = activation in ("silu", "situ", "gelu", "swigluoai", "swiglustep", "gelu_tanh")
            intermediate_size = w1.shape[1] // 2 if gated else w1.shape[1]
        elif w2.dim() >= 3:
            intermediate_size = w2.shape[-1]
        topk = None
        if topk_ids.dim() > 0:
            topk = topk_ids.shape[-1]
        elif topk_weights.dim() > 0:
            topk = topk_weights.shape[-1]
        logger.info(
            f"quant={moe_config.quant_type}, "
            f"backend={moe_config.solution_type}, "
            f"(tokens,interM_size,hidden_size)=({tokens},{intermediate_size},{hidden_size}), "
            f"num_experts={global_num_experts}, "
            f"topk={topk}, "
            f"inplace={inplace}, "
            f"act={activation}, "
            f"block_shape={block_shape}, "
            f"in_dtype={hidden_states.dtype}, "
            f"out_dtype={output_dtype}, "
            f"routed_scaling_factor={routed_scaling_factor}, "
            f"use_shuffle={use_weight_shuffle}"
        )

    use_int4_w4a16 = moe_config.quant_type == MoeQuantType.W4A16
    use_fp4_w4a16 = moe_config.quant_type == MoeQuantType.WFP4A16
    use_fp4_w4a8 = moe_config.quant_type == MoeQuantType.WFP4A8
    use_int8_w8a8 = moe_config.quant_type == MoeQuantType.W8A8
    use_fp8_w8a8 = moe_config.quant_type == MoeQuantType.FP8_W8A8
    use_int8_w4a8 = moe_config.quant_type == MoeQuantType.W4A8
    use_int8_w8a16 = moe_config.quant_type == MoeQuantType.INT8_W8A16
    use_w16a16 = moe_config.quant_type == MoeQuantType.W16A16

    if moe_config.solution_type == MoeSolutionType.MOE_C:
        from .fused_moe_c import moe_c_fused_experts

        float_dtypes = (torch.float16, torch.bfloat16, torch.float32)
        if hidden_states.dtype in float_dtypes:
            compute_dtype = hidden_states.dtype
            if moe_config.dtype is not None and moe_config.dtype != compute_dtype:
                raise ValueError(
                    "aiter_moe(moe_c): moe_config was selected for "
                    f"{moe_config.dtype}, but hidden_states has {compute_dtype}; "
                    "call get_aiter_moe_config again with the runtime dtype"
                )
            if moe_config.dtype is None:
                logger.warning(
                    "aiter_moe(moe_c): moe_config.dtype is None; dtype check "
                    "skipped, execute-time config lookup may miss"
                )
        else:
            compute_dtype = moe_config.dtype or torch.bfloat16

        if output_dtype is None:
            output_dtype = compute_dtype
        if output_dtype not in float_dtypes:
            raise ValueError(
                "aiter_moe(moe_c): output_dtype must be fp16, bf16, or fp32, "
                f"got {output_dtype}"
            )
        if output_dtype != compute_dtype:
            raise ValueError(
                "aiter_moe(moe_c): output_dtype must equal moe_c compute dtype; "
                f"got output_dtype={output_dtype}, compute_dtype={compute_dtype}"
            )

        moe_c_kwargs = {}
        if use_fp8_w8a8 and moe_config.config:
            padded_k = moe_config.config.get("PADDED_K")
            if padded_k is not None:
                padded_k = int(padded_k)
                w2_scale = _pad_tensor_dim(w2_scale, 1, padded_k) if w2_scale is not None else None
                moe_c_kwargs["fp8_w8a8_config"] = [
                    int(moe_config.config["ORIGINAL_K"]),
                    padded_k,
                ]
        if use_w16a16 and moe_config.config:
            gemm1_config = moe_config.config.get("GEMM1_CONFIG")
            gemm2_config = moe_config.config.get("GEMM2_CONFIG")
            if gemm1_config and gemm2_config:
                bm1 = int(gemm1_config["BLOCK_SIZE_M"])
                bm2 = int(gemm2_config["BLOCK_SIZE_M"])
                w16a16_config = [
                    int(gemm1_config["MODE"]),
                    bm1,
                    int(gemm1_config.get("DELTA", 1)),
                    int(gemm2_config["MODE"]),
                    bm2,
                    int(gemm2_config.get("DELTA", bm1 // bm2)),
                ]
                padded_k = moe_config.config.get("PADDED_K")
                if padded_k is not None:
                    w16a16_config.extend([
                        int(moe_config.config["ORIGINAL_K"]),
                        int(padded_k),
                    ])
                moe_c_kwargs["w16a16_config"] = w16a16_config
        return moe_c_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=inplace,
            use_int4_w4a16=use_int4_w4a16,
            use_fp4_w4a16=use_fp4_w4a16,
            use_fp4_w4a8=use_fp4_w4a8,
            use_int8_w8a8=use_int8_w8a8,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w4a8=use_int8_w4a8,
            use_int8_w8a16=use_int8_w8a16,
            use_w16a16=use_w16a16,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_limit=gemm1_limit,
            compute_dtype=compute_dtype,
            **moe_c_kwargs,
        )

    if moe_config.solution_type == MoeSolutionType.ASM:
        from .fused_moe_asm_wna16 import fused_experts_asm_impl
        if output_dtype == torch.float32:
            raise ValueError(
                "aiter_moe(asm): fp32 output is unsupported; asm supports "
                "only fp16 and bf16 output"
            )
        per_channel_quant = True if block_shape is None else False
        cfg = moe_config.config
        solution_id = f"{cfg['SOL_ID1']}+{cfg['SOL_ID2']}"
        padded_k = cfg.get("PADDED_K") if cfg else None
        if padded_k is not None:
            padded_k = int(padded_k)
            w2_scale = _pad_tensor_dim(w2_scale, 1, padded_k) if w2_scale is not None else None
        return fused_experts_asm_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            dtype=output_dtype,
            inplace=inplace,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a8=use_int8_w8a8,
            use_fp8_w8a8=use_fp8_w8a8,
            activation=activation,
            per_channel_quant = per_channel_quant,
            use_int8_w4a8=use_int8_w4a8,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            use_shuffle=use_weight_shuffle,
            solution_id=solution_id,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_limit=gemm1_limit,
            padded_k=padded_k,
        )

    if moe_config.solution_type == MoeSolutionType.TRITON:
        from boltops.fused_moe.triton.fused_moe import fused_experts_impl

        # W8A8 / W8A16 / W4A8 channel-wise requires per_channel_quant=True.
        per_channel_quant = (use_int8_w8a8 or use_fp8_w8a8 or use_int8_w8a16 or use_int8_w4a8) and block_shape is None

        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            output_dtype=output_dtype,
            inplace=inplace,
            activation=activation,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            use_int4_w4a8=use_int8_w4a8,
            per_channel_quant=per_channel_quant,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_limit=gemm1_limit
        )

    if moe_config.solution_type == MoeSolutionType.CK:
        from .fused_moe_ck import run_fused_experts_ck_impl

        solution_id = moe_config.config["solution_id"]
        return run_fused_experts_ck_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            odtype=output_dtype,
            inplace=inplace,
            use_int8_w8a8=use_int8_w8a8,
            use_fp8_w8a8=use_fp8_w8a8,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            use_shuffle=use_weight_shuffle,
            routed_scaling_factor=routed_scaling_factor,
            solution_id=solution_id,
        )

    raise ValueError(f"Unknown solution_type: {moe_config.solution_type}")


# ---------------------------------------------------------------------------
# 共享重排核心
#
#   * aiter_moe_shfl_weight / aiter_moe_shfl_scale 是它的薄封装（aiter_moe 调用方零改动）
#   * fused_moe_prepare_weights 也调用它，保证两条路径逐位一致
# 布局词汇：raw / asm_shuffle / marlin；量化家族词汇见 _shuffle_quant_kind。
# ---------------------------------------------------------------------------

RAW_LAYOUT = "raw"
MARLIN_LAYOUT = "marlin"
ASM_SHUFFLE_LAYOUT = "asm_shuffle"

_MOE_QUANT_TO_KIND = {
    MoeQuantType.W8A8: "w8a8_int8",
    MoeQuantType.FP8_W8A8: "w8a8_fp8",
    MoeQuantType.INT8_W8A16: "w8a16",
    MoeQuantType.W4A16: "w4a16",
    MoeQuantType.W4A8: "w4a8",
    MoeQuantType.WFP4A16: "wfp4a16",
    MoeQuantType.WFP4A8: "wfp4a8",
    MoeQuantType.W16A16: "w16a16",
}


def _shuffle_quant_kind(quant_type) -> Optional[str]:
    """aiter.moe.MoeQuantType -> 共享核心使用的家族词汇。"""
    return _MOE_QUANT_TO_KIND.get(quant_type)


def _shared_shuffle_impl(w1, w2, scale1=None, scale2=None, *,
                         quant_kind, layout=MARLIN_LAYOUT, padded_k=None,
                         block_shape=None):
    """唯一重排实现：权重 + scale + 补 K 都只在这里做一次。返回四元组。

    Args:
        quant_kind: ``w8a8_int8`` / ``w8a8_fp8`` / ``w8a16`` / ``w4a16`` /
            ``w4a8`` / ``wfp4a16`` / ``wfp4a8`` / ``w16a16`` / ``none``。
        layout: ``raw`` / ``asm_shuffle`` / ``marlin``。
        padded_k: 补到的 model_dim 长度（asm 与 moe_c 的 w8a8_fp8 需要；None 表示不补）。
        block_shape: fp4 家族区分 channelwise / groupwise32 的依据。

    Returns:
        ``(shuffled_w1, shuffled_w2, shuffled_scale1, shuffled_scale2)``；
        对应输入为 ``None`` 时该项仍为 ``None``。

    scale 规则：
      * ``asm_shuffle``：``w2_scale`` 第 1 维按 ``padded_k`` 补零（与 aiter_moe
        的行为一致），``w1_scale`` 原样；
      * ``marlin``：只有 w4a16 / wfp4a16 需要 marlin / e8m0 重排，其余透传；
      * ``raw``：权重与 scale 都不动。
    """
    from .ops.shuffle import (
        asm_shuffle_weight_b8,
        moe_layout_shuffle_gemm2,
        w4a16_marlin_scale,
        w4a16_marlin_weight_1,
        w4a16_marlin_weight_2,
        w4a8_moe_layout_shuffle,
        w4a8_moe_repack_shuffle,
        w8a16_marlin_weight_1,
        w8a16_marlin_weight_2,
        w16a16_marlin_weight,
        w16a16_marlin_weight_k_n,
        wfp4a16_e8m0_scale,
        wfp4a8_moe_layout_shuffle,
    )

    if layout == RAW_LAYOUT:
        return w1, w2, scale1, scale2

    if layout == ASM_SHUFFLE_LAYOUT:
        # asm 只服务 none / w8a8 / w8a8_block；其它家族不该走到这里
        def _pad(x, dim):
            return _pad_tensor_dim(x, dim, padded_k)

        s1 = asm_shuffle_weight_b8(_pad(w1, 2), stage=1) if w1 is not None else None
        s2 = asm_shuffle_weight_b8(_pad(w2, 1), stage=2) if w2 is not None else None
        if padded_k is not None and scale2 is not None:
            scale2 = _pad_tensor_dim(scale2, 1, int(padded_k))
        return s1, s2, scale1, scale2

    # ---- marlin（moe_c）----
    if quant_kind in ("w8a8_int8", "w8a8_fp8"):
        if quant_kind == "w8a8_fp8" and padded_k is not None:
            k1 = int(padded_k)
            o1 = moe_layout_shuffle_gemm2(_pad_tensor_dim(w1, 2, k1)) \
                .contiguous().view(w1.shape[0], w1.shape[1], k1)
            o2 = moe_layout_shuffle_gemm2(_pad_tensor_dim(w2, 1, k1)) \
                .contiguous().view(w2.shape[0], k1, w2.shape[2])
        else:
            o1 = moe_layout_shuffle_gemm2(w1).contiguous().view(*w1.shape) \
                if w1 is not None else None
            o2 = moe_layout_shuffle_gemm2(w2).contiguous().view(*w2.shape) \
                if w2 is not None else None
        return o1, o2, scale1, scale2

    if quant_kind == "w8a16":
        o1 = w8a16_marlin_weight_1(w1).contiguous() if w1 is not None else None
        o2 = w8a16_marlin_weight_2(w2).contiguous() if w2 is not None else None
        return o1, o2, scale1, scale2

    if quant_kind == "w4a16":
        o1 = w4a16_marlin_weight_1(w1).view(-1).view(torch.uint8).view(*w1.shape) \
            if w1 is not None else None
        o2 = w4a16_marlin_weight_2(w2).view(-1).view(torch.uint8).view(*w2.shape) \
            if w2 is not None else None
        if scale1 is not None:
            scale1 = w4a16_marlin_scale(scale1)
        if scale2 is not None:
            scale2 = w4a16_marlin_scale(scale2)
        return o1, o2, scale1, scale2

    if quant_kind == "w4a8":
        def _repack(x):
            if x is None:
                return None
            out = w4a8_moe_repack_shuffle(x).contiguous()
            # moe_c 的 w4a8 marlin op 需要 signed char 容器
            return out.view(torch.int8) if out.dtype == torch.uint8 else out

        return _repack(w1), _repack(w2), scale1, scale2

    if quant_kind == "wfp4a8":
        fn = wfp4a8_moe_layout_shuffle if block_shape else w4a8_moe_layout_shuffle

        def _fp4(x):
            if x is None:
                return None
            return torch.stack([fn(x[i]) for i in range(x.shape[0])]) \
                .contiguous().view(*x.shape)

        return _fp4(w1), _fp4(w2), scale1, scale2

    if quant_kind == "wfp4a16":
        # fp4x2 权重按产出格式原样消费；e8m0 scale 需要重排
        if scale1 is not None:
            scale1 = wfp4a16_e8m0_scale(scale1)
        if scale2 is not None:
            scale2 = wfp4a16_e8m0_scale(scale2)
        return w1, w2, scale1, scale2

    if quant_kind == "w16a16":
        # 目前没有可达路由（moe_c 缺 is_bottom=True 的配置），保留实现供未来放行
        if _w16a16_moec_weight_layout(w1, w2) == "canonical":
            o1 = _shuffle_w16a16_moec_weight(
                w1, w16a16_marlin_weight, pad_dim=2, padded_k=padded_k)
            o2 = _shuffle_w16a16_moec_weight(
                w2, w16a16_marlin_weight, pad_dim=1, padded_k=padded_k)
            return o1, o2, scale1, scale2
        o1 = _shuffle_w16a16_moec_weight(
            w1, w16a16_marlin_weight_k_n, pad_dim=1,
            padded_k=padded_k, transpose_output=True)
        o2 = _shuffle_w16a16_moec_weight(
            w2, w16a16_marlin_weight_k_n, pad_dim=2,
            padded_k=padded_k, transpose_output=True)
        return o1, o2, scale1, scale2

    if quant_kind == "none":
        raise ValueError(
            "shared shuffle: layout=marlin is not valid for the non-quantized "
            "family (moe_c has no W16A16 stage-2 configs)")

    raise ValueError(f"shared shuffle: unsupported quant kind {quant_kind!r}")


def aiter_moe_shfl_weight(
    w1: Optional[torch.Tensor],
    w2: Optional[torch.Tensor],
    moe_config: AiterMoeConfig,
    block_shape: Optional[list] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Shuffle weights according to the backend and quant type in *moe_config*.

    Each backend may expect weights in a specific tiled or packed layout.
    This function applies the correct shuffle for the backend/quant_type
    combination so the caller can pass the result directly to
    :func:`aiter_moe`.

    Args:
        w1: Weight for GEMM1 (gate+up projection), shape ``[E, N1, K]``
            or quantized variant. May be ``None`` to skip.
        w2: Weight for GEMM2 (down projection), shape ``[E, K, N2]``
            or quantized variant. May be ``None`` to skip.
        moe_config: Config returned by :func:`get_aiter_moe_config`.
        block_shape: Groupwise block shape. ``None`` means channelwise.

    Returns:
        Tuple of ``(shuffled_w1, shuffled_w2)`` ready for :func:`aiter_moe`.
        Corresponding entry is ``None`` when input is ``None``.
    """
    quant_type = moe_config.quant_type
    sol_type = moe_config.solution_type

    # 共享核心（唯一重排实现）：fused_moe 准备函数与本函数都走这里。
    if sol_type == MoeSolutionType.MOE_C:
        kind = _shuffle_quant_kind(quant_type)
        if kind is None:
            raise ValueError(
                f"Shuffle not supported for quant_type={quant_type} "
                f"with solution_type={sol_type}")
        padded_k = (moe_config.config or {}).get("PADDED_K")
        padded_k = int(padded_k) if padded_k is not None else None
        o1, o2, _s1, _s2 = _shared_shuffle_impl(
            w1, w2, quant_kind=kind, padded_k=padded_k, block_shape=block_shape,
            layout=MARLIN_LAYOUT)
        return o1, o2

    if sol_type == MoeSolutionType.ASM:
        padded_k = (moe_config.config or {}).get("PADDED_K")
        padded_k = int(padded_k) if padded_k is not None else None
        o1, o2, _s1, _s2 = _shared_shuffle_impl(
            w1, w2, quant_kind="none", padded_k=padded_k,
            block_shape=block_shape, layout=ASM_SHUFFLE_LAYOUT)
        return o1, o2

    if sol_type == MoeSolutionType.CK:
        from .ops.shuffle import ck_shuffle_weight, ck_shuffle_weight_down
        s1 = ck_shuffle_weight(w1) if w1 is not None else None
        s2 = ck_shuffle_weight_down(w2) if w2 is not None else None
        return s1, s2

    if sol_type == MoeSolutionType.TRITON:
        return w1, w2

    raise ValueError(f"Unknown solution_type: {sol_type}")

def aiter_moe_shfl_scale(
    scale1: Optional[torch.Tensor],
    scale2: Optional[torch.Tensor],
    moe_config: AiterMoeConfig,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Shuffle scale according to the backend and quant type in *moe_config*.

    Each backend may expect weights in a specific tiled or packed layout.
    This function applies the correct shuffle for the backend/quant_type
    combination so the caller can pass the result directly to
    :func:`aiter_moe`.

    Args:
        scale1: Scale for GEMM1 (gate+up projection), shape ``[E, N1, K]``
            or quantized variant. May be ``None`` to skip.
        scale1: Scale for GEMM2 (down projection), shape ``[E, K, N2]``
            or quantized variant. May be ``None`` to skip.
        moe_config: Config returned by :func:`get_aiter_moe_config`.

    Returns:
        Tuple of ``(shuffled_scale1, shuffled_scale2)`` ready for :func:`aiter_moe`.
        Corresponding entry is ``None`` when input is ``None``.
    """
    quant_type = moe_config.quant_type
    sol_type = moe_config.solution_type

    if sol_type == MoeSolutionType.MOE_C:
        kind = _shuffle_quant_kind(quant_type)
        if kind not in ("w4a16", "wfp4a16"):
            raise ValueError(
                f"Shuffle scale not supported for quant_type={quant_type} "
                f"with solution_type={sol_type}"
            )
        _w1, _w2, s1, s2 = _shared_shuffle_impl(
            None, None, scale1, scale2, quant_kind=kind, layout=MARLIN_LAYOUT)
        return s1, s2

    raise ValueError(f"Shuffle scale Unknown solution_type: {sol_type}")


def get_aiter_moe_config_w4a16(
    M: int,
    E: int,
    N1: int,
    N2: int,
    K: int,
    top_k: int,
    block_size: int,
    dtype: torch.dtype,
) -> Tuple[bool, AiterMoeConfig]:
    """Backward-compatible wrapper for w4a16 config lookup."""
    return get_aiter_moe_config(M, E, N1, N2, K, top_k, block_size, dtype, MoeQuantType.W4A16)


def aiter_moe_w4a16(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    moe_config: AiterMoeConfig,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> torch.Tensor:
    """Backward-compatible wrapper for w4a16 execution."""
    return aiter_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        moe_config=moe_config,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )
