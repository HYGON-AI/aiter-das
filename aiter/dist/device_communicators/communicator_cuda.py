# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import torch
from torch.distributed import ProcessGroup

should_nccl_symm_mem_allreduce = False
from aiter.dist.parallel_state import is_global_first_rank
from aiter import logger
from aiter.utility.dtypes import FP8_DTYPES
from .base_device_communicator import DeviceCommunicatorBase


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class CudaCommunicator(DeviceCommunicatorBase):
    # Upstream AITER uses the message size to choose the fused AR+RMSNorm
    # implementation. Keep an override for focused A/B testing and emergency
    # rollback without rebuilding the extension.
    _ar_1stage_override = {"1": True, "0": False}.get(
        os.environ.get("AITER_AR_1STAGE", "")
    )
    _ar_1stage_max_kb = int(os.environ.get("AITER_AR_1STAGE_MAX_KB", -1))

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        if "tp" not in unique_name:
            # custom allreduce or torch symm mem can be used only by tp
            use_custom_allreduce = False
            use_torch_symm_mem = False
        else:
            from aiter.dist.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE

            use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
            use_torch_symm_mem = False

        self.use_custom_allreduce = use_custom_allreduce
        self.use_torch_symm_mem = use_torch_symm_mem

        # lazy import to avoid documentation build error
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        # from aiter.dist.device_communicators.symm_mem import SymmMemCommunicator

        self.pynccl_comm = None
        if self.world_size > 1:
            from aiter.dist.device_communicators.communicator_pynccl import (
                PyNcclCommunicator,
            )

            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )
            # if is_symmetric_memory_enabled():
            #     register_nccl_symmetric_ops(self.pynccl_comm)

        self.ca_comm: CustomAllreduce | None = None
        self.qr_comm = None
        self.symm_mem_comm = None
        # if use_torch_symm_mem and current_platform.is_cuda():
        #     self.symm_mem_comm = SymmMemCommunicator(
        #         group=self.cpu_group,
        #         device=self.device,
        #     )

        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            # AITER_AR_ENABLE_REG_CAPTURE controls whether inputs captured
            # inside a CUDA graph are assumed to already live in the
            # pre-registered IPC buffer (True, default), or whether the
            # in-graph all-reduce should fall back to the unregistered
            # copy-in path (False). Set this to "0" when callers cannot
            # guarantee that captured input pointers were registered via
            # ``CustomAllreduce.register_buffer``.
            enable_register_for_capturing = _env_flag(
                "AITER_AR_ENABLE_REG_CAPTURE", default=True
            )
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                enable_register_for_capturing=enable_register_for_capturing,
                # symm_mem_enabled=(
                #     self.symm_mem_comm is not None and not self.symm_mem_comm.disabled
                # ),
            )

        if self.world_size > 1:
            from aiter.dist.device_communicators.quick_all_reduce import (
                QuickAllReduce,
            )

            #     # Initialize a custom quick all-reduce implementation for hygon.
            #     # Quick reduce is designed as a complement to custom allreduce.
            #     # Based on quickreduce (https://github.com/mk1-project/quickreduce).
            #     # If it's a rocm, 'use_custom_allreduce==True' means it must
            #     # currently be an HYGON series.
            self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)

        if self.use_all2all:
            if self.all2all_backend == "naive":
                from .all2all import NaiveAll2AllManager

                self.all2all_manager = NaiveAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "allgather_reducescatter":
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "pplx":
                from .all2all import PPLXAll2AllManager

                self.all2all_manager = PPLXAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "deepep_high_throughput":
                from .all2all import DeepEPHTAll2AllManager

                self.all2all_manager = DeepEPHTAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "deepep_low_latency":
                from .all2all import DeepEPLLAll2AllManager

                self.all2all_manager = DeepEPLLAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "flashinfer_all2allv":
                from .all2all import FlashInferAllToAllManager

                self.all2all_manager = FlashInferAllToAllManager(self.cpu_group)
            else:
                raise ValueError(f"Unknown all2all backend: {self.all2all_backend}")

            if is_global_first_rank():
                logger.info(
                    "Using %s all2all manager.",
                    self.all2all_manager.__class__.__name__,
                )

    def all_reduce(
        self,
        input_,
        use_new: bool = True,
        ca_fp8_quant: bool = False,
        prefill_support: bool = False,
    ) -> torch.Tensor:
        # always try quick reduce first, then custom allreduce,
        # and then pynccl. (quick reduce just for ROCM MI3*)
        qr_comm = self.qr_comm
        if (
            qr_comm is not None
            and not qr_comm.disabled
            and qr_comm.should_quick_allreduce(input_)
        ):
            out = qr_comm.quick_all_reduce(input_)
            assert out is not None
            return out

        ca_comm = self.ca_comm
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input_)
        ):
            out = ca_comm.custom_all_reduce(input_, use_new=use_new, open_fp8_quant=ca_fp8_quant)
            assert out is not None
            return out
        symm_mem_comm = self.symm_mem_comm
        if symm_mem_comm is not None and symm_mem_comm.should_use_symm_mem(input_):
            out = symm_mem_comm.all_reduce(input_)
            assert out is not None
            return out
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            out = pynccl_comm.all_reduce(input_)
            assert out is not None
            return out
        # fall back to the default all-reduce using PyTorch.
        # this usually happens during testing.
        # when we run the model, allreduce only happens for the TP
        # group, where we always have either custom allreduce or pynccl.
        out = input_.clone()
        torch.distributed.all_reduce(out, group=self.device_group)
        return out

    def fused_allreduce_rmsnorm(
        self, input_, res_inp_, weight_, eps, prefill_support: bool = False,
        x_pad_to_multiple: int = 0, gemma_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = weight_.numel()
        if x_pad_to_multiple < 0:
            raise ValueError("x_pad_to_multiple must be nonnegative")
        out_hidden_dim = ((valid + x_pad_to_multiple - 1) // x_pad_to_multiple * x_pad_to_multiple) if x_pad_to_multiple else 0
        if valid <= 0 or input_.shape[-1] < valid or res_inp_.shape != input_.shape[:-1] + (valid,):
            raise ValueError("input, weight and residual widths do not match")
        if out_hidden_dim and out_hidden_dim < valid:
            raise ValueError("output width must contain the valid columns")
        total_bytes = input_.numel() * input_.element_size()
        ca = self.ca_comm
        limit = self._ar_1stage_max_kb if self._ar_1stage_max_kb >= 0 else 128 * 7168 * 2 // self.world_size
        one_stage = self._ar_1stage_override if self._ar_1stage_override is not None else total_bytes <= limit
        qr = getattr(self, "qr_comm", None)
        if (not one_stage and not x_pad_to_multiple and not gemma_norm
                and input_.shape[-1] == valid and qr is not None
                and qr.should_quick_allreduce_rmsnorm(input_, res_inp_, weight_, valid)):
            return qr.quick_all_reduce_rmsnorm(input_, res_inp_, weight_, eps, valid)
        if (ca is not None and not ca.disabled and self.world_size in (2, 4, 8)
                and valid <= 16384 and total_bytes < 64 * 1024 * 1024
                and ca.should_custom_ar(input_, prefill_support)):
            result = ca.custom_fused_ar_rms(input_, res_inp_, weight_, eps,
                use_1stage=one_stage, out_hidden_dim=out_hidden_dim, gemma_norm=gemma_norm)
            if result is not None:
                return result
        ar_out = self.all_reduce(input_.contiguous())[..., :valid]
        if not gemma_norm and input_.shape[-1] == valid and not out_hidden_dim:
            from aiter import rmsnorm2d_fwd_with_add
            out, residual_out = torch.empty_like(ar_out), torch.empty_like(res_inp_)
            rmsnorm2d_fwd_with_add(out, ar_out, res_inp_, residual_out, weight_, eps)
            return out, residual_out
        # Match the fused path's FP32 residual addition and RMS accumulation.
        z = ar_out.float() + res_inp_.float()
        norm = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
        norm = norm * (weight_.float() + (1 if gemma_norm else 0))
        out = norm.to(input_.dtype)
        out_dim = out_hidden_dim or input_.shape[-1]
        if out_dim > valid:
            out = torch.nn.functional.pad(out, (0, out_dim - valid))
        return out, z.to(input_.dtype)

    @staticmethod
    def _quantize_norm(out, group_size, transpose_scale=False):
        if group_size <= 0 or out.shape[-1] % group_size:
            raise ValueError("group_size must divide the hidden width")
        if transpose_scale and out.ndim != 2:
            raise ValueError("transpose_scale requires a two-dimensional input")
        from aiter.utility.dtypes import fp8 as platform_fp8
        if platform_fp8 != torch.uint8 and group_size in (128, out.shape[-1]):
            from aiter import get_hip_quant, QuantType
            kind = QuantType.per_Token if group_size == out.shape[-1] else QuantType.per_1x128
            quant, scale = get_hip_quant(kind)(out, quant_dtype=platform_fp8)
            if transpose_scale:
                transposed = torch.empty((scale.shape[1], scale.shape[0]), dtype=scale.dtype, device=scale.device).t()
                transposed.copy_(scale)
                scale = transposed
            return quant, scale
        values = out.float().reshape(*out.shape[:-1], out.shape[-1] // group_size, group_size)
        amax = values.abs().amax(-1)
        scale = torch.where(amax == 0, torch.ones_like(amax), amax / 448.0)
        quant = (values / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(out.shape)
        if transpose_scale:
            transposed = torch.empty((scale.shape[1], scale.shape[0]), dtype=scale.dtype, device=scale.device).t()
            transposed.copy_(scale)
            scale = transposed
        return quant, scale

    def fused_allreduce_rmsnorm_quant(
        self, input_, res_inp_, weight_, eps, prefill_support: bool = False,
        quant_type="per_token", group_size: int = 128, emit_bf16: bool = False,
        transpose_scale: bool = False, gemma_norm: bool = False,
    ):
        from ..communication_op import _canonical_fused_ar_rms_quant_type
        kind = _canonical_fused_ar_rms_quant_type(quant_type)
        if kind != "per_token" and gemma_norm:
            raise ValueError("Gemma quantization is supported only for per-token FP8")
        if kind == "per_group":
            return self.fused_allreduce_rmsnorm_quant_per_group(input_, res_inp_, weight_, eps,
                group_size, prefill_support, emit_bf16, transpose_scale)
        if transpose_scale:
            raise ValueError("transpose_scale is supported only for per-group FP8")
        if kind == "mxfp4":
            ca = self.ca_comm
            if ca is None or ca.disabled:
                raise RuntimeError("MXFP4 fusion requires an enabled custom communicator")
            result = ca.custom_fused_ar_rms_mxfp4_quant(input_, res_inp_, weight_, eps,
                emit_bf16=emit_bf16)
            if result is None:
                raise ValueError("MXFP4 fusion input is outside communicator capacity")
            return result
        total_bytes = input_.numel() * input_.element_size()
        K = input_.shape[-1]
        one_stage = total_bytes <= 128 * 1024
        ca = self.ca_comm
        result = None
        if (ca is not None and not ca.disabled and self.world_size in (2, 4, 8)
                and K in (512, 1024, 2048, 4096) and total_bytes <= 4096 * 1024
                and not (input_.dtype == torch.bfloat16 and not one_stage and K == 4096)):
            result = ca.custom_fused_ar_rms_quant(input_, res_inp_, weight_, eps,
                use_1stage=one_stage, gemma_norm=gemma_norm, emit_bf16=emit_bf16)
        if result is not None:
            return result
        norm, residual = self.fused_allreduce_rmsnorm(input_, res_inp_, weight_, eps,
            prefill_support, gemma_norm=gemma_norm)
        out, scale = self._quantize_norm(norm, K)
        return (out, residual, scale, norm) if emit_bf16 else (out, residual, scale)

    def fused_allreduce_rmsnorm_quant_per_group(
        self, input_, res_inp_, weight_, eps, group_size=128,
        prefill_support: bool = False, emit_bf16: bool = False,
        transpose_scale: bool = False,
    ):
        from .custom_all_reduce import _validate_per_group_size
        K = input_.shape[-1]
        _validate_per_group_size(group_size, input_.element_size(), K)
        if transpose_scale and input_.ndim != 2:
            raise ValueError("transpose_scale requires a two-dimensional input")
        total_bytes = input_.numel() * input_.element_size()
        one_stage = total_bytes <= 128 * 1024
        ca = self.ca_comm
        result = None
        if (ca is not None and not ca.disabled and self.world_size in (2, 4, 8)
                and K <= 16384 and total_bytes < 64 * 1024 * 1024
                and not (input_.dtype == torch.bfloat16 and not one_stage)):
            result = ca.custom_fused_ar_rms_per_group_quant(input_, res_inp_, weight_, eps,
                group_size, use_1stage=one_stage, emit_bf16=emit_bf16,
                transpose_scale=transpose_scale)
        if result is not None:
            return result
        norm, residual = self.fused_allreduce_rmsnorm(input_, res_inp_, weight_, eps, prefill_support)
        out, scale = self._quantize_norm(norm, group_size, transpose_scale)
        return (out, residual, scale, norm) if emit_bf16 else (out, residual, scale)

    def fused_qknorm_allreduce_rope(
        self,
        qkv_in,
        q_w,
        k_w,
        cos_sin_cache,
        position_ids,
        head_dim,
        rotary_dim,
        eps,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_out, k_out, v_out = self.ca_comm.custom_fused_qknorm_ar_rope(
            qkv_in,
            q_w,
            k_w,
            cos_sin_cache,
            position_ids,
            head_dim,
            rotary_dim,
            eps,
        )
        assert q_out is not None
        assert k_out is not None
        assert v_out is not None
        return q_out, k_out, v_out


    def fused_qknorm_allreduce(
        self,
        qkv_in,
        q_w,
        k_w,
        eps,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_out, k_out, v_out = self.ca_comm.custom_fused_qknorm_ar(
            qkv_in, q_w, k_w, eps
        )
        assert q_out is not None
        assert k_out is not None
        assert v_out is not None
        return q_out, k_out, v_out

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size
        if not -input_.ndim <= dim < input_.ndim:
            raise ValueError("invalid reduce_scatter dimension")
        dim %= input_.ndim
        if self.ca_comm is not None and self.ca_comm.should_custom_rs(input_, dim):
            shape = list(input_.shape)
            shape[dim] //= world_size
            output = torch.empty(shape, dtype=input_.dtype, device=input_.device)
            self.ca_comm.custom_reduce_scatter(input_, output, dim=dim)
            return output
        # RCCL FP8 SUM does not implement our ordered FP32 accumulation and
        # single non-saturating rounding contract. FNUZ is unsupported as well.
        if input_.dtype in FP8_DTYPES or input_.dtype in (
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2fnuz", None),
        ):
            raise ValueError(
                "FP8 reduce_scatter requires a supported custom input and an active "
                "custom communicator; RCCL fallback and FNUZ are unsupported"
            )
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            raise RuntimeError("reduce_scatter requires an active RCCL fallback communicator")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(dim, 0).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(dim, 0).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if sizes is not None:
            pynccl_comm.reduce_scatterv(output, input_tensor, sizes=sizes)
        else:
            pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
        if self.qr_comm is not None:
            self.qr_comm.close()
            self.qr_comm = None
        if self.ca_comm is not None:
            self.ca_comm = None
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None and not pynccl_comm.disabled

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert (
                    input_.shape[dim] == sizes[self.rank_in_group]
                ), f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )
            if sizes is not None:
                pynccl_comm.all_gatherv(output_tensor, input_, sizes=sizes)
            else:
                pynccl_comm.all_gather(output_tensor, input_)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        pynccl_comm.group_start()
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        pynccl_comm.group_end()

        return output_list

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.all2all_manager is not None
        hidden_states, router_logits = self.all2all_manager.dispatch(
            hidden_states, router_logits, is_sequence_parallel
        )
        return hidden_states, router_logits

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        assert self.all2all_manager is not None
        hidden_states = self.all2all_manager.combine(
            hidden_states, is_sequence_parallel
        )
        return hidden_states
