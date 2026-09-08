# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import torch
from torch.distributed import ProcessGroup

should_nccl_symm_mem_allreduce = False
from aiter.dist.parallel_state import is_global_first_rank
from aiter import logger
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
        self, input_, res_inp_, weight_, eps, prefill_support: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = input_.shape[-1]
        total_bytes = input_.numel() * input_.element_size()
        can_use_fuse_ar_rms = (
            n <= 16384
            and total_bytes < 8 * 1024 * 8192
            and self.world_size != 6
        )
        ca_comm = self.ca_comm
        can_use_custom_ar = (
            ca_comm is not None and not ca_comm.disabled and can_use_fuse_ar_rms
        )
        total_bytes_limit = (
            self._ar_1stage_max_kb
            if self._ar_1stage_max_kb >= 0
            else 128 * 7168 * 2 // self.world_size
        )
        use_1stage = (
            self._ar_1stage_override
            if self._ar_1stage_override is not None
            else (total_bytes <= total_bytes_limit)
        )
        if (
            can_use_custom_ar
            and ca_comm.should_custom_ar(input_, prefill_support)
        ):
            out, res_out = ca_comm.custom_fused_ar_rms(
                input_, res_inp_, weight_, eps, use_1stage=use_1stage
            )
            assert out is not None
            assert res_out is not None
            return out, res_out
        # call split kernel
        ar_out = self.all_reduce(input_)
        out = torch.empty_like(ar_out)
        residual_out = torch.empty_like(ar_out)
        from aiter import rmsnorm2d_fwd_with_add

        rmsnorm2d_fwd_with_add(
            out,
            ar_out,
            input_,
            residual_out,
            weight_,
            eps,
        )
        return out, residual_out

    def fused_allreduce_rmsnorm_quant(
        self,
        input_,
        res_inp_,
        weight_,
        eps,
        prefill_support: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_bytes = input_.numel() * input_.element_size()
        K = int(input_.shape[-1])
        use_1stage = total_bytes <= 128 * 1024
        # Hygon (gfx938/gfx946) kernel-level bug: the fused
        # AR+RMSNorm+FP8-quant kernel produces 100% NaN for bf16 at
        # K=4096 whenever use_1stage=False, regardless of which post-
        # 1-stage path is dispatched (the 2-stage 128KB-512KB path AND
        # the split >512KB path both fail). Empirically confirmed on
        # shapes (17,4096), (32,4096), (128,4096), and (and very likely
        # (512,4096)+ which we don't proceed to). The 2-stage kernel
        # works for fp16 at the same K=4096, and for bf16 at K=7168 /
        # K=8192; the failure is therefore K=4096-and-bf16-specific.
        # Until the C++ kernel is fixed upstream, fall back to the
        # Python split path (RMSNorm-only fused kernel + separate
        # hip_quant) for the entire problematic configuration.
        problematic_bf16_non_1stage = (
            input_.dtype == torch.bfloat16
            and not use_1stage
            and K == 4096
        )
        if (
            K in [512, 1024, 2048, 4096]
            and total_bytes <= 4096 * 1024
            and not problematic_bf16_non_1stage
        ):
            out, res_out, scale_out = self.ca_comm.custom_fused_ar_rms_quant(
                input_, res_inp_, weight_, eps, use_1stage
            )
        else:
            out_, res_out = self.fused_allreduce_rmsnorm(
                input_, res_inp_, weight_, eps, prefill_support
            )
            from aiter import get_hip_quant, QuantType
            from aiter.utility.dtypes import fp8
            hip_quant = get_hip_quant(QuantType.per_Token)
            out, scale_out = hip_quant(out_, quant_dtype=fp8)
        assert out is not None
        assert res_out is not None
        assert scale_out is not None
        return out, res_out, scale_out

    def fused_allreduce_rmsnorm_quant_per_group(
        self,
        input_,
        res_inp_,
        weight_,
        eps,
        group_size=128,
        prefill_support: bool = False,
        emit_bf16: bool = False,
    ):
        total_bytes = input_.numel() * input_.element_size()
        K = int(input_.shape[-1])
        use_1stage = total_bytes <= 128 * 1024
        out = res_out = scale_out = bf16_out = None
        fused_ok = False
        # See ``fused_allreduce_rmsnorm_quant`` for context, with one
        # important difference: per-token quant's custom-kernel
        # whitelist is K in {512, 1024, 2048, 4096}, so larger K values
        # (6144 / 7168 / 8192) always go to the Python fallback there
        # and never expose the kernel bug for those K. Per-group quant
        # has a much wider whitelist (any K with K % group_size == 0
        # and K <= 16384), so it surfaces the same bug at additional K
        # values (K=4096 and K=6144 both empirically confirmed NaN;
        # K=7168 / K=8192 untested but likely affected). Widen the
        # fallback to all bf16 + non-1-stage configurations to be safe;
        # the perf cost is limited since this only affects bf16 inputs
        # whose total bytes exceed 128 KB (medium / large prefill).
        problematic_bf16_non_1stage = (
            input_.dtype == torch.bfloat16
            and not use_1stage
        )
        if (
            K % group_size == 0
            and K <= 16384
            and total_bytes < 8 * 1024 * 8192
            and not problematic_bf16_non_1stage
        ):
            try:
                result = self.ca_comm.custom_fused_ar_rms_per_group_quant(
                    input_, res_inp_, weight_, eps, group_size, use_1stage,
                    emit_bf16=emit_bf16,
                )
                if emit_bf16:
                    out, res_out, scale_out, bf16_out = result
                else:
                    out, res_out, scale_out = result
                fused_ok = True
            except Exception:
                pass
        if not fused_ok:
            out_, res_out = self.fused_allreduce_rmsnorm(
                input_, res_inp_, weight_, eps, prefill_support
            )
            from aiter import get_hip_quant, QuantType
            from aiter.utility.dtypes import fp8
            hip_quant = get_hip_quant(QuantType.per_1x128)
            out, scale_out = hip_quant(out_, quant_dtype=fp8)
            if emit_bf16:
                bf16_out = out_
        assert out is not None
        assert res_out is not None
        assert scale_out is not None
        if emit_bf16:
            assert bf16_out is not None
            return out, res_out, scale_out, bf16_out
        return out, res_out, scale_out

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
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

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
        input_tensor = input_.movedim(0, dim).contiguous()

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
