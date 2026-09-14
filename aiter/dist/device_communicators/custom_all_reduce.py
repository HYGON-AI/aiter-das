"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2026, The vLLM team.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import os
import pickle
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# import vllm.envs as envs
# from vllm import _custom_ops as ops
import aiter as ops
from aiter.dist.parallel_state import in_the_same_node_as
from aiter import logger
from aiter.utility.dtypes import fp8

try:
    ops.meta_size()
    custom_ar = True
except Exception as e:
    # For CPUs
    custom_ar = False
    logger.warning(f"Custom allreduce is disabled: {e}")


_AR_TRANSPORT_IDS = {"ipc": 0, "fabric": 1}
_DEFAULT_AR_MAX_SIZE_MB = 256
_LEGACY_AR_MAX_SIZE = 64 * 1024 * 1024


def _requested_ar_max_size() -> int:
    """Ordinary AllReduce admission limit in bytes, independent of allocation."""
    value = os.environ.get("AITER_AR_MAX_SIZE_MB", str(_DEFAULT_AR_MAX_SIZE_MB))
    try:
        size_mb = int(value)
    except ValueError:
        raise ValueError("AITER_AR_MAX_SIZE_MB must be a positive integer") from None
    if size_mb <= 0:
        raise ValueError("AITER_AR_MAX_SIZE_MB must be a positive integer")
    return size_mb * 1024 * 1024


def _requested_ar_transport() -> str:
    transport = os.environ.get("AITER_AR_TRANSPORT", "ipc").strip().lower()
    if transport not in ("ipc", "fabric", "auto"):
        raise ValueError(
            "AITER_AR_TRANSPORT must be one of ipc|fabric|auto, "
            f"got {transport!r}"
        )
    return transport


def _fabric_ar_available() -> bool:
    try:
        return bool(ops.fabric_ar_available())
    except (AttributeError, TypeError, RuntimeError):
        return False


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


# Wavefront width on AMD CDNA / gfx94x / gfx950. ``__shfl_xor`` in the
# fused per-group FP8 quant epilogue is scoped to a single wavefront, so
# ``threads_per_group = group_size / PACK_SIZE`` must fit inside it.
_AITER_AR_WAVEFRONT_SIZE = 64


def _validate_per_group_size(group_size: int, element_size: int, n: int) -> None:
    """Validate ``group_size`` for the fused AR + RMSNorm + per-group FP8
    quant kernel. Mirrors the C++ host dispatcher checks in
    ``dispatchFusedAllReduceRMSNormQuantPerGroup`` so callers fail fast
    with a clear Python-level ``ValueError`` (rather than a generic
    ``RuntimeError`` from the extension, which aborts CUDA-graph capture
    asynchronously).

    The fused epilogue imposes five constraints on ``group_size``:

    (a) ``group_size > 0``
    (b) ``group_size % PACK_SIZE == 0`` with ``PACK_SIZE = 16 // element_size``
        (each thread owns a full 16-byte pack, so a group must be made of
        whole packs).
    (c) ``threads_per_group = group_size / PACK_SIZE`` must be a power of two
        (butterfly ``__shfl_xor`` reduction strides ``{tpg/2, tpg/4, ..., 1}``).
    (d) ``threads_per_group`` must fit inside a wavefront
        (``<= 64`` on AMD CDNA); cross-warp shuffles do not exist on HIP.
    (e) ``n % group_size == 0`` so ``num_groups = n / group_size`` is an
        integer.
    """
    if not isinstance(group_size, int):
        raise TypeError(
            f"per-group quant group_size must be int, got {type(group_size).__name__}"
        )
    if group_size <= 0:
        raise ValueError(
            f"per-group quant requires group_size > 0, got group_size={group_size}"
        )
    if element_size <= 0 or 16 % element_size != 0:
        raise ValueError(
            "per-group quant requires an element_size that divides 16 "
            f"(bf16/fp16: 2), got element_size={element_size}"
        )
    pack_size = 16 // element_size
    if group_size % pack_size != 0:
        raise ValueError(
            f"per-group quant requires group_size divisible by PACK_SIZE="
            f"{pack_size} (16 // element_size), got group_size={group_size}"
        )
    threads_per_group = group_size // pack_size
    if threads_per_group & (threads_per_group - 1) != 0:
        raise ValueError(
            "per-group quant requires group_size/PACK_SIZE to be a power of "
            "two (butterfly __shfl_xor reduction), got "
            f"group_size={group_size} PACK_SIZE={pack_size} "
            f"threads_per_group={threads_per_group}"
        )
    if threads_per_group > _AITER_AR_WAVEFRONT_SIZE:
        raise ValueError(
            "per-group quant requires group_size/PACK_SIZE <= wavefront size "
            f"({_AITER_AR_WAVEFRONT_SIZE}), got group_size={group_size} "
            f"PACK_SIZE={pack_size} threads_per_group={threads_per_group}"
        )
    if n % group_size != 0:
        raise ValueError(
            f"per-group quant requires n divisible by group_size, "
            f"got n={n} group_size={group_size}"
        )


class IPCBuffer:
    """A single IPC-accessible device buffer.

    Pure data container — owns a pre-allocated GPU allocation with a fixed
    device address.  All IPC handle / broadcast / registration logic lives
    in IPCBufferPool.

    When *uncached* is False (default), memory is allocated through PyTorch's
    caching allocator (torch.empty).  When True, memory is allocated via
    hipExtMallocWithFlags with hipDeviceMallocUncached, bypassing the cache.
    Uncached buffers are suitable for cross-GPU synchronization metadata and
    signal buffers where cache coherence overhead is undesirable.
    """

    def __init__(
        self,
        size: int,
        device: torch.device,
        uncached: bool = False,
        transport: int = 0,
    ):
        self._size = size
        self._uncached = uncached
        self._transport = transport
        self._raw_ptr = 0
        if uncached:
            self._buffer = None
            self._raw_ptr = ops.allocate_meta_buffer(size, transport)
        else:
            self._buffer = torch.empty(size, dtype=torch.uint8, device=device)
            self._raw_ptr = self._buffer.data_ptr()

    @property
    def data_ptr(self) -> int:
        return self._raw_ptr

    @property
    def tensor(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError(
                "Uncached IPCBuffer has no backing tensor; use .data_ptr"
            )
        return self._buffer

    @property
    def max_size(self) -> int:
        return self._size

    @property
    def uncached(self) -> bool:
        return self._uncached

    def __del__(self):
        if self._uncached and self._raw_ptr:
            try:
                ops.free_meta_buffer(self._raw_ptr)
            except (AttributeError, TypeError):
                pass
            self._raw_ptr = 0


class IPCBufferPool:
    """Manages a collection of named IPCBuffers and provides IPC broadcast
    infrastructure for cross-GPU communication.

    Buffers are stored in an internal dict and accessed by string key.

    Two sets of operations:

    Eager mode (named internal buffers):
        create(key, size) allocates a buffer and stores it under *key*.
        get_ipc_meta(key) broadcasts IPC handles for that buffer.

    Graph mode (arbitrary external tensors):
        get_external_ipc_meta(tensor) broadcasts IPC handles for any tensor.
        flush_graph_buffers(ar_ptr) batch-registers addresses that the C++
        backend collected during CUDA graph capture.
    """

    _pool_seq: int = 0

    def __init__(self, device: torch.device, group: ProcessGroup, transport: int = 0):
        self._device = device
        self._group = group
        self._transport = transport
        self._handle_size = int(ops.ar_handle_size(transport))
        self._rank = dist.get_rank(group=group)
        self._world_size = dist.get_world_size(group=group)
        self._buffers: Dict[str, IPCBuffer] = {}

        self._store = dist.distributed_c10d._get_default_store()
        self._assert_pure_tcp_store(self._store)

        ranks_tag = "_".join(map(str, sorted(dist.get_process_group_ranks(group))))
        self._store_key_prefix = f"aiter_ipc/p{IPCBufferPool._pool_seq}/g{ranks_tag}"
        IPCBufferPool._pool_seq += 1
        self._ipc_seq = 0

    @staticmethod
    def _assert_pure_tcp_store(store) -> None:
        """Verify the store is a pure-TCP KV store, free from any collective
        communication backend (RCCL / gloo / MPI).
        Emits a warning rather than aborting to allow non-TCP store setups."""
        s = store
        while isinstance(s, dist.PrefixStore):
            s = s.underlying_store
        if not isinstance(s, dist.TCPStore):
            logger.warning(
                "IPC metadata exchange prefers a pure-TCP KV store "
                "(torch.distributed.TCPStore), got %s. "
                "If IPC handle exchange fails, ensure MASTER_ADDR/MASTER_PORT "
                "are set and the process group is initialised with a TCPStore.",
                type(s).__name__,
            )

    # ---- Buffer lifecycle ----

    def create(self, key: str, size: int, uncached: bool = False) -> IPCBuffer:
        """Allocate a new IPCBuffer and store it under *key*.

        Args:
            key: unique name for this buffer in the pool.
            size: buffer size in bytes.
            uncached: if True, allocate via hipMalloc (uncached);
                      if False (default), allocate via torch.empty (cached).
        """
        if key in self._buffers:
            raise KeyError(f"IPCBuffer '{key}' already exists in the pool")
        buf = IPCBuffer(
            size,
            self._device,
            uncached=uncached,
            transport=self._transport,
        )
        self._buffers[key] = buf
        return buf

    def __getitem__(self, key: str) -> IPCBuffer:
        return self._buffers[key]

    def __contains__(self, key: str) -> bool:
        return key in self._buffers

    # ---- Eager mode: named buffer IPC meta ----

    def get_ipc_meta(self, key: str) -> Tuple[List, List]:
        """Broadcast IPC handles for the named buffer across all ranks."""
        return self.gather_meta(self.get_local_meta(key))

    def get_local_meta(self, key: str):
        """Export one named buffer without entering a blocking store read."""
        buf = self._buffers[key]
        return self._export_meta(buf.data_ptr, buf.max_size, 0)

    def gather_meta(self, local_meta) -> Tuple[List, List]:
        """Exchange already-exported metadata across all ranks."""
        return self._gather_ipc_meta(local_meta)

    # ---- Graph mode: external buffer IPC meta ----

    def get_external_ipc_meta(self, tensor: torch.Tensor) -> Tuple[List, List]:
        """Broadcast IPC handles for an arbitrary external tensor."""
        if self._transport == _AR_TRANSPORT_IDS["fabric"]:
            storage = tensor.untyped_storage()
            base_ptr = storage.data_ptr()
            offset = tensor.data_ptr() - base_ptr
            return self._gather_ipc_meta(
                self._export_meta(base_ptr, storage.nbytes(), offset)
            )
        return self._gather_ipc_meta(
            self._export_meta(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
        )

    def flush_graph_buffers(self, ar_ptr):
        """Batch-register buffer addresses collected during CUDA graph capture.

        During graph capture the C++ backend records addresses of buffers that
        are not yet IPC-registered.  After capture ends this method exchanges
        their IPC handles across all ranks and completes registration.
        """
        count = ops.get_graph_buffer_count(ar_ptr)
        if count == 0:
            return
        if self._transport != _AR_TRANSPORT_IDS["ipc"]:
            raise RuntimeError(
                "direct graph-buffer registration is not supported for fabric; "
                "set AITER_AR_ENABLE_REG_CAPTURE=0"
            )
        handle_sz = self._handle_size
        handle = torch.empty(count * handle_sz, dtype=torch.uint8)
        offset = torch.empty(count, dtype=torch.int64)
        ops.get_graph_buffer_ipc_meta(ar_ptr, handle.data_ptr(), offset.data_ptr())
        handles, offsets = self._gather_ipc_meta((handle, offset))
        logger.info("Registering %d cuda graph addresses", count)
        ops.register_graph_buffers(
            ar_ptr,
            [h.data_ptr() for h in handles],
            [o.data_ptr() for o in offsets],
        )

    # ---- Private IPC primitives ----

    def _export_meta(self, data_ptr: int, size: int, offset: int):
        """Export one allocation using the selected transport."""
        handle = torch.empty(self._handle_size, dtype=torch.uint8)
        ops.get_meta_buffer_handle(
            data_ptr,
            size,
            handle.data_ptr(),
            self._transport,
        )
        return handle, offset

    def _gather_ipc_meta(self, shard_data) -> Tuple[List, List]:
        """Exchange IPC metadata (handle + offset) across all ranks via TCP store.

        Each rank writes its serialised *shard_data* under a unique key, then
        reads every other rank's data.  ``store.get()`` blocks until the key
        is available, providing natural barrier semantics without involving any
        collective communication backend.
        """
        seq = self._ipc_seq
        self._ipc_seq += 1
        prefix = f"{self._store_key_prefix}/{seq}"

        self._store.set(f"{prefix}/r{self._rank}", pickle.dumps(shard_data))

        handles = []
        offsets = []
        for r in range(self._world_size):
            raw = self._store.get(f"{prefix}/r{r}")
            h, o = pickle.loads(raw)
            handles.append(h)
            offsets.append(o)
        return handles, offsets


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8, 16, 32, 40]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=1024 * 1024 * 1024,  # Input staging capacity in bytes (1 GiB).
        enable_register_for_capturing: bool = True,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        self._ptr = 0
        self.requested_transport = _requested_ar_transport()
        self._all_reduce_max_size = _requested_ar_max_size()
        self.transport = "ipc"
        self.transport_id = _AR_TRANSPORT_IDS["ipc"]

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return

        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."

        same_node = None
        if self.requested_transport in ("ipc", "auto"):
            same_node = all(in_the_same_node_as(group, source_rank=0))

        if self.requested_transport == "ipc":
            if not same_node:
                # Preserve the legacy same-node-only IPC behavior.
                logger.warning(
                    "Custom allreduce IPC is disabled because this process group "
                    "spans across nodes. Set AITER_AR_TRANSPORT=fabric or auto "
                    "with a supernode-enabled build to use fabric memory."
                )
                return
            self.transport = "ipc"
        elif self.requested_transport == "auto":
            self.transport = "ipc" if same_node else "fabric"
            if self.transport == "fabric" and not _fabric_ar_available():
                logger.warning(
                    "Custom allreduce auto transport selected fabric for a multi-node "
                    "group, but fabric support is not compiled; falling back to the "
                    "next communicator. Rebuild with AITER_ENABLE_SUPERNODE_AR=1."
                )
                return
        else:
            self.transport = "fabric"
            if not _fabric_ar_available():
                raise RuntimeError(
                    "AITER_AR_TRANSPORT=fabric was requested, but fabric support is "
                    "not compiled. Set AITER_ENABLE_SUPERNODE_AR=1 before building "
                    "module_custom_all_reduce."
                )
        self.transport_id = _AR_TRANSPORT_IDS[self.transport]

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        # device_ids = get_cuda_visible_devices()

        # physical_device_id = device_ids[device.index]
        # tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        # gather_list = [
        #     torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        # ]
        # dist.all_gather(gather_list, tensor, group=self.group)
        # physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        # assert current_platform.is_cuda() or current_platform.is_rocm()
        # fully_connected = current_platform.is_full_nvlink(physical_device_ids)
        fully_connected = True
        if world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On HCU, p2p is always enabled between HSL connected GPUs
        # if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
        #     logger.warning(
        #         "Custom allreduce is disabled because your platform lacks "
        #         "GPU P2P capability or P2P test failed. To silence this "
        #         "warning, specify disable_custom_all_reduce=True explicitly.")
        #     return

        self.disabled = False
        if self.transport == "fabric" and enable_register_for_capturing:
            logger.warning(
                "Fabric custom allreduce does not export arbitrary graph allocations; "
                "forcing AITER_AR_ENABLE_REG_CAPTURE=0 copy-in mode."
            )
            enable_register_for_capturing = False
        # Every graph collective must use copy-in mode when registration is
        # disabled; Fabric cannot register arbitrary graph allocations later.
        self.enable_register_for_capturing = enable_register_for_capturing
        debug_init = os.environ.get("AITER_AR_DEBUG_STAGES", "0") == "1"

        def debug_stage(stage: str) -> None:
            if debug_init:
                print(f"CUSTOM_AR_STAGE rank={rank} stage={stage}", flush=True)

        # This stores tuples of pointers to buffers from all ranks. RankData
        # has kMaxRanks=40 pointer slots, so each tuple is always 320 bytes,
        # including when the active world size is smaller. The fixed 8 MiB pool
        # therefore holds 26214 tuples; the largest model observed needs fewer
        # than 10000, so the allocation size itself does not increase.
        debug_stage("rank_data_begin")
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        debug_stage("rank_data_done")
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size

        # Use a gloo-based barrier to synchronise init status across all ranks.
        # If any rank fails (e.g. allocate_meta_buffer throws on platforms that
        # do not support hipDeviceMallocUncached), the barrier ensures all other
        # ranks learn about the failure instead of hanging forever in
        # _gather_ipc_meta's store.get().
        init_error = None
        debug_stage("buffer_allocation_begin")
        try:
            # Create IPC buffer pool and allocate all named buffers.
            # "meta" uses hipAlloc (uncached) for synchronization metadata +
            # intermediate allreduce temp storage.
            # "input" is also uncached/exportable for D2D relay in eager mode.
            self._pool = IPCBufferPool(
                self.device,
                self.group,
                transport=self.transport_id,
            )
            self._pool.create("meta", ops.meta_size() + max_size * 2, uncached=True)
            self._pool.create("input", max_size, uncached=True)
        except Exception as e:
            init_error = e
        if not self._sync_init_stage("buffer allocation", init_error):
            return
        debug_stage("buffer_allocation_done")

        # Export first, then agree across ranks before entering blocking store
        # reads. This prevents one failed fabric create from stranding peers.
        local_meta = None
        init_error = None
        try:
            local_meta = self._pool.get_local_meta("meta")
        except Exception as e:
            init_error = e
        if not self._sync_init_stage("meta handle export", init_error):
            return
        debug_stage("meta_handle_export_done")
        handles, offsets = self._pool.gather_meta(local_meta)
        debug_stage("meta_handle_gather_done")

        self.fully_connected = fully_connected
        init_error = None
        try:
            self._ptr = ops.init_custom_ar(
                self._pool["meta"].data_ptr,
                self.rank_data.data_ptr(),
                self.rank_data.numel(),
                [h.data_ptr() for h in handles],
                offsets,
                rank,
                self.fully_connected,
                self.transport_id,
            )
        except Exception as e:
            init_error = e
        if not self._sync_init_stage("peer meta attach", init_error):
            return
        debug_stage("peer_meta_attach_done")

        local_meta = None
        init_error = None
        try:
            local_meta = self._pool.get_local_meta("input")
        except Exception as e:
            init_error = e
        if not self._sync_init_stage("input handle export", init_error):
            return
        debug_stage("input_handle_export_done")
        handles, offsets = self._pool.gather_meta(local_meta)
        debug_stage("input_handle_gather_done")
        init_error = None
        try:
            ops.register_input_buffer(
                self._ptr,
                self._pool["input"].data_ptr,
                [h.data_ptr() for h in handles],
                offsets,
            )
        except Exception as e:
            init_error = e
        if not self._sync_init_stage("peer input attach", init_error):
            return
        debug_stage("peer_input_attach_done")

        logger.info(
            "Custom allreduce initialized with transport=%s requested=%s rank=%d/%d",
            self.transport,
            self.requested_transport,
            self.rank,
            self.world_size,
        )

    def _sync_init_stage(self, stage: str, error: Optional[Exception]) -> bool:
        """Make transport initialization failure collective and hang-safe."""
        local_error = None
        if error is not None:
            local_error = f"rank {self.rank}: {type(error).__name__}: {error}"
        errors = [None for _ in range(self.world_size)]
        dist.all_gather_object(errors, local_error, group=self.group)
        failures = [item for item in errors if item]
        if not failures:
            return True

        if self._ptr:
            try:
                ops.dispose(self._ptr)
            except (AttributeError, TypeError, RuntimeError):
                pass
            self._ptr = 0
        self.disabled = True
        message = (
            f"Custom allreduce {self.transport} {stage} failed collectively: "
            + "; ".join(failures)
        )
        if self.requested_transport == "fabric":
            raise RuntimeError(message)
        logger.warning("%s. Falling back to the next communicator.", message)
        return False

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        flush_graph_buffers call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self._pool.flush_graph_buffers(self._ptr)

    def register_input_buffer(self, inp: torch.Tensor):
        """Register an external tensor as an IPC input buffer."""
        handles, offsets = self._pool.get_external_ipc_meta(inp)
        ops.register_input_buffer(
            self._ptr, inp.data_ptr(), [h.data_ptr() for h in handles], offsets
        )

    def register_output_buffer(self, out: torch.Tensor):
        """Register an external tensor as an IPC output buffer."""
        handles, offsets = self._pool.get_external_ipc_meta(out)
        ops.register_output_buffer(
            self._ptr, out.data_ptr(), [h.data_ptr() for h in handles], offsets
        )

    def register_graph_buffers(self):
        """Batch-register graph-captured buffer addresses."""
        self._pool.flush_graph_buffers(self._ptr)

    def should_custom_ar(self, inp: torch.Tensor, prefill_support: bool = False):
        """Admit ordinary AR without requiring frameworks to identify prefill.

        Both framework dispatch and custom_all_reduce use this same policy.
        The configured limit does not resize the preallocated buffers. Keep
        the existing half-capacity bound for two-stage temporary storage.
        Explicit prefill callers retain their existing capacity-based limit.
        """
        if self.disabled:
            return False
        max_bytes = self.max_size // 2
        if not prefill_support:
            max_bytes = min(max_bytes, self._all_reduce_max_size)
        return self._should_custom_ar(inp, max_bytes)

    def _should_custom_ar(self, inp: torch.Tensor, max_bytes: int):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.fully_connected:
            return inp_size <= max_bytes
        return False

    def should_custom_ag(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # all_gather output = input * world_size, so the per-rank input
        # must fit within max_size / world_size
        if self.world_size == 2 or self.fully_connected:
            return inp_size <= (self.max_size / (self.world_size * 2))
        return False

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        use_new: bool = True,
        open_fp8_quant: bool = False,
        registered_input: bool = False,
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
        reg_inp = 0 if registered_input else self._pool["input"].data_ptr
        reg_inp_bytes = 0 if registered_input else self._pool["input"].max_size
        ops.all_reduce(
            self._ptr,
            inp,
            out,
            use_new,
            open_fp8_quant,
            reg_inp,
            reg_inp_bytes,
        )
        return out

    def custom_all_reduce(
        self, input: torch.Tensor, use_new: bool = True, open_fp8_quant: bool = False
    ) -> Optional[torch.Tensor]:
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(
                    input,
                    use_new=use_new,
                    open_fp8_quant=open_fp8_quant,
                    registered_input=self.enable_register_for_capturing,
                )
            else:
                # if warm up, mimic the allocation pattern
                # since custom allreduce is out-of-place
                return torch.zeros_like(input)
        else:
            # note: outside of cuda graph context,
            # custom allreduce incurs a cost of cudaMemcpy, which should
            # be small(<=1% of overall latency) compared to the performance
            # gains of using custom kernels
            return self.all_reduce(
                input,
                use_new=use_new,
                open_fp8_quant=open_fp8_quant,
                registered_input=False,
            )

    def reduce_scatter(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        *,
        registered: bool = False,
    ):
        assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
        reg = 0 if registered else self._pool["input"].data_ptr
        reg_bytes = 0 if registered else self._pool["input"].max_size
        ops.reduce_scatter(
            self._ptr,
            inp,
            out,
            reg,
            reg_bytes,
        )

    def custom_reduce_scatter(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> Optional[torch.Tensor]:
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self._should_custom_ar(input, _LEGACY_AR_MAX_SIZE):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.reduce_scatter(
                    input, output, registered=self.enable_register_for_capturing
                )
        else:
            return self.reduce_scatter(input, output, registered=False)

    def _allgather_out_shape(self, inp: torch.Tensor, dim: int):
        ndim = inp.dim()
        if dim == 0:
            return (inp.shape[0] * self.world_size,) + inp.shape[1:]
        if dim == -1 or dim == ndim - 1:
            return inp.shape[:-1] + (inp.shape[-1] * self.world_size,)
        print(
            f"[aiter] allgather does not support dim={dim}, falling back to 1-D output"
        )
        return (inp.numel() * self.world_size,)

    def all_gather_reg(self, inp: torch.Tensor, out: torch.Tensor = None, dim: int = 0):
        if out is None:
            out = torch.empty(
                self._allgather_out_shape(inp, dim),
                dtype=inp.dtype,
                device=inp.device,
            )
        assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
        ops.all_gather_reg(
            self._ptr,
            inp,
            out,
            dim,
        )
        return out

    def all_gather_unreg(
        self, inp: torch.Tensor, out: torch.Tensor = None, dim: int = 0
    ):
        if out is None:
            out = torch.empty(
                self._allgather_out_shape(inp, dim),
                dtype=inp.dtype,
                device=inp.device,
            )
        assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
        ops.all_gather_unreg(
            self._ptr,
            inp,
            self._pool["input"].data_ptr,
            out,
            self._pool["input"].max_size,
            dim,
        )
        return out

    def custom_all_gather(
        self, inp: torch.Tensor, dim: int = 0
    ) -> Optional[torch.Tensor]:
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                if self.enable_register_for_capturing:
                    return self.all_gather_reg(inp, dim=dim)
                return self.all_gather_unreg(inp, dim=dim)
            else:
                return torch.zeros(
                    self._allgather_out_shape(inp, dim),
                    dtype=inp.dtype,
                    device=inp.device,
                )
        else:
            return self.all_gather_unreg(inp, dim=dim)

    def fused_ar_rms(
        self,
        inp: torch.Tensor,
        res_inp: torch.Tensor,
        *,
        res_out: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        scale_out: Optional[torch.Tensor] = None,
        w: torch.Tensor,
        eps: float,
        registered: bool = False,
        use_1stage: bool = False,
        post_per_token_quant: bool = False,
    ):
        if res_out is None:
            res_out = torch.empty_like(inp)
        reg = 0 if registered else self._pool["input"].data_ptr
        reg_bytes = 0 if registered else self._pool["input"].max_size
        if not post_per_token_quant:
            if out is None:
                out = torch.empty_like(inp)
            assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
            ops.fused_allreduce_rmsnorm(
                self._ptr,
                inp,
                res_inp,
                res_out,
                out,
                w,
                eps,
                reg,
                reg_bytes,
                use_1stage,
            )
            return out, res_out
        else:
            if out is None:
                out = torch.empty(inp.shape, dtype=fp8, device=inp.device)
            assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
            if scale_out is None:
                scale_out = torch.empty(
                    inp.shape[:-1] + (1,), dtype=torch.float32, device=inp.device
                )
            ops.fused_allreduce_rmsnorm_quant(
                self._ptr,
                inp,
                res_inp,
                res_out,
                out,
                scale_out,
                w,
                eps,
                reg,
                reg_bytes,
                use_1stage,
            )
            return out, res_out, scale_out

    def custom_fused_ar_rms(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool = False,
    ) -> Optional[torch.Tensor]:
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self._should_custom_ar(input, _LEGACY_AR_MAX_SIZE):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.fused_ar_rms(
                    input,
                    residual_inp,
                    w=weight,
                    eps=eps,
                    registered=self.enable_register_for_capturing,
                    use_1stage=use_1stage,
                )
            else:
                return torch.zeros_like(input), torch.zeros_like(input)
        else:
            return self.fused_ar_rms(
                input,
                residual_inp,
                w=weight,
                eps=eps,
                registered=False,
                use_1stage=use_1stage,
            )

    def custom_fused_ar_rms_quant(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool = False,
    ):
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self._should_custom_ar(input, _LEGACY_AR_MAX_SIZE):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.fused_ar_rms(
                    input,
                    residual_inp,
                    w=weight,
                    eps=eps,
                    registered=self.enable_register_for_capturing,
                    use_1stage=use_1stage,
                    post_per_token_quant=True,
                )
            else:
                dummy_out = torch.zeros(input.shape, dtype=fp8, device=input.device)
                dummy_scale_out = torch.zeros(
                    input.shape[:-1] + (1,), dtype=torch.float32, device=input.device
                )
                return dummy_out, torch.zeros_like(input), dummy_scale_out
        else:
            return self.fused_ar_rms(
                input,
                residual_inp,
                w=weight,
                eps=eps,
                registered=False,
                use_1stage=use_1stage,
                post_per_token_quant=True,
            )

    def fused_ar_rms_per_group_quant(
        self,
        inp: torch.Tensor,
        res_inp: torch.Tensor,
        *,
        w: torch.Tensor,
        eps: float,
        group_size: int = 128,
        registered: bool = False,
        use_1stage: bool = False,
        emit_bf16: bool = False,
    ):
        K = inp.shape[-1]
        # Fail fast on bad ``group_size`` at the Python boundary. Mirrors
        # the C++ host dispatcher checks; catching it here surfaces a
        # synchronous ``ValueError`` instead of a post-launch
        # ``RuntimeError`` that would only fire at CUDA-graph replay and
        # would be much harder to attribute to the offending call site.
        _validate_per_group_size(group_size, inp.element_size(), K)
        res_out = torch.empty_like(inp)
        num_groups = K // group_size
        out = torch.empty(inp.shape, dtype=fp8, device=inp.device)
        scale_out = torch.empty(
            inp.shape[:-1] + (num_groups,), dtype=torch.float32, device=inp.device
        )
        # Optional bf16/fp16 mirror of the pre-quantization normed output.
        # Requested by GDN-style layers that also need an unquantized view
        # (e.g. Qwen3.5 in_proj_ba). Zero-overhead when not requested
        # because the kernel branches on the pointer being non-null.
        bf16_out = None
        bf16_ptr = 0
        if emit_bf16:
            bf16_out = torch.empty_like(inp)
            bf16_ptr = int(bf16_out.data_ptr())
        reg = 0 if registered else self._pool["input"].data_ptr
        reg_bytes = 0 if registered else self._pool["input"].max_size
        ops.fused_allreduce_rmsnorm_quant_per_group(
            self._ptr,
            inp,
            res_inp,
            res_out,
            out,
            scale_out,
            w,
            eps,
            group_size,
            reg,
            reg_bytes,
            use_1stage,
            bf16_ptr,
        )
        if emit_bf16:
            return out, res_out, scale_out, bf16_out
        return out, res_out, scale_out
    
    def fused_qknorm_ar(
        self,
        qkv_in: torch.Tensor,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        eps: float,
        registered: bool = False,
    ):
        dtype = qkv_in.dtype
        device = qkv_in.device
        hidden_dim_q = q_w.shape[-1]
        hidden_dim_k = k_w.shape[-1]
        token_num = qkv_in.shape[0]
        hidden_dim_v = qkv_in.shape[1] - (hidden_dim_q + hidden_dim_k)
        q_out = torch.empty((token_num, hidden_dim_q), dtype=dtype, device=device)
        k_out = torch.empty((token_num, hidden_dim_k), dtype=dtype, device=device)
        v_out = torch.empty((token_num, hidden_dim_v), dtype=dtype, device=device)
        reg = 0 if registered else self._pool["input"].data_ptr
        reg_bytes = 0 if registered else self._pool["input"].max_size
        ops.fused_qknorm_allreduce(
            self._ptr,
            qkv_in,
            q_w,
            k_w,
            q_out,
            k_out,
            v_out,
            eps,
            reg,
            reg_bytes,
        )
        return q_out, k_out, v_out
    
    def custom_fused_qknorm_ar(
        self,
        qkv_in: torch.Tensor,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        eps: float,
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = qkv_in.dtype
        if self.disabled:
            return (
                torch.empty((qkv_in.shape[0], q_w.shape[-1]), dtype=dtype, device=qkv_in.device),
                torch.empty((qkv_in.shape[0], k_w.shape[-1]), dtype=dtype, device=qkv_in.device),
                torch.empty((qkv_in.shape[0], qkv_in.shape[1] - q_w.shape[-1] - k_w.shape[-1]), dtype=dtype, device=qkv_in.device)
            )
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.fused_qknorm_ar(
                    qkv_in,
                    q_w,
                    k_w,
                    eps,
                    registered=self.enable_register_for_capturing,
                )
            else:
                return (
                    torch.empty((qkv_in.shape[0], q_w.shape[-1]), dtype=dtype, device=qkv_in.device),
                    torch.empty((qkv_in.shape[0], k_w.shape[-1]), dtype=dtype, device=qkv_in.device),
                    torch.empty((qkv_in.shape[0], qkv_in.shape[1] - q_w.shape[-1] - k_w.shape[-1]), dtype=dtype, device=qkv_in.device)
                )
        else:
            return self.fused_qknorm_ar(
                qkv_in,
                q_w,
                k_w,
                eps,
                registered=False,
            )

    def custom_fused_ar_rms_per_group_quant(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        group_size: int = 128,
        use_1stage: bool = False,
        emit_bf16: bool = False,
    ):
        if self.disabled or not self._should_custom_ar(input, _LEGACY_AR_MAX_SIZE):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.fused_ar_rms_per_group_quant(
                    input,
                    residual_inp,
                    w=weight,
                    eps=eps,
                    group_size=group_size,
                    registered=self.enable_register_for_capturing,
                    use_1stage=use_1stage,
                    emit_bf16=emit_bf16,
                )
            else:
                K = input.shape[-1]
                num_groups = K // group_size
                dummy_out = torch.zeros(input.shape, dtype=fp8, device=input.device)
                dummy_scale = torch.zeros(
                    input.shape[:-1] + (num_groups,),
                    dtype=torch.float32,
                    device=input.device,
                )
                if emit_bf16:
                    return (
                        dummy_out,
                        torch.zeros_like(input),
                        dummy_scale,
                        torch.zeros_like(input),
                    )
                return dummy_out, torch.zeros_like(input), dummy_scale
        else:
            return self.fused_ar_rms_per_group_quant(
                input,
                residual_inp,
                w=weight,
                eps=eps,
                group_size=group_size,
                registered=False,
                use_1stage=use_1stage,
                emit_bf16=emit_bf16,
            )

    def close(self):
        if not self.disabled and getattr(self, "_ptr", 0):
            try:
                ops.dispose(self._ptr)
            except (AttributeError, TypeError):
                pass
            self._ptr = 0

    def __del__(self):
        self.close()
