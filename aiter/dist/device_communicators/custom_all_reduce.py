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
from aiter.dist.device_communicators.topology import is_fully_connected
from aiter import logger
from aiter.utility.dtypes import fp8 as _platform_fp8
# gfx936 has no native FP8 conversion. Communication kernels encode E4M3FN
# in software, so expose a real FP8 tensor rather than interpreting bytes as ints.
fp8 = torch.float8_e4m3fn if _platform_fp8 == torch.uint8 else _platform_fp8

try:
    ops.meta_size()
    custom_ar = True
except Exception as e:
    # For CPUs
    custom_ar = False
    logger.warning(f"Custom allreduce is disabled: {e}")


_AR_TRANSPORT_IDS = {"ipc": 0, "fabric": 1}
# Match AMD's default ordinary custom-AR cutoff (64 MiB).
_DEFAULT_AR_MAX_SIZE_MB = 64
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


def _validate_mxfp4_hidden_dim(n: int, element_size: int) -> None:
    """Validate hidden-dim constraints for the fused AR+RMSNorm+MXFP4 epilogue."""
    if element_size <= 0 or 16 % element_size != 0:
        raise ValueError(
            "MXFP4 fused quant requires an element_size that divides 16 "
            f"(bf16/fp16: 2), got element_size={element_size}"
        )
    if n <= 0:
        raise ValueError(f"MXFP4 fused quant requires hidden_dim n > 0, got n={n}")
    pack_size = 16 // element_size
    if n % 32 != 0:
        raise ValueError(f"MXFP4 fused quant requires n divisible by 32, got n={n}")
    if n % pack_size != 0:
        raise ValueError(
            f"MXFP4 fused quant requires n divisible by PACK_SIZE={pack_size}, got n={n}"
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

    def close(self):
        if self._uncached and self._raw_ptr:
            ops.free_meta_buffer(self._raw_ptr)
        self._raw_ptr = 0
        self._buffer = None

    def __del__(self):
        try:
            self.close()
        except (AttributeError, TypeError, RuntimeError):
            pass


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

    def close(self):
        """Release local allocations after native peer mappings are disposed."""
        for buf in self._buffers.values():
            buf.close()
        self._buffers.clear()

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
                    "next communicator. Rebuild module_custom_all_reduce with "
                    "supernode support enabled (the build default)."
                )
                return
        else:
            self.transport = "fabric"
            if not _fabric_ar_available():
                raise RuntimeError(
                    "AITER_AR_TRANSPORT=fabric was requested, but fabric support is "
                    "not compiled. Rebuild module_custom_all_reduce with supernode "
                    "support enabled (the build default)."
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

        # Preserve the two-rank path and Fabric's separate transport contract.
        # HIP IPC with more than two ranks requires the vLLM ROCm link predicate.
        fully_connected = True
        if self.transport == "ipc" and world_size > 2 and torch.version.hip:
            fully_connected = is_fully_connected(self.group, self.device)
        self.fully_connected = fully_connected
        if world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce IPC is disabled: the selected GPU group "
                "does not have confirmed type=2, hops=1 connectivity for every "
                "pair. Falling back to the next communicator."
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
        if self.transport == "ipc" and enable_register_for_capturing and torch.version.hip:
            arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
            if arch in ("gfx936", "gfx938"):
                # On gfx936/gfx938, direct IPC reads of cached graph allocations can
                # observe stale inputs. End barriers and system-scope fences
                # alone do not fix this (reproduced on DTK 26.04). The existing
                # uncached staging pool provides the validated copy-in path.
                logger.warning(
                    "%s IPC graph input registration can return stale data; "
                    "using copy-in mode despite enable_register_for_capturing=True. "
                    "AITER custom allreduce remains enabled.",
                    arch,
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

        # Opt-in until per-architecture thresholds have passed correctness and
        # performance gates. Agree before allocating/exporting any extra memory.
        requested_small_ar = os.environ.get("AITER_AR_SMALL_MESSAGE", "off").strip().lower()
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0] if torch.version.hip else ""
        small_capable = self.transport == "ipc" and world_size in (2, 4, 8) and arch in ("gfx936", "gfx938")
        small_configs = [None] * world_size
        dist.all_gather_object(small_configs, (requested_small_ar, max_size, small_capable), group=self.group)
        if any(c[:2] != small_configs[0][:2] for c in small_configs):
            raise ValueError("AITER_AR_SMALL_MESSAGE and max_size must agree across ranks")
        small_modes = {"off": 0, "ll": 1, "ll128": 2, "auto": 3}
        if requested_small_ar not in small_modes:
            raise ValueError("AITER_AR_SMALL_MESSAGE must be off, ll, ll128, or auto")
        self.small_ar_mode = "off"
        small_bytes = 0
        meta_bytes = ops.meta_size() + 2 * max_size
        # All ranks must choose the same protocol, including heterogeneous groups.
        if requested_small_ar != "off" and all(c[2] for c in small_configs):
            small_bytes = ops.small_ar_scratch_size(world_size)
            if max_size >= small_bytes:
                self.small_ar_mode = requested_small_ar
                meta_bytes = ((meta_bytes + 127)//128)*128 + small_bytes
            else:
                logger.warning("Small-message CAR scratch needs %d bytes; max_size=%d uses the traditional path", small_bytes, max_size)

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
            self._pool.create("meta", meta_bytes, uncached=True)
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

        init_error = None
        try:
            if self.small_ar_mode != "off":
                ops.configure_small_ar(self._ptr, max_size, meta_bytes, small_modes[self.small_ar_mode])
        except Exception as e:
            init_error = e
        if not self._sync_init_stage("small-message scratch configuration", init_error):
            return

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
        if inp_size == 0 or inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.fully_connected:
            return inp_size <= max_bytes
        return False

    def should_custom_ag(self, inp: torch.Tensor, dim: int = 0):
        if self.disabled or self.world_size not in (2, 4, 8):
            return False
        if inp.device != self.device:
            return False
        if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.int16, torch.int32, torch.int64):
            return False
        if inp.numel() == 0 or not inp.is_contiguous():
            return False
        if not -inp.ndim <= dim < inp.ndim:
            return False
        dim %= inp.ndim
        if dim not in (0, inp.ndim - 1):
            return False
        if dim != 0 and inp.shape[-1] * inp.element_size() % 16:
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
        if inp.numel() == 0 or inp.numel() * inp.element_size() % 16:
            raise ValueError("all_reduce requires a positive multiple of 16 input bytes")
        if not is_weak_contiguous(inp):
            raise ValueError("all_reduce input must be weak-contiguous")
        if out is None:
            out = torch.empty_like(inp)
        if out.numel() != inp.numel() or out.dtype != inp.dtype or out.device != inp.device:
            raise ValueError("all_reduce output size, dtype and device must match input")
        assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
        # DTK 26.04 also exposes stale peer inputs in ordinary AR when a
        # framework replays a graph with new data. Preserve the registration
        # request API and use the validated graph-local copy-in capability.
        registered_input = registered_input and not ops.ag_rs_requires_copy_in()
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
            # custom allreduce includes a staging copy. Its latency depends
            # on payload and platform; measure it separately when profiling.
            return self.all_reduce(
                input,
                use_new=use_new,
                open_fp8_quant=open_fp8_quant,
                registered_input=False,
            )

    _RS_SPLIT_FIRST, _RS_SPLIT_LAST, _RS_SPLIT_MID = 0, 1, 2

    @staticmethod
    def _compute_rs_args(shape, dim, numel):
        if dim == 0:
            return 0, 0, numel, 0
        m, k = 1, 1
        for size in shape[:dim]:
            m *= size
        for size in shape[dim + 1:]:
            k *= size
        if dim == len(shape) - 1:
            return 0, m, shape[dim], 1
        return m, shape[dim], k, 2

    def should_custom_rs(self, inp: torch.Tensor, dim: int = 0) -> bool:
        if self.disabled or self.world_size not in (2, 4, 8):
            return False
        if inp.device != self.device:
            return False
        if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return False
        if not inp.is_contiguous() or not self._should_custom_ar(
            inp, min(_LEGACY_AR_MAX_SIZE, self.max_size // 2)
        ):
            return False
        ndim = inp.dim()
        if not -ndim <= dim < ndim or inp.numel() == 0:
            return False
        dim %= ndim
        if inp.shape[dim] % self.world_size:
            return False
        return dim != 0 or inp.numel() % (self.world_size * (16 // inp.element_size())) == 0

    def reduce_scatter(self, inp: torch.Tensor, out: torch.Tensor, dim: int = 0, *, registered: bool = False):
        if not self.should_custom_rs(inp, dim):
            raise ValueError("unsupported reduce_scatter input, dimension or communicator")
        dim %= inp.dim()
        shape = list(inp.shape)
        shape[dim] //= self.world_size
        if tuple(out.shape) != tuple(shape) or out.dtype != inp.dtype or out.device != inp.device or not out.is_contiguous():
            raise ValueError("reduce_scatter output must match the contiguous rank shard")
        self._check_disjoint(inp, out)
        if registered and self.transport == "fabric":
            raise ValueError("fabric reduce_scatter requires copy-in mode")
        # DTK graph launches do not guarantee visibility of mutable peer
        # input allocations. Copy-in is graph-safe and preserves the API.
        registered = registered and not ops.ag_rs_requires_copy_in()
        m, n, k, split_dim = self._compute_rs_args(tuple(inp.shape), dim, inp.numel())
        reg = 0 if registered else self._pool["input"].data_ptr
        reg_bytes = 0 if registered else self._pool["input"].max_size
        ops.reduce_scatter_dim(self._ptr, inp, out, m, n, k, split_dim, reg, reg_bytes)

    def custom_reduce_scatter(self, input: torch.Tensor, output: torch.Tensor, dim: int = 0):
        if not self.should_custom_rs(input, dim):
            return None
        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing() and self.enable_register_for_capturing
        # Warmup must produce real data for callers that retain derived state.
        return self.reduce_scatter(input, output, dim, registered=registered)

    def _allgather_out_shape(self, inp: torch.Tensor, dim: int):
        if inp.dim() == 0 or not -inp.dim() <= dim < inp.dim():
            raise ValueError("invalid all_gather dimension")
        dim %= inp.dim()
        if dim not in (0, inp.dim() - 1):
            raise ValueError("all_gather supports only the first or last dimension")
        shape = list(inp.shape)
        shape[dim] *= self.world_size
        return tuple(shape)

    # Copying int64 as two fp32 lanes also works with H's Tensor ABI, which
    # does not expose float64. There is no arithmetic in the AG kernels.
    _INT_TO_FP_VIEW = {torch.int16: torch.float16, torch.int32: torch.float32, torch.int64: torch.float32}

    @staticmethod
    def _check_disjoint(inp, out):
        input_start, output_start = inp.data_ptr(), out.data_ptr()
        if (input_start < output_start + out.numel() * out.element_size()
                and output_start < input_start + inp.numel() * inp.element_size()):
            raise ValueError("communication input and output must not overlap")

    def _all_gather_tensors(self, inp, out, dim):
        if not self.should_custom_ag(inp, dim):
            raise ValueError("unsupported all_gather input or communicator")
        shape = self._allgather_out_shape(inp, dim)
        dim %= inp.dim()
        view_dtype = self._INT_TO_FP_VIEW.get(inp.dtype, inp.dtype)
        native_inp = inp.view(view_dtype) if view_dtype != inp.dtype else inp
        if dim != 0 and native_inp.shape[-1] % (16 // native_inp.element_size()):
            raise ValueError("all_gather last dimension must be 16-byte aligned")
        if out is None:
            out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
        if tuple(out.shape) != shape or out.dtype != inp.dtype or out.device != inp.device or not out.is_contiguous():
            raise ValueError("all_gather output shape, dtype, device or layout mismatch")
        self._check_disjoint(inp, out)
        native_out = out.view(view_dtype) if view_dtype != inp.dtype else out
        return out, native_inp, native_out, dim

    def all_gather_reg(self, inp: torch.Tensor, out: torch.Tensor = None, dim: int = 0):
        if self.transport == "fabric":
            raise ValueError("fabric all_gather requires copy-in mode")
        if ops.ag_rs_requires_copy_in():
            return self.all_gather_unreg(inp, out=out, dim=dim)
        out, native_inp, native_out, dim = self._all_gather_tensors(inp, out, dim)
        ops.all_gather_reg(self._ptr, native_inp, native_out, dim)
        return out

    def all_gather_unreg(self, inp: torch.Tensor, out: torch.Tensor = None, dim: int = 0):
        out, native_inp, native_out, dim = self._all_gather_tensors(inp, out, dim)
        ops.all_gather_unreg(self._ptr, native_inp, self._pool["input"].data_ptr, native_out, self._pool["input"].max_size, dim)
        return out

    def custom_all_gather(self, inp: torch.Tensor, dim: int = 0) -> Optional[torch.Tensor]:
        if not self.should_custom_ag(inp, dim):
            return None
        if self._IS_CAPTURING and torch.cuda.is_current_stream_capturing() and self.enable_register_for_capturing:
            return self.all_gather_reg(inp, dim=dim)
        return self.all_gather_unreg(inp, dim=dim)

    def should_custom_ar_bytes(self, inp: torch.Tensor, prefill_support: bool = False):
        if self.disabled or self.world_size not in (2, 4, 8):
            return False
        if inp.device != self.device or inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if inp.ndim != 2 or inp.stride(-1) != 1 or inp.stride(0) < inp.shape[-1]:
            return False
        size = inp.numel() * inp.element_size()
        return (0 < size <= min(_LEGACY_AR_MAX_SIZE, self.max_size // 2)
                and size % 16 == 0 and (self.world_size == 2 or self.fully_connected))

    def _validate_fused_inputs(self, inp, residual, weight, *, packed=False, quant=False):
        if self.disabled or self.world_size not in (2, 4, 8):
            raise ValueError("fused communication requires an enabled 2/4/8-rank communicator")
        if inp.ndim < 2 or inp.numel() == 0:
            raise ValueError("fused communication requires a nonempty tensor with at least two dimensions")
        if inp.device != self.device or inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("unsupported fused input dtype or device")
        if not inp.is_contiguous() and not (packed and inp.ndim == 2 and inp.stride(-1) == 1 and inp.stride(0) >= inp.shape[-1]):
            raise ValueError("input must be contiguous or a supported last-dimension slice")
        if weight.ndim != 1 or not weight.is_contiguous() or weight.dtype != inp.dtype or weight.device != inp.device:
            raise ValueError("weight must be a contiguous vector matching input dtype and device")
        valid = weight.numel()
        pack = 16 // inp.element_size()
        if valid <= 0 or valid % pack or inp.shape[-1] < valid or inp.shape[-1] % pack:
            raise ValueError("input and weight widths must satisfy the 16-byte pack contract")
        expected = tuple(inp.shape[:-1]) + (valid,)
        if tuple(residual.shape) != expected or residual.dtype != inp.dtype or residual.device != inp.device or not residual.is_contiguous():
            raise ValueError("residual must match the valid unpadded shape, dtype and device")
        if inp.numel() * inp.element_size() > min(_LEGACY_AR_MAX_SIZE, self.max_size // 2):
            raise ValueError("fused input exceeds communicator capacity")
        if quant and (inp.shape[-1] != valid or valid // pack > 1024):
            raise ValueError("quantized fusion requires matching widths and at most 1024 packed threads")
        return valid

    @staticmethod
    def _validate_fused_output(out, shape, dtype, device, name):
        if tuple(out.shape) != tuple(shape) or out.dtype != dtype or out.device != device or not out.is_contiguous():
            raise ValueError(f"{name} shape, dtype, device or layout mismatch")

    def fused_ar_rms(
        self,
        inp: torch.Tensor,
        res_inp: torch.Tensor,
        *,
        res_out: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        scale_out: torch.Tensor | None = None,
        w: torch.Tensor,
        eps: float,
        registered: bool = False,
        use_1stage: bool = False,
        post_per_token_quant: bool = False,
        out_hidden_dim: int = 0,
        gemma_norm: bool = False,
        emit_bf16: bool = False,
    ):
        valid_dim = self._validate_fused_inputs(inp, res_inp, w, packed=True, quant=post_per_token_quant)
        if not inp.is_contiguous():
            registered = False
        if out_hidden_dim and (out_hidden_dim < valid_dim or out_hidden_dim % (16 // inp.element_size())):
            raise ValueError("padded output width must contain the valid columns and be 16-byte aligned")
        if res_out is None:
            res_out = torch.empty(
                inp.shape[:-1] + (valid_dim,), dtype=inp.dtype, device=inp.device
            )
        self._validate_fused_output(res_out, inp.shape[:-1] + (valid_dim,), inp.dtype, inp.device, "residual output")
        # DTK external registered inputs have the same changing-replay
        # visibility limitation as AG/RS. Keep communication inside the graph
        # and copy into the pre-registered pool before the fused kernel.
        registered = registered and not ops.ag_rs_requires_copy_in()
        reg = 0 if registered else self._pool["input"].data_ptr
        reg_bytes = 0 if registered else self._pool["input"].max_size
        if not post_per_token_quant:
            if out is None:
                out_dim = out_hidden_dim or inp.shape[-1]
                out = torch.empty(
                    inp.shape[:-1] + (out_dim,), dtype=inp.dtype, device=inp.device
                )
            self._validate_fused_output(out, inp.shape[:-1] + (out_hidden_dim or inp.shape[-1],), inp.dtype, inp.device, "norm output")
            if inp.shape[-1] == valid_dim and out.shape[-1] == inp.shape[-1]:
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
                    gemma_norm,
                )
            else:
                ops.fused_allreduce_rmsnorm_pad(
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
                    gemma_norm,
                )
            return out, res_out
        else:
            if out is None:
                out = torch.empty(inp.shape, dtype=fp8, device=inp.device)
            self._validate_fused_output(out, inp.shape, fp8, inp.device, "quant output")
            if scale_out is None:
                scale_out = torch.empty(
                    inp.shape[:-1] + (1,), dtype=torch.float32, device=inp.device
                )
            self._validate_fused_output(scale_out, inp.shape[:-1] + (1,), torch.float32, inp.device, "scale output")
            # Optional pre-quantization bf16/fp16 mirror of the normed output.
            # Requested by v32 DSA models (e.g. GLM-5.2) whose indexer GEMMs run
            # in bf16 while attention QKV keeps per-token FP8. Zero-overhead when
            # not requested because the kernel branches on the pointer being null.
            bf16_out = None
            bf16_ptr = 0
            if emit_bf16:
                bf16_out = torch.empty_like(inp)
                bf16_ptr = int(bf16_out.data_ptr())
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
                gemma_norm,
                bf16_ptr,
            )
            if emit_bf16:
                return out, res_out, scale_out, bf16_out
            return out, res_out, scale_out

    def custom_fused_ar_rms(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool = False,
        out_hidden_dim: int = 0,
        gemma_norm: bool = False,
    ) -> torch.Tensor | None:
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self._should_custom_ar(input, min(_LEGACY_AR_MAX_SIZE, self.max_size // 2)):
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
                    out_hidden_dim=out_hidden_dim,
                    gemma_norm=gemma_norm,
                )
            else:
                out_dim = out_hidden_dim or input.shape[-1]
                return (
                    torch.zeros(
                        input.shape[:-1] + (out_dim,),
                        dtype=input.dtype,
                        device=input.device,
                    ),
                    torch.zeros(
                        input.shape[:-1] + (weight.numel(),),
                        dtype=input.dtype,
                        device=input.device,
                    ),
                )
        else:
            return self.fused_ar_rms(
                input,
                residual_inp,
                w=weight,
                eps=eps,
                registered=False,
                use_1stage=use_1stage,
                out_hidden_dim=out_hidden_dim,
                gemma_norm=gemma_norm,
            )

    def custom_fused_ar_rms_quant(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool = False,
        gemma_norm: bool = False,
        emit_bf16: bool = False,
    ):
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self._should_custom_ar(input, min(_LEGACY_AR_MAX_SIZE, self.max_size // 2)):
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
                    gemma_norm=gemma_norm,
                    emit_bf16=emit_bf16,
                )
            else:
                dummy_out = torch.zeros(input.shape, dtype=fp8, device=input.device)
                dummy_scale_out = torch.zeros(
                    input.shape[:-1] + (1,), dtype=torch.float32, device=input.device
                )
                if emit_bf16:
                    return (
                        dummy_out,
                        torch.zeros_like(input),
                        dummy_scale_out,
                        torch.zeros_like(input),
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
                gemma_norm=gemma_norm,
                emit_bf16=emit_bf16,
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
        transpose_scale: bool = False,
    ):
        self._validate_fused_inputs(inp, res_inp, w, quant=True)
        if transpose_scale and inp.ndim != 2:
            raise ValueError("transpose_scale requires a two-dimensional input")
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
        if transpose_scale:
            M = inp.shape[0]
            scale_out = torch.empty(
                (num_groups, M), dtype=torch.float32, device=inp.device
            ).transpose(0, 1)
        else:
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
        # DTK external registered inputs have the same changing-replay
        # visibility limitation as AG/RS. Keep communication inside the graph
        # and copy into the pre-registered pool before the fused kernel.
        registered = registered and not ops.ag_rs_requires_copy_in()
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
            transpose_scale,
        )
        if emit_bf16:
            return out, res_out, scale_out, bf16_out
        return out, res_out, scale_out
    
    def fused_qknorm_ar_rope(
        self,
        qkv_in: torch.Tensor,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        position_ids: torch.Tensor,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        registered: bool = False,
    ):
        if self.disabled:
            raise RuntimeError("QK RoPE requires an enabled custom communicator")
        if qkv_in.numel() * qkv_in.element_size() > self._pool["input"].max_size:
            raise ValueError("QK RoPE input exceeds communicator capacity")
        registered = registered and not ops.ag_rs_requires_copy_in()
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
        ops.fused_qknorm_allreduce_rope(
            self._ptr,
            qkv_in,
            q_w,
            k_w,
            q_out,
            k_out,
            v_out,
            cos_sin_cache,
            position_ids,
            head_dim,
            rotary_dim,
            eps,
            reg,
            reg_bytes,
        )
        return q_out, k_out, v_out


    def custom_fused_qknorm_ar_rope(self, qkv_in, q_w, k_w, cos_sin_cache,
                                  position_ids, head_dim, rotary_dim, eps):
        return self.fused_qknorm_ar_rope(qkv_in, q_w, k_w, cos_sin_cache,
            position_ids, head_dim, rotary_dim, eps,
            registered=self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
                and self.enable_register_for_capturing)

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
        # DTK external registered inputs have the same changing-replay
        # visibility limitation as AG/RS. Keep communication inside the graph
        # and copy into the pre-registered pool before the fused kernel.
        registered = registered and not ops.ag_rs_requires_copy_in()
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
            raise RuntimeError("QK norm allreduce requires an enabled custom communicator")
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
                    torch.zeros((qkv_in.shape[0], q_w.shape[-1]), dtype=dtype, device=qkv_in.device),
                    torch.zeros((qkv_in.shape[0], k_w.shape[-1]), dtype=dtype, device=qkv_in.device),
                    torch.zeros((qkv_in.shape[0], qkv_in.shape[1] - q_w.shape[-1] - k_w.shape[-1]), dtype=dtype, device=qkv_in.device)
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
        transpose_scale: bool = False,
    ):
        if self.disabled or not self._should_custom_ar(input, min(_LEGACY_AR_MAX_SIZE, self.max_size // 2)):
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
                    transpose_scale=transpose_scale,
                )
            else:
                K = input.shape[-1]
                num_groups = K // group_size
                dummy_out = torch.zeros(input.shape, dtype=fp8, device=input.device)
                if transpose_scale:
                    M = input.shape[0]
                    dummy_scale = torch.zeros(
                        (num_groups, M), dtype=torch.float32, device=input.device
                    ).transpose(0, 1)
                else:
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
                transpose_scale=transpose_scale,
            )

    def custom_fused_ar_rms_packed_input(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool = False,
        out_hidden_dim: int = 0,
        prefill_support: bool = False,
        gemma_norm: bool = False,
    ) -> torch.Tensor | None:
        # Let the C++ wrapper pack supported last-dim sliced views directly
        # into the registered IPC buffer so eager and graph paths both avoid
        # materializing an intermediate contiguous tensor in Python.
        if self.disabled or not self.should_custom_ar_bytes(input, prefill_support):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.fused_ar_rms(
                    input,
                    residual_inp,
                    w=weight,
                    eps=eps,
                    registered=False,
                    use_1stage=use_1stage,
                    out_hidden_dim=out_hidden_dim,
                    gemma_norm=gemma_norm,
                )
            else:
                out_dim = out_hidden_dim or input.shape[-1]
                return (
                    torch.zeros(
                        input.shape[:-1] + (out_dim,),
                        dtype=input.dtype,
                        device=input.device,
                    ),
                    torch.zeros(
                        input.shape[:-1] + (weight.numel(),),
                        dtype=input.dtype,
                        device=input.device,
                    ),
                )
        return self.fused_ar_rms(
            input,
            residual_inp,
            w=weight,
            eps=eps,
            registered=False,
            use_1stage=use_1stage,
            out_hidden_dim=out_hidden_dim,
            gemma_norm=gemma_norm,
        )

    def fused_ar_rms_mxfp4_quant(
        self,
        inp: torch.Tensor,
        res_inp: torch.Tensor,
        *,
        w: torch.Tensor,
        eps: float,
        registered: bool = False,
        use_1stage: bool = False,
        emit_bf16: bool = False,
    ):
        self._validate_fused_inputs(inp, res_inp, w, quant=True)
        if inp.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MXFP4 fusion supports FP16 and BF16 inputs")
        K = inp.shape[-1]
        _validate_mxfp4_hidden_dim(K, inp.element_size())
        res_out = torch.empty_like(inp)
        out = torch.empty(
            inp.shape[:-1] + (K // 2,), dtype=torch.uint8, device=inp.device
        )
        scale_out = torch.empty(
            inp.shape[:-1] + (K // 32,), dtype=torch.uint8, device=inp.device
        )
        bf16_out = None
        bf16_ptr = 0
        if emit_bf16:
            bf16_out = torch.empty_like(inp)
            bf16_ptr = int(bf16_out.data_ptr())
        # DTK external registered inputs have the same changing-replay
        # visibility limitation as AG/RS. Keep communication inside the graph
        # and copy into the pre-registered pool before the fused kernel.
        registered = registered and not ops.ag_rs_requires_copy_in()
        reg = 0 if registered else self._pool["input"].data_ptr
        reg_bytes = 0 if registered else self._pool["input"].max_size
        ops.fused_allreduce_rmsnorm_mxfp4_quant(
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
            bf16_ptr,
        )
        if emit_bf16:
            return out, res_out, scale_out, bf16_out
        return out, res_out, scale_out

    def custom_fused_ar_rms_mxfp4_quant(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool = False,
        emit_bf16: bool = False,
    ):
        if self.disabled or not self._should_custom_ar(input, min(_LEGACY_AR_MAX_SIZE, self.max_size // 2)):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.fused_ar_rms_mxfp4_quant(
                    input,
                    residual_inp,
                    w=weight,
                    eps=eps,
                    registered=self.enable_register_for_capturing,
                    use_1stage=use_1stage,
                    emit_bf16=emit_bf16,
                )
            else:
                K = input.shape[-1]
                dummy_out = torch.zeros(
                    input.shape[:-1] + (K // 2,), dtype=torch.uint8, device=input.device
                )
                dummy_scale = torch.zeros(
                    input.shape[:-1] + (K // 32,),
                    dtype=torch.uint8,
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
        return self.fused_ar_rms_mxfp4_quant(
            input,
            residual_inp,
            w=weight,
            eps=eps,
            registered=False,
            use_1stage=use_1stage,
            emit_bf16=emit_bf16,
        )

    def close(self):
        """Close after all ranks have finished using this communicator's graphs.

        This method is deliberately not collective: callers must drain and
        coordinate their ranks before teardown. Replaying a graph after close
        is invalid, just as using a disposed native handle is invalid.
        """
        if getattr(self, "_ptr", 0):
            torch.cuda.synchronize(self.device)
            ops.dispose(self._ptr)
            self._ptr = 0
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.close()
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except (AttributeError, TypeError, RuntimeError):
            pass
