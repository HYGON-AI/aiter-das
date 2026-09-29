# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Implemented for AITER. The link predicate follows
# RocmPlatform.is_fully_connected in vLLM v0.8.5:
# https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/platforms/rocm.py#L179-L201
# PCI device mapping and collective agreement are implemented locally.
"""Topology admission for same-node HIP IPC collectives."""

import ctypes
import logging
from typing import List

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _device_pci_bus_id(device: torch.device) -> str:
    """Resolve the runtime device, including HIP/ROCR visibility remapping."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    # The runtime knows the mapping even for UUID masks and reordered devices.
    # Do not assume its ordinal matches the management library's enumeration.
    runtime = ctypes.CDLL("libamdhip64.so")
    get_bus_id = runtime.hipDeviceGetPCIBusId
    get_bus_id.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    get_bus_id.restype = ctypes.c_int
    bus_id = ctypes.create_string_buffer(64)
    status = get_bus_id(bus_id, len(bus_id), index)
    if status != 0:
        raise RuntimeError(f"hipDeviceGetPCIBusId({index}) failed: {status}")
    return bus_id.value.decode("ascii").lower()


def _query_hsl_connectivity(bus_ids: List[str]) -> bool:
    """Apply vLLM's ROCm predicate: every pair reports type=2 and hops=1.

    The SMI hop count is provider-defined, not proof of a physical full mesh.
    Missing identity/topology information must not enable the IPC fast path.
    """
    if len(set(bus_ids)) != len(bus_ids):
        raise ValueError("Each IPC rank must use a distinct physical GPU")
    import amdsmi

    amdsmi.amdsmi_init()
    try:
        handles = {
            amdsmi.amdsmi_get_gpu_device_bdf(handle).lower(): handle
            for handle in amdsmi.amdsmi_get_processor_handles()
        }
        selected = [handles[bus_id] for bus_id in bus_ids]
        for i, handle in enumerate(selected):
            for j in range(i + 1, len(selected)):
                link = amdsmi.amdsmi_topo_get_link_type(handle, selected[j])
                if link["type"] != 2 or link["hops"] != 1:
                    logger.warning(
                        "Custom IPC topology rejected %s <-> %s: type=%s, hops=%s "
                        "(requires type=2, hops=1)",
                        bus_ids[i], bus_ids[j], link["type"], link["hops"],
                    )
                    return False
        return True
    finally:
        amdsmi.amdsmi_shut_down()


def is_fully_connected(group: dist.ProcessGroup, device: torch.device) -> bool:
    """Collectively check the HIP devices in a non-NCCL, same-node group.

    Gather actual PCI identities, then broadcast one decision so a local probe
    failure cannot leave some ranks initializing custom AR while others fall back.
    This is an IPC admission check; Fabric transport has a separate contract.
    """
    try:
        local = {"bus_id": _device_pci_bus_id(device)}
    except Exception as exc:
        local = {"error": str(exc)}
    devices = [None] * dist.get_world_size(group)
    dist.all_gather_object(devices, local, group=group)

    result = [False]
    if dist.get_rank(group) == 0:
        errors = [item["error"] for item in devices if "error" in item]
        if errors:
            logger.warning("Custom IPC device identification failed: %s", errors)
        else:
            bus_ids = [item["bus_id"] for item in devices]
            try:
                result[0] = _query_hsl_connectivity(bus_ids)
            except Exception as exc:
                logger.warning("Custom IPC topology detection failed: %s", exc)
            logger.info(
                "Custom IPC topology: devices=%s fully_connected=%s",
                bus_ids, result[0],
            )
    source_rank = dist.get_global_rank(group, 0) if group is not None else 0
    dist.broadcast_object_list(result, src=source_rank, group=group)
    return result[0]
