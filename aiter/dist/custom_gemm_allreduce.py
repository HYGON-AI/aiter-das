'''
 * Copyright (c) 2024, The vLLM team.
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
 '''

from contextlib import contextmanager
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from aiter.dist.parallel_state import GroupCoordinator

from aiter.ops.triton.gemm_allreduce_a16w4 import (
    gemm_allreduce_a16w4,
)

from aiter.ops.triton.gemm_allreduce_w8a8 import (
    gemm_allreduce_w8a8,
    gemm_allreduce_w8a8_v2,
)

# use rocSHMEM
USE_NVSHMEM = False

if USE_NVSHMEM:
    import pynvshmem
    from pynvshmem import nvshmem_create_tensor_list_intra_node as shm_create_tensor_list
    from pynvshmem import nvshmem_create_tensor as shm_create_tensor
else:
    import pyrocshmem
    from pyrocshmem import rocshmem_create_tensor_list_intra_node as shm_create_tensor_list
    from pyrocshmem import rocshmem_create_tensor as shm_create_tensor

import aiter as ops
import os
from .custom_all_reduce_utils import (
    gpu_p2p_access_check)
from .parallel_state import in_the_same_node_as
from aiter import logger

def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if not gpu_p2p_access_check(rank, i):
            return False
    return True

# NOTE:
QUANTIZATION_METHOD_NAMES = [
    "awq",
    # TODO: support other types
]

def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b

class CustomGemmAllreduce:
    
    _SUPPORTED_QUANTIZATION_METHOD = ["awq", "blockwise_int8", "fp8"]
    
    # max_size: max supported allreduce size
    def __init__(self,
                 tp: GroupCoordinator,
                 pp: GroupCoordinator,
                 #group: ProcessGroup,
                 #local_world_size: int,
                 quant_method: str,
                 output_size: int,
                 min_block_size_m: Optional[int] = 16,
                 min_block_size_n: Optional[int] = 16,
                 max_seq_len: Optional[int] = 256) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            output_size: second dimension of matrix A. (N)
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self.disabled = True
        
        if quant_method not in CustomGemmAllreduce._SUPPORTED_QUANTIZATION_METHOD:
            logger.warning(
                "Custom gemm-allreduce is disabled due to an unsupported quant method"
                ": %s. Supported quant methods: %s",
                quant_method, str(CustomGemmAllreduce._SUPPORTED_QUANTIZATION_METHOD))
            return
        
        self.tp = tp
        self.pp = pp
        self.world_size = tp.world_size
        self.cur_rank = tp.rank_in_group
        # self.local_world_size = local_world_size
        # self.nnodes = self.world_size // self.local_world_size
        self.max_seq_len = max_seq_len
        self.quant_method = quant_method
        self.min_block_size_m = min_block_size_m
        self.min_block_size_n = min_block_size_n

        # shmem buffer
        try:
            if USE_NVSHMEM:
                pynvshmem.init_nvshmem_by_uniqueid(tp.device_group)
                self.rocshmem_ctx = None
            else:
                start_rank = pp.rank_in_group * tp.world_size
                pyrocshmem.init_rocshmem_by_uniqueid(start_rank, tp.device_group)
                self.rocshmem_ctx = pyrocshmem.rocshmemx_get_device_ctx_cached()
        except Exception as e:
            logger.exception(f"Error in request: {e}")
            logger.warning(
                "pyrocshmem init failed")
            return

        # NOTE: make sure that this is the minimum block size in awq-gemm configs
        #       this size will be checked also in gemm_allreduce_a16w4
        self.max_tiles = \
            cdiv(max_seq_len, min_block_size_m) * cdiv(output_size, min_block_size_n)

        # TODO: group the return vals by (bufs), (buf)
        self.barriers_buf, self.scatters_buf, self.reduces_buf, self.barrier_buf, self.scatter_buf, self.reduce_buf = \
            self.register_shmem(output_size)
        
        if self.barriers_buf is None or self.scatters_buf is None:
            logger.warning(
                "Custom gemm-allreduce register shmem failed, quant_method"
                ": %s. output_size: %d",
                quant_method, output_size)
            return
        
        self.disabled = False

    # NOTE: this func/args are specific according to kernel design
    def register_shmem(
        self,
        output_size: int, # N
    ) -> List[torch.Tensor]:

        # barrier buffer
        barrier_bufs = shm_create_tensor_list((self.max_tiles, ), dtype=torch.int32)
        barrier_bufs[self.cur_rank].zero_()
        barriers_buf = torch.tensor([t.data_ptr() for t in barrier_bufs], device=barrier_bufs[self.cur_rank].device)
        # barriers_buf = shm_create_tensor((self.max_tiles, ), dtype=torch.int32)
        # barriers_buf.zero_()

        # scatter buffer
        scatter_bufs = shm_create_tensor_list([self.max_seq_len, output_size], dtype=torch.float32)
        scatters_buf = torch.tensor([t.data_ptr() for t in scatter_bufs], device=scatter_bufs[self.cur_rank].device)
        # scatters_buf = shm_create_tensor([self.max_seq_len, output_size], dtype=torch.float32)

        # reduce_buffer
        # output_size_per_rank = cdiv(output_size, self.local_world_size)
        reduce_bufs = shm_create_tensor_list([self.max_seq_len, output_size], dtype=torch.float16)
        reduces_buf = torch.tensor([t.data_ptr() for t in reduce_bufs], device=scatter_bufs[self.cur_rank].device)

        return [barriers_buf, scatters_buf, reduces_buf, barrier_bufs[self.cur_rank], scatter_bufs[self.cur_rank], reduce_bufs[self.cur_rank]]

    def close(self):
        if not self.disabled:
            if self.barriers_buf is not None:
                del self.barriers_buf
            if self.scatters_buf is not None:
                del self.scatters_buf

    def __del__(self):
        self.close()
    
    def gemm_allreduce(
        self,
        input: torch.tensor,
        *args,
    ):
        if self.quant_method == "awq":
            qweight, scales, qzeros = args
            return gemm_allreduce_a16w4(
                input, qweight, scales, qzeros,
                self.barriers_buf, self.scatters_buf,
                self.max_tiles,
                self.group,
                self.rocshmem_ctx)
        elif self.quant_method == "blockwise_int8" or self.quant_method == "fp8":
            weight, x_scale, w_scale, block_size, dtype = args
            # return gemm_allreduce_w8a8(
            #     input, weight, x_scale, w_scale, block_size, dtype,
            #     self.barriers_buf, self.scatters_buf,
            #     self.barrier_buf, self.scatter_buf,
            #     self.max_tiles,
            #     self.group,
            #     self.local_world_size,
            #     self.rocshmem_ctx)
            return gemm_allreduce_w8a8_v2(
                input, weight, x_scale, w_scale, block_size, dtype,
                self.barriers_buf, self.scatters_buf, self.reduces_buf,
                self.barrier_buf, self.scatter_buf, self.reduce_buf,
                self.max_seq_len,
                self.max_tiles,
                self.tp,
                self.pp,
                self.rocshmem_ctx)

            # for debug
            # if self.group.rank() == 0:
            #     print(f"{self.group.rank()=} {out.shape=} {self.scatter_buf[:1,::384]} {out[:1, 0]}")
            #     print(f"{self.group.rank()=} {self.scatter_buf.shape=} {self.scatter_buf[:4,:384]=}")
            #     print(f"{self.group.rank()=} {self.scatter_buf.shape=} {self.scatter_buf[:4,384:]=}")
            # return out

        # TODO: support more quant types here
        # currently should not reach here
        assert self.quant_method in self._SUPPORTED_QUANTIZATION_METHOD, (
                f"{self.quant_method=} currently not supported")
        
    def should_custom_gemm_allreduce(self, input: torch.tensor):
        if self.disabled:
            return False
        if input.shape[0] > self.max_seq_len:
            return False
        return True
            
    