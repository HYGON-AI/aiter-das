# SPDX-License-Identifier: MIT
"""AITER 兼容入口；workspace/meta 属于实现内部，不再作为包级导出。"""

from .api import chunk_kimi_delta_attn

__all__ = ["chunk_kimi_delta_attn"]
