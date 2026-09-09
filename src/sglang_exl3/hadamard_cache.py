"""RadixAttention Prefix-Cached Hadamard Transforms for EXL3 MoE (SGLANG_OPT_MOE_QUANT_ONCE).

In EXL3, routed MoE experts multiply input activations transformed by Walsh-Hadamard
matrices (H * X). In multi-turn chat and shared-prefix workloads, recomputing the
Hadamard transform at every layer for repeated prefix tokens is redundant.

This module provides a prefix-keyed cache for Hadamard-transformed activations,
reusing the transformed representation when RadixAttention prefix hits occur.
"""

from __future__ import annotations

import collections
import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger("sglang.exl3.hadamard_cache")

OPT_MOE_QUANT_ONCE_ENV = "SGLANG_OPT_MOE_QUANT_ONCE"
MAX_CACHE_ENTRIES_DEFAULT = 128


def is_moe_quant_once_enabled() -> bool:
    """Return whether RadixAttention Hadamard prefix caching is enabled."""
    val = os.environ.get(OPT_MOE_QUANT_ONCE_ENV, "1").strip().lower()
    return val in {"1", "true", "yes", "on"}


class HadamardPrefixCache:
    """LRU cache storing pre-transformed Hadamard activations keyed by (prefix_hash, layer_id)."""

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES_DEFAULT):
        self.max_entries = max_entries
        self._cache: collections.OrderedDict[Tuple[int, int], torch.Tensor] = collections.OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, prefix_hash: int, layer_id: int, prefix_len: int) -> Optional[torch.Tensor]:
        if not is_moe_quant_once_enabled() or prefix_hash is None or prefix_len <= 0:
            return None

        key = (prefix_hash, layer_id)
        if key in self._cache:
            cached = self._cache[key]
            # Ensure cached tensor has at least prefix_len tokens
            if cached.shape[0] >= prefix_len:
                self._cache.move_to_end(key)
                self.hits += 1
                return cached[:prefix_len]

        self.misses += 1
        return None

    def put(self, prefix_hash: int, layer_id: int, transformed: torch.Tensor) -> None:
        if not is_moe_quant_once_enabled() or prefix_hash is None:
            return

        key = (prefix_hash, layer_id)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)
        self._cache[key] = transformed.detach()

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# Global singleton cache instance
_GLOBAL_HADAMARD_CACHE = HadamardPrefixCache()


def get_global_hadamard_cache() -> HadamardPrefixCache:
    return _GLOBAL_HADAMARD_CACHE


def transform_hadamard_fast(x: torch.Tensor, suh: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes row-wise Walsh-Hadamard transform (blocks of 128) with optional suh scaling."""
    # Ensure 2D tensor
    x2d = x.view(-1, x.shape[-1])
    n_rows, hidden = x2d.shape
    device = x.device
    dtype = x.dtype

    # Walsh-Hadamard matrix construction for 128-element block
    block_size = 128
    if hidden % block_size != 0:
        # Fallback identity if hidden dim is not multiple of 128
        return x

    # Construct normalized Hadamard matrix if not cached on device
    h_mat = getattr(transform_hadamard_fast, f"_h_{device.type}", None)
    if h_mat is None or h_mat.device != device:
        # 128x128 Hadamard
        h = torch.tensor([[1.0]], dtype=torch.float32, device=device)
        for _ in range(7):  # 2^7 = 128
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        h_mat = (h * (1.0 / (128.0 ** 0.5))).to(dtype=dtype)
        setattr(transform_hadamard_fast, f"_h_{device.type}", h_mat)

    # Reshape to (n_rows * (hidden // 128), 128) and multiply
    x_blocks = x2d.view(-1, block_size)
    out_blocks = torch.matmul(x_blocks, h_mat)
    out = out_blocks.view(n_rows, hidden)

    if suh is not None and suh.numel() == hidden:
        out = out * suh.view(1, hidden).to(dtype=dtype)

    return out.view_as(x)
