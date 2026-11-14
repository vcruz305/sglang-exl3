"""Route EXL3 checkpoint sidecars onto per-shard linear parameters.

Dense EXL3 linears register ``exl3_{i}_{suffix}`` parameters, one set per output shard
(see ``dense.EXL3LinearMethod.create_weights``). A checkpoint stores one group per module
(``<module>.trellis``, ``.suh``, ``.svh``, ``.mcg``, ``.mul1``). Model loaders that use
SGLang's stacked-parameter mapping do not know these names, so this module maps them.

Cases handled:
- ``q_a_proj`` / ``kv_a_proj_with_mqa`` -> ``fused_qkv_a_proj_with_mqa`` shards 0 / 1
- ``gate_proj`` / ``up_proj``           -> ``gate_up_proj`` shards 0 / 1
- ``qkv_proj`` stored as ONE fused group (q|k|v along the output axis) -> split along output
  tiles into the QKV shards, using the module's ``_exl3_lin_output_partition_sizes``
- any other EXL3 module -> shard 0

Unknown modules return False so the caller's post-load check fails closed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

SIDECAR_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
TILE = 16

# checkpoint leaf -> (merged module leaf, shard index)
_MERGED_LEAVES = {
    "q_a_proj": ("fused_qkv_a_proj_with_mqa", 0),
    "kv_a_proj_with_mqa": ("fused_qkv_a_proj_with_mqa", 1),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


def split_sidecar_name(name: str) -> Optional[Tuple[str, str]]:
    """``'a.b.trellis' -> ('a.b', 'trellis')``; None when not a sidecar name."""
    for suffix in SIDECAR_SUFFIXES:
        end = "." + suffix
        if name.endswith(end):
            return name[: -len(end)], suffix
    return None


def _n_shards(params: Dict[str, Any], base: str) -> int:
    n = 0
    while f"{base}.exl3_{n}_trellis" in params:
        n += 1
    return n


def split_fused_group(
    owner_sizes: List[int], suffix: str, tensor: torch.Tensor
) -> List[torch.Tensor]:
    """Split one fused EXL3 group along the output axis into per-shard groups.

    ``trellis`` is [in_tiles, out_tiles, k_words]; ``svh`` is [out]; ``suh`` (input axis) and the
    scalar ``mcg``/``mul1`` markers are shared by every shard. Sizes must be multiples of 16.
    """
    if any(s % TILE for s in owner_sizes):
        raise ValueError(f"EXL3 fused split: output sizes {owner_sizes} are not multiples of {TILE}")
    out_total = sum(owner_sizes)
    if suffix == "trellis":
        if tensor.dim() != 3 or tensor.shape[1] != out_total // TILE:
            raise ValueError(
                f"EXL3 fused trellis shape {tuple(tensor.shape)} does not match out={out_total}"
            )
        parts, o = [], 0
        for s in owner_sizes:
            parts.append(tensor[:, o // TILE : (o + s) // TILE, :])
            o += s
        return parts
    if suffix == "svh":
        if tensor.shape[0] != out_total:
            raise ValueError(f"EXL3 fused svh shape {tuple(tensor.shape)} does not match out={out_total}")
        parts, o = [], 0
        for s in owner_sizes:
            parts.append(tensor[o : o + s])
            o += s
        return parts
    return [tensor for _ in owner_sizes]


def route_exl3_sidecar(params: Dict[str, Any], name: str, tensor: torch.Tensor) -> bool:
    """Load one checkpoint tensor if it is an EXL3 sidecar of a dense EXL3 linear.

    Returns True when the tensor was consumed (the caller must skip the name), False otherwise.
    MoE expert sidecars are excluded: they go through the expert mapping.
    """
    parsed = split_sidecar_name(name)
    if parsed is None or ".experts." in name or name in params:
        return False
    base, suffix = parsed
    leaf = base.rsplit(".", 1)[-1]
    parent = base[: len(base) - len(leaf)]

    target, shard = base, 0
    if leaf in _MERGED_LEAVES:
        merged_leaf, shard = _MERGED_LEAVES[leaf]
        target = parent + merged_leaf

    n = _n_shards(params, target)
    if n == 0:
        return False  # not an EXL3 linear here; caller's post-load check fails closed

    if leaf == "qkv_proj" and n > 1:
        owner = params[f"{target}.exl3_0_{suffix}"]._exl3_owner
        sizes = list(owner._exl3_lin_output_partition_sizes)
        if len(sizes) != n:
            raise RuntimeError(f"EXL3 fused qkv {target}: {n} shards but sizes {sizes}")
        for i, part in enumerate(split_fused_group(sizes, suffix, tensor)):
            p = params[f"{target}.exl3_{i}_{suffix}"]
            p.weight_loader(p, part, shard_id=i)
        return True

    if shard >= n:
        raise RuntimeError(f"EXL3 sidecar {name}: shard {shard} >= {n} shards of {target}")
    p = params[f"{target}.exl3_{shard}_{suffix}"]
    p.weight_loader(p, tensor, shard_id=shard)
    return True
