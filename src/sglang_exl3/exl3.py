"""Portions of this module derive from Mia's AI Lab, overlay/exl3.py in
GLM-5.3-Flash-EXL3-2x-DGX-Sparks, first published 2026-08-27, which precedes this
project. The routed-expert EXL3/MCG path, its pointer-table construction, expert-map
pinning, and diagnostic strings originate there.

Copyright (c) 2026 Mia's AI Lab. MIT. See THIRD_PARTY_NOTICES.md.

The EXL3 trellis format, the MCG codebook, and the quantization method are ExLlamaV3's
work, Copyright (c) 2025 Turboderp, MIT. See THIRD_PARTY_NOTICES.md.
"""

# SPDX-License-Identifier: Apache-2.0
# Native sm_121 Blackwell EXL3/MCG trellis quantization for SGLang.
#
# Checkpoint ABI used by this pack:
#   quant_method=exl3, codebook=mcg
#   per expert matrix: trellis (int16) + suh/svh (fp16) + mcg (int32 marker)
#
# Non-routed tensors stay native (UnquantizedLinearMethod). Experts never
# expand to a persistent BF16 weight; LinearEXL3 / sglang_exl3_c / exllamav3_ext runs the
# trellis GEMM.

from __future__ import annotations

import importlib
import logging
import math
import os
import re
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

logger = logging.getLogger("sglang.exl3")

# SGLang base classes with robust local fallbacks
try:
    from sglang.srt.layers.quantization.base_config import (
        FusedMoEMethodBase,
        LinearMethodBase,
        QuantizationConfig,
        QuantizeMethodBase,
    )
except ImportError:
    class QuantizeMethodBase:
        def create_weights(self, layer: torch.nn.Module, *args, **kwargs):
            raise NotImplementedError()
        def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError()
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            pass

    class LinearMethodBase(QuantizeMethodBase):
        pass

    class FusedMoEMethodBase(QuantizeMethodBase):
        runner = None
        def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config: Any):
            self.runner = None

    class QuantizationConfig:
        def __init__(self):
            self.packed_modules_mapping = {}
        def update_packed_modules_mapping(self, mapping):
            self.packed_modules_mapping = mapping

try:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
except ImportError:
    StandardCombineInput = None
    StandardDispatchOutput = None

try:
    from sglang.srt.layers.linear import LinearBase
except ImportError:
    LinearBase = None

try:
    from sglang.srt.layers.radix_attention import RadixAttention
except ImportError:
    RadixAttention = None



MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
MUL1_MULTIPLIER = 0x83DCD12D
MUL1_MARKER_SIGNED_INT32 = -2082680531
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
SWIGLU_LIMIT_DEFAULT = float("inf")  # no model-requested clip
TEMP_ROWS_FUSED = 2048
MOE_ACT_SILU = 0

_FUSED_TEMP_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

DEFAULT_SPECULATIVE_SCHEDULE: list[tuple[int, int, int]] = [
    (1, 4, 3),
    (5, 8, 2),
    (9, 16, 1),
]
SPECULATIVE_SCHEDULE_ENV = "SGLANG_EXL3_SPEC_SCHEDULE"
ADAPTIVE_VERIFICATION_ENV = "SGLANG_EXL3_ADAPTIVE_VERIFICATION"


def compute_mla_kv_cache_bytes(
    context_len: int,
    num_layers: int = 43,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    dtype_bytes: int = 1,
) -> int:
    """Return the FP8 MLA KV-cache footprint for ``context_len`` tokens."""
    values = {
        "context_len": context_len,
        "num_layers": num_layers,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "dtype_bytes": dtype_bytes,
    }
    normalized: dict[str, int] = {}
    for name, value in values.items():
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be an integer") from exc
        if integer != value:
            raise ValueError(f"{name} must be an integer")
        if integer < 0:
            raise ValueError(f"{name} must be non-negative")
        normalized[name] = integer

    return (
        normalized["context_len"]
        * normalized["num_layers"]
        * (normalized["kv_lora_rank"] + normalized["qk_rope_head_dim"])
        * normalized["dtype_bytes"]
    )


def _env_float_override(default: float, *names: str, minimum: float | None = None) -> float:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip())
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(value) or (minimum is not None and value < minimum):
            continue
        return value
    return default


def _env_int_override(default: int, *names: str, minimum: int | None = None) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw.strip(), 10)
        except (AttributeError, TypeError, ValueError):
            continue
        if minimum is not None and value < minimum:
            continue
        return value
    return default


def validate_context_scaling(
    max_model_len: int,
    model_weights_gb: float = 95.4,
    total_mem_gb: float = 128.0,
    mem_util: float = 0.90,
    chunk_size: int = 2048,
) -> dict[str, float | int | bool]:
    """Validate an MLA context ceiling against a unified-memory budget."""
    if isinstance(max_model_len, bool):
        raise ValueError("max_model_len must be positive")
    try:
        max_model_len = int(max_model_len)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_model_len must be positive") from exc
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")

    model_weights_gb = _env_float_override(
        float(model_weights_gb),
        "SGLANG_EXL3_CONTEXT_MODEL_WEIGHTS_GB",
        "SGLANG_EXL3_MODEL_WEIGHTS_GB",
        "VLLM_EXL3_CONTEXT_MODEL_WEIGHTS_GB",
        "VLLM_EXL3_MODEL_WEIGHTS_GB",
        minimum=0.0,
    )
    total_mem_gb = _env_float_override(
        float(total_mem_gb),
        "SGLANG_EXL3_CONTEXT_TOTAL_MEM_GB",
        "SGLANG_EXL3_TOTAL_MEM_GB",
        "VLLM_EXL3_CONTEXT_TOTAL_MEM_GB",
        "VLLM_EXL3_TOTAL_MEM_GB",
        minimum=0.0,
    )
    mem_util = _env_float_override(
        float(mem_util),
        "SGLANG_EXL3_CONTEXT_MEM_UTIL",
        "SGLANG_EXL3_MEM_UTIL",
        "VLLM_EXL3_CONTEXT_MEM_UTIL",
        "VLLM_EXL3_MEM_UTIL",
        minimum=0.0,
    )
    chunk_size = _env_int_override(
        int(chunk_size),
        "SGLANG_EXL3_CONTEXT_CHUNK_SIZE",
        "SGLANG_EXL3_CHUNK_SIZE",
        "VLLM_EXL3_CONTEXT_CHUNK_SIZE",
        "VLLM_EXL3_CHUNK_SIZE",
        minimum=1,
    )

    kv_cache_bytes = compute_mla_kv_cache_bytes(max_model_len)
    kv_cache_gb = kv_cache_bytes / (1024**3)
    usable_mem_gb = total_mem_gb * mem_util
    available_headroom_gb = usable_mem_gb - model_weights_gb - kv_cache_gb
    safety_margin_gb = total_mem_gb - model_weights_gb - kv_cache_gb
    return {
        "max_model_len": max_model_len,
        "kv_cache_bytes": kv_cache_bytes,
        "kv_cache_gb": kv_cache_gb,
        "usable_mem_gb": usable_mem_gb,
        "available_headroom_gb": available_headroom_gb,
        "fits": available_headroom_gb > 0.0,
        "safety_margin_gb": safety_margin_gb,
        "chunk_size": chunk_size,
    }


def _validated_speculative_schedule(schedule: object) -> list[tuple[int, int, int]] | None:
    if not isinstance(schedule, (list, tuple)) or not schedule:
        return None

    normalized: list[tuple[int, int, int]] = []
    for entry in schedule:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            return None
        values: list[int] = []
        for value in entry:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                values.append(value)
                continue
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                values.append(int(value.strip(), 10))
                continue
            return None
        min_bs, max_bs, draft_tokens = values
        if min_bs < 1 or max_bs < min_bs or draft_tokens < 0:
            return None
        normalized.append((min_bs, max_bs, draft_tokens))

    normalized.sort(key=lambda item: (item[0], item[1], item[2]))
    for previous, current in zip(normalized, normalized[1:]):
        if current[0] <= previous[1]:
            return None
    return normalized


def parse_speculative_schedule(schedule_str: str) -> list[tuple[int, int, int]]:
    if not isinstance(schedule_str, str) or not schedule_str.strip():
        return list(DEFAULT_SPECULATIVE_SCHEDULE)

    entries: list[list[int]] = []
    try:
        for raw_entry in schedule_str.split(","):
            fields = [field.strip() for field in raw_entry.split(":")]
            if len(fields) != 3 or any(not field for field in fields):
                raise ValueError("each schedule entry must contain three integers")
            entries.append([int(field, 10) for field in fields])
    except (TypeError, ValueError, OverflowError):
        return list(DEFAULT_SPECULATIVE_SCHEDULE)

    normalized = _validated_speculative_schedule(entries)
    return normalized if normalized is not None else list(DEFAULT_SPECULATIVE_SCHEDULE)


def get_speculative_draft_tokens(batch_size: int, schedule: list | None = None) -> int:
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError, OverflowError):
        return 0
    if batch_size <= 0:
        return 0

    if schedule is None:
        configured = os.environ.get(SPECULATIVE_SCHEDULE_ENV) or os.environ.get("VLLM_EXL3_SPEC_SCHEDULE")
        active_schedule = (
            parse_speculative_schedule(configured)
            if configured is not None
            else list(DEFAULT_SPECULATIVE_SCHEDULE)
        )
    elif isinstance(schedule, str):
        active_schedule = parse_speculative_schedule(schedule)
    else:
        active_schedule = _validated_speculative_schedule(schedule)
        if active_schedule is None:
            active_schedule = list(DEFAULT_SPECULATIVE_SCHEDULE)

    for min_bs, max_bs, draft_tokens in active_schedule:
        if min_bs <= batch_size <= max_bs:
            return draft_tokens
    return 0


def is_adaptive_verification_enabled() -> bool:
    value = os.environ.get(ADAPTIVE_VERIFICATION_ENV, "") or os.environ.get("VLLM_EXL3_ADAPTIVE_VERIFICATION", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def filter_speculative_candidates(probs: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor | int]:
    if not isinstance(probs, torch.Tensor):
        raise TypeError(f"probs must be a torch.Tensor, got {type(probs).__name__}")
    if probs.ndim == 0:
        raise ValueError("probs must have a candidate dimension (at least 1D)")

    num_candidates = probs.shape[-1]
    if num_candidates == 0:
        mask = torch.zeros_like(probs, dtype=torch.bool)
        if probs.ndim == 1:
            return mask, 0
        return mask, torch.zeros(probs.shape[:-1], dtype=torch.long, device=probs.device)

    confident = torch.ge(probs, threshold)
    mask = torch.cumprod(confident.to(dtype=torch.int64), dim=-1).to(dtype=torch.bool)
    kept_counts = mask.sum(dim=-1, dtype=torch.long)
    if probs.ndim == 1:
        return mask, int(kept_counts.item())
    return mask, kept_counts


def _narrow_tp(tensor: torch.Tensor, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    size = int(tensor.shape[dim])
    if size % tp_size:
        raise ValueError(
            f"EXL3 TP shard: dim {dim} size {size} is not divisible by tp={tp_size}"
        )
    chunk = size // tp_size
    return tensor.narrow(dim, chunk * tp_rank, chunk).contiguous()


def shard_exl3_col(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    if suffix == "trellis":
        return _narrow_tp(loaded, 1, tp_rank, tp_size)
    if suffix == "svh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def shard_exl3_row(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    if suffix == "trellis":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    if suffix == "suh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def set_weight_attrs(param: Parameter, extra_attrs: dict[str, Any]) -> None:
    try:
        from sglang.srt.utils import set_weight_attrs as _sgl_set_weight_attrs
        _sgl_set_weight_attrs(param, extra_attrs)
    except Exception:
        for k, v in extra_attrs.items():
            setattr(param, k, v)


def _load_native_exl3_ext():
    for name in ("sglang_exl3_c", "vllm_exl3_c"):
        try:
            mod = importlib.import_module(name)
            if callable(getattr(mod, "p2b_fused_moe", None)):
                return mod
        except Exception:
            continue
    return None


def native_moe_kernel_available() -> bool:
    return _load_native_exl3_ext() is not None


def load_exllamav3_ext():
    import exllamav3_ext
    return exllamav3_ext


def _exllamav3_moe_available() -> bool:
    try:
        return callable(getattr(load_exllamav3_ext(), "exl3_moe", None))
    except Exception:
        return False


def get_moe_kernel_backend() -> str:
    requested = os.environ.get("SGLANG_EXL3_MOE_KERNEL", os.environ.get("VLLM_EXL3_MOE_KERNEL", "auto")).strip().lower()
    if requested not in {"native", "exllamav3", "auto"}:
        logger.warning("Unknown MOE_KERNEL backend=%r; using auto selection", requested)
        requested = "auto"
    if requested != "auto":
        return requested
    if native_moe_kernel_available():
        return "native"
    if _exllamav3_moe_available():
        return "exllamav3"
    return "loop"


def fused_moe_enabled() -> bool:
    return os.environ.get("EXL3_FUSED_MOE", "1") != "0"


_EXL3_MOE_NARGS: int | None = None


def _exl3_moe_nargs(fn) -> int:
    global _EXL3_MOE_NARGS
    if _EXL3_MOE_NARGS is None:
        import re
        doc = getattr(fn, "__doc__", "") or ""
        argn = [int(x) for x in re.findall(r"arg(\d+)", doc)]
        _EXL3_MOE_NARGS = max(argn) if argn else 28
    return _EXL3_MOE_NARGS


def _install_exllamav3_namespace() -> None:
    import exllamav3_ext  # noqa: F401
    importlib.import_module("exllamav3.modules.quant.exl3")


def load_linear_exl3_cls():
    _install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3


def make_linear_exl3(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor | None = None,
    mul1: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.float16,
):
    def _align(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        t = t.contiguous()
        if t.data_ptr() % 256 != 0:
            t = t.clone()
        return t

    cls = load_linear_exl3_cls()
    return cls(
        config=None,
        in_features=int(suh.numel()),
        out_features=int(svh.numel()),
        trellis=_align(trellis),
        suh=_align(suh),
        svh=_align(svh),
        mcg=_align(mcg),
        mul1=_align(mul1),
        out_dtype=out_dtype,
        transformers_fix=True,
    )


def pin_exl3_expert_map(layer: torch.nn.Module, device: torch.device) -> torch.Tensor | None:
    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    if emap.device != device or emap.dtype != torch.long:
        layer.expert_map = emap.to(device=device, dtype=torch.long)
    return layer.expert_map


def map_topk_to_local(ids: torch.Tensor, n_local: int, expert_map: torch.Tensor | None) -> torch.Tensor:
    original_shape = ids.shape
    flat = ids.reshape(-1)
    if expert_map is None:
        invalid = (flat < 0) | (flat >= n_local)
        return torch.where(invalid, flat.new_full(flat.shape, n_local), flat).reshape(original_shape)
    if expert_map.device != flat.device or expert_map.dtype != torch.long:
        raise RuntimeError(
            "EXL3 expert_map is not pinned to the hidden-state device; "
            "call pin_exl3_expert_map before fused apply"
        )
    n_global = int(expert_map.numel())
    safe = flat.clamp(min=0, max=max(n_global - 1, 0))
    mapped = expert_map[safe] if n_global else flat.new_full(flat.shape, n_local)
    invalid = (flat < 0) | (flat >= n_global) | (mapped < 0) | (mapped >= n_local)
    return torch.where(invalid, flat.new_full(flat.shape, n_local), mapped).reshape(original_shape)


def apply_exl3_python_loop(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
    *,
    only_experts: set[int] | None = None,
    out: torch.Tensor | None = None,
    layer: torch.nn.Module | None = None,
) -> torch.Tensor:
    tokens, hidden = x2d.shape
    if (
        os.environ.get("EXL3_FUSED_MOE", "1") != "0"
        and tokens <= 16
        and layer is not None
        and only_experts is None
        and out is None
    ):
        batched_out = apply_exl3_batched_moe_gemv(x2d, ids, weights, layer, inners, expert_map, limit)
        if batched_out is not None:
            return batched_out
    if out is None:
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    unique = torch.unique(ids)
    for raw in unique.tolist():
        e_raw = int(raw)
        if e_raw < 0:
            continue
        e = e_raw
        if expert_map is not None:
            mapped = int(expert_map[e].item()) if expert_map.numel() > e else e
            if mapped < 0:
                continue
            e = mapped
        if e >= len(inners):
            continue
        if only_experts is not None and e not in only_experts:
            continue
        token_idx, k_pos = (ids == int(raw)).nonzero(as_tuple=True)
        h = x2d.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        up = pack["up"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(act.contiguous().half(), {}, out_dtype=torch.float32)
        scale = weights[token_idx, k_pos].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out


def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    try:
        exllamav3_ext = load_exllamav3_ext()
    except Exception:
        exllamav3_ext = None

    device = layer.w13_trellis.device
    n_exp = len(inners)
    hidden = int(getattr(layer, "_exl3_hidden_size", 4096))
    intermediate = int(getattr(layer, "_exl3_intermediate_local", 2048))

    def _ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor(
            [int(getattr(pack[which], attr).data_ptr()) for pack in inners],
            dtype=torch.int64,
            device=device,
        )

    layer._exl3_ptrs = {
        "gate_trellis": _ptrs("gate", "trellis"),
        "gate_suh": _ptrs("gate", "suh"),
        "gate_svh": _ptrs("gate", "svh"),
        "up_trellis": _ptrs("up", "trellis"),
        "up_suh": _ptrs("up", "suh"),
        "up_svh": _ptrs("up", "svh"),
        "down_trellis": _ptrs("down", "trellis"),
        "down_suh": _ptrs("down", "suh"),
        "down_svh": _ptrs("down", "svh"),
    }
    idx = int(device.index) if device.index is not None else 0
    if exllamav3_ext is not None and hasattr(exllamav3_ext, "exl3_moe_max_concurrency"):
        concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(idx))
        if concurrency < 1:
            concurrency = 1
    else:
        layer._exl3_fused_temps = None
        layer._exl3_fused_concurrency = 0
        layer._exl3_k = int(getattr(layer, "_exl3_bits", 4))
        return

    key = (str(device), hidden, intermediate, concurrency)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(getattr(layer, "_exl3_bits", 4))


_MOE_DIAG_LOGS = 0
_MOE_DECODE_LOGS = 0


def _native_moe_dimensions_supported(
    x2d: torch.Tensor, layer: torch.nn.Module, inners: list[dict[str, Any]]
) -> bool:
    global _MOE_DIAG_LOGS
    hidden_meta = int(getattr(layer, "_exl3_hidden_size", x2d.shape[1]))
    inter_meta = int(getattr(layer, "_exl3_intermediate_local", 2048))
    k_bits = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 4)))
    native_max_m = int(os.environ.get("SGLANG_EXL3_NATIVE_MOE_MAX_M", "16"))
    valid = (
        x2d.is_cuda
        and x2d.dim() == 2
        and 1 <= int(x2d.shape[0]) <= native_max_m
        and int(x2d.shape[1]) == hidden_meta == 4096
        and inter_meta == 2048
        and k_bits in (2, 3, 4)
        and len(inners) > 0
    )
    if not valid and _MOE_DIAG_LOGS < 10:
        reasons = []
        if not x2d.is_cuda:
            reasons.append("not_cuda")
        if x2d.dim() != 2:
            reasons.append(f"dim={x2d.dim()}")
        if not (1 <= int(x2d.shape[0]) <= native_max_m):
            reasons.append(f"bs={x2d.shape[0]}>max_m={native_max_m}")
        if int(x2d.shape[1]) != 4096:
            reasons.append(f"cols={x2d.shape[1]}!=4096")
        if hidden_meta != 4096:
            reasons.append(f"hidden_meta={hidden_meta}!=4096")
        if inter_meta != 2048:
            reasons.append(f"inter_meta={inter_meta}!=2048")
        if k_bits not in (2, 3, 4):
            reasons.append(f"k_bits={k_bits} not in (2,3,4)")
        if len(inners) <= 0:
            reasons.append("no_inners")
        import sys
        sys.stderr.write(f"[EXL3-MOE-UNSUPPORTED] Reason: {', '.join(reasons)}\n")
        sys.stderr.flush()
    return valid


def _apply_native_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float | None = None,
) -> torch.Tensor | None:
    """Run the native cooperative kernel for decode rows when it is safe.

    The native ABI consumes one input row and one routing list per launch.  A
    decode batch is therefore submitted as row views into one preallocated
    output tensor.  No per-token pointer/weight tensors are allocated; the only
    conversion is one contiguous int32 routing table for the complete batch.
    Invalid/non-local IDs are clamped to a valid pointer and receive zero
    routing weight, preventing an out-of-bounds read while preserving fallback
    semantics.
    """
    module = _load_native_exl3_ext()
    if module is None or not _native_moe_dimensions_supported(x2d, layer, inners):
        return None
    intermediate = int(getattr(layer, "_exl3_intermediate_local", 2048))
    clamp_limit = float(limit) if limit is not None else 0.0
    extended_abi = getattr(module, "P2B_MOE_ABI_VERSION", 1) >= 2
    if not extended_abi:
        raise RuntimeError("Native SGLang port requires MoE ABI 2; refusing a stale extension")
    ptrs = getattr(layer, "_exl3_ptrs", None)
    if not isinstance(ptrs, dict):
        return None
    required = (
        "gate_trellis",
        "gate_suh",
        "gate_svh",
        "up_trellis",
        "up_suh",
        "up_svh",
        "down_trellis",
        "down_suh",
        "down_svh",
    )
    if any(key not in ptrs for key in required):
        return None

    n_exp = len(inners)
    local = map_topk_to_local(ids, n_exp, expert_map).reshape(ids.shape)
    topk = int(local.shape[-1])
    if topk < 1:
        return None
    # p2b_fused_moe reads int32 IDs and fp16 routing weights.  Clamp before
    # conversion so the invalid sentinel cannot wrap into a large int32 value.
    safe_ids = local.clamp(min=0, max=n_exp - 1).to(dtype=torch.int32).contiguous()
    valid = (local >= 0) & (local < n_exp)
    safe_weights = (
        weights.reshape_as(local)
        .to(dtype=torch.float16)
        .mul(valid.to(dtype=torch.float16))
        .contiguous()
    )
    xh = x2d.to(dtype=torch.float16).contiguous()
    native_out = torch.empty_like(xh)
    k = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 4)))
    fn = module.p2b_fused_moe
    extra_args = (intermediate, clamp_limit) if extended_abi else ()
    for row in range(int(x2d.shape[0])):
        result = fn(
            xh[row : row + 1],
            native_out[row : row + 1],
            ptrs["gate_trellis"],
            ptrs["gate_suh"],
            ptrs["gate_svh"],
            ptrs["up_trellis"],
            ptrs["up_suh"],
            ptrs["up_svh"],
            ptrs["down_trellis"],
            ptrs["down_suh"],
            ptrs["down_svh"],
            safe_ids[row],
            safe_weights[row],
            k,
            k,
            k,
            True,
            *extra_args,
        )
        # pybind returns the same output tensor, while lightweight test doubles
        # may return a fresh tensor.  Accommodate both without synchronizing.
        if isinstance(result, torch.Tensor) and result is not native_out:
            native_out[row : row + 1].copy_(result.reshape(1, -1))
    return native_out.to(dtype=torch.float32)


def apply_exl3_batched_moe_gemv(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
) -> torch.Tensor | None:
    """Accelerated single-token MoE execution using batched GEMV (p2b_gemv_batched).
    
    Replaces individual expert linear forward passes with 3 batched GPU kernels.
    """
    ptrs = getattr(layer, "_exl3_ptrs", None)
    if not isinstance(ptrs, dict):
        return None

    module = _load_native_exl3_ext()
    gemv_fn = getattr(module, "p2b_gemv_batched", None)
    if gemv_fn is None:
        gemv_op = getattr(getattr(torch.ops, "sglang_exl3", None), "p2b_gemv_batched", None)
        if gemv_op is not None:
            gemv_fn = gemv_op
    if gemv_fn is None:
        return None

    n_exp = len(inners)
    local = map_topk_to_local(ids, n_exp, expert_map)
    topk = int(local.shape[-1])
    m_tokens = int(x2d.shape[0])
    if topk < 1 or m_tokens < 1 or m_tokens > 16:
        return None

    xh = x2d.to(dtype=torch.float16).contiguous()
    k = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 2)))

    try:
        if m_tokens == 1:
            safe_ids = local[0].clamp(min=0, max=n_exp - 1).to(dtype=torch.int32).contiguous()
            valid = (local[0] >= 0) & (local[0] < n_exp)
            safe_weights = (
                weights[0]
                .to(dtype=torch.float32)
                .mul(valid.to(dtype=torch.float32))
                .contiguous()
            )

            # 1. Batched Gate: (1, 4096) -> (topk, 1, 2048)
            gate_out = gemv_fn(
                xh,
                ptrs["gate_trellis"],
                ptrs["gate_suh"],
                ptrs["gate_svh"],
                safe_ids,
                k,
                True,
                2048,
            )

            # 2. Batched Up: (1, 4096) -> (topk, 1, 2048)
            up_out = gemv_fn(
                xh,
                ptrs["up_trellis"],
                ptrs["up_suh"],
                ptrs["up_svh"],
                safe_ids,
                k,
                True,
                2048,
            )

            # 3. SwiGLU Activation: silu(clamp(gate)) * clamp(up)
            gate_f = gate_out.to(dtype=torch.float32)
            up_f = up_out.to(dtype=torch.float32)
            act_f = F.silu(gate_f.clamp(max=limit)) * up_f.clamp(min=-limit, max=limit)
            act_h = act_f.to(dtype=torch.float16).contiguous()

            # 4. Batched Down: (topk, 1, 2048) -> (topk, 1, 4096)
            down_out = gemv_fn(
                act_h,
                ptrs["down_trellis"],
                ptrs["down_suh"],
                ptrs["down_svh"],
                safe_ids,
                k,
                True,
                4096,
            )

            # 5. Routing scale and sum reduction
            down_f = down_out.to(dtype=torch.float32)
            scaled = down_f * safe_weights.view(topk, 1, 1)
            return scaled.sum(dim=0)
        else:
            # Multi-token speculative verification (M in [2, 16])
            hidden = int(x2d.shape[1])
            out = torch.empty((m_tokens, hidden), dtype=torch.float32, device=x2d.device)
            for t in range(m_tokens):
                safe_ids_t = local[t].clamp(min=0, max=n_exp - 1).to(dtype=torch.int32).contiguous()
                valid_t = (local[t] >= 0) & (local[t] < n_exp)
                safe_weights_t = (
                    weights[t]
                    .to(dtype=torch.float32)
                    .mul(valid_t.to(dtype=torch.float32))
                    .contiguous()
                )
                xt = xh[t:t + 1]

                gate_out = gemv_fn(
                    xt,
                    ptrs["gate_trellis"],
                    ptrs["gate_suh"],
                    ptrs["gate_svh"],
                    safe_ids_t,
                    k,
                    True,
                    2048,
                )
                up_out = gemv_fn(
                    xt,
                    ptrs["up_trellis"],
                    ptrs["up_suh"],
                    ptrs["up_svh"],
                    safe_ids_t,
                    k,
                    True,
                    2048,
                )

                gate_f = gate_out.to(dtype=torch.float32)
                up_f = up_out.to(dtype=torch.float32)
                act_f = F.silu(gate_f.clamp(max=limit)) * up_f.clamp(min=-limit, max=limit)
                act_h = act_f.to(dtype=torch.float16).contiguous()

                down_out = gemv_fn(
                    act_h,
                    ptrs["down_trellis"],
                    ptrs["down_suh"],
                    ptrs["down_svh"],
                    safe_ids_t,
                    k,
                    True,
                    hidden,
                )
                down_f = down_out.to(dtype=torch.float32)
                scaled = down_f * safe_weights_t.view(topk, 1, 1)
                out[t] = scaled.sum(dim=0).squeeze(0)
            return out
    except Exception as e:
        logger.debug("p2b_gemv_batched failed, falling back: %s", e)
        return None



_FAT_SCRATCH_CACHE: dict[tuple[str, int, int, int], dict[str, torch.Tensor]] = {}


def _fat_scratch(device: torch.device, capacity: int, gate: Any) -> dict[str, torch.Tensor]:
    hidden = int(getattr(gate, "in_features", 4096))
    intermediate = int(getattr(gate, "out_features", 2048))
    key = (str(device), capacity, intermediate, hidden)
    scratch = _FAT_SCRATCH_CACHE.get(key)
    if scratch is not None:
        return scratch

    in_tiles, out_tiles, k_words = map(int, gate.trellis.shape)
    scratch = {
        "packed13": torch.empty((in_tiles, 2 * out_tiles, k_words), dtype=torch.int16, device=device),
        "svh13": torch.empty(2 * intermediate, dtype=torch.float16, device=device),
        "w13": torch.empty((hidden, 2 * intermediate), dtype=torch.float16, device=device),
        "w2": torch.empty((intermediate, hidden), dtype=torch.float16, device=device),
        "h": torch.empty((capacity, hidden), dtype=torch.float16, device=device),
        "h13": torch.empty((capacity, hidden), dtype=torch.float16, device=device),
        "gate_up": torch.empty((capacity, 2 * intermediate), dtype=torch.float32, device=device),
        "act": torch.empty((capacity, intermediate), dtype=torch.float32, device=device),
        "act_h": torch.empty((capacity, intermediate), dtype=torch.float16, device=device),
        "h2": torch.empty((capacity, intermediate), dtype=torch.float16, device=device),
        "down": torch.empty((capacity, hidden), dtype=torch.float32, device=device),
    }
    _FAT_SCRATCH_CACHE[key] = scratch
    return scratch


def _fat_kernel_available() -> bool:
    native_c = _load_native_exl3_ext()
    if native_c and hasattr(native_c, "exl3_fat_gemm"):
        return True
    try:
        ext = load_exllamav3_ext()
        return hasattr(ext, "exl3_fat_gemm")
    except Exception:
        return False


def apply_exl3_batched_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float,
    cap: int,
    out: torch.Tensor,
    use_kernel: bool = True,
) -> torch.Tensor:
    native_c = _load_native_exl3_ext()
    ext = native_c if (native_c and hasattr(native_c, "exl3_fat_gemm")) else load_exllamav3_ext()
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue

        token_idx = token_sorted[start:offset]
        gate = inners[e]["gate"]
        up = inners[e]["up"]
        down = inners[e]["down"]
        scratch = _fat_scratch(xh.device, n_rows, gate)
        intermediate = int(gate.out_features)

        h = scratch["h"][:n_rows]
        h13 = scratch["h13"][:n_rows]
        torch.index_select(xh, 0, token_idx, out=h)
        ext.had_r_128(h, h13, gate.suh, None, 1.0)

        packed13 = scratch["packed13"]
        out_tiles = int(gate.trellis.shape[1])
        packed13[:, :out_tiles].copy_(gate.trellis)
        packed13[:, out_tiles:].copy_(up.trellis)
        gate_up = scratch["gate_up"][:n_rows]
        svh13 = scratch["svh13"]
        svh13[:intermediate].copy_(gate.svh)
        svh13[intermediate:].copy_(up.svh)

        k = int(getattr(gate, "K", 4))
        mcg = bool(getattr(gate, "mcg", True))
        mul1 = bool(getattr(gate, "mul1", False))

        if use_kernel and k == 4 and hasattr(ext, "exl3_fat_gemm"):
            ext.exl3_fat_gemm(h13, packed13, gate_up, svh13, k, mcg, mul1)
        else:
            w13 = scratch["w13"]
            ext.reconstruct(w13, packed13, k, mcg, mul1)
            ext.hgemm(h13, w13, gate_up)
            ext.had_r_128(gate_up, gate_up, None, svh13, 1.0)

        gate_out = gate_up[:, :intermediate]
        up_out = gate_up[:, intermediate:]
        gate_out.clamp_(max=limit)
        up_out.clamp_(min=-limit, max=limit)
        act = scratch["act"][:n_rows]
        torch.sigmoid(gate_out, out=act)
        act.mul_(gate_out).mul_(up_out)
        act_h = scratch["act_h"][:n_rows]
        act_h.copy_(act)

        h2 = scratch["h2"][:n_rows]
        ext.had_r_128(act_h, h2, down.suh, None, 1.0)
        if use_kernel and k == 4 and hasattr(ext, "exl3_fat_gemm_scatter"):
            ext.exl3_fat_gemm_scatter(
                h2, down.trellis, out, down.svh, token_idx, weight_sorted[start:offset], k, mcg, mul1
            )
        else:
            w2 = scratch["w2"]
            ext.reconstruct(w2, down.trellis, k, mcg, mul1)
            down_out = scratch["down"][:n_rows]
            ext.hgemm(h2, w2, down_out)
            ext.had_r_128(down_out, down_out, None, down.svh, 1.0)
            down_out.mul_(weight_sorted[start:offset].unsqueeze(-1))
            out.index_add_(0, token_idx, down_out)
    return out


def apply_exl3_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
) -> torch.Tensor:
    tokens, hidden = x2d.shape
    n_exp = len(inners)

    native_max_m = int(os.environ.get("SGLANG_EXL3_NATIVE_MOE_MAX_M", "16"))
    if get_moe_kernel_backend() == "native" and 1 <= tokens <= native_max_m:
        try:
            native_out = _apply_native_fused_moe(x2d, ids, weights, layer, inners, expert_map, None if limit == float("inf") else limit)
            if native_out is not None:
                layer._exl3_last_apply = "native"
                return native_out
        except Exception as exc:
            logger.debug("Native EXL3 MoE dispatch failed in fused entry: %s", exc)

    try:
        import exllamav3_ext
    except Exception:
        return apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit, layer=layer)

    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    local = map_topk_to_local(ids, n_exp, expert_map)
    local_flat = local.reshape(-1)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(dtype=torch.float16)
    order = local_flat.argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)
    expert_count.scatter_add_(0, local_flat.long(), torch.ones(local_flat.shape, dtype=torch.long, device=local.device))
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    xh = x2d.contiguous().half()

    counts = expert_count[:n_exp]
    if tokens > TEMP_ROWS_FUSED and bool((counts > TEMP_ROWS_FUSED).any().item()):
        for s in range(0, tokens, TEMP_ROWS_FUSED):
            e = min(s + TEMP_ROWS_FUSED, tokens)
            out[s:e] = apply_exl3_fused_moe(x2d[s:e], ids[s:e], weights[s:e], layer, inners, expert_map, limit)
        return out

    fn = exllamav3_ext.exl3_moe
    k = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 4)))
    args = (
        xh,
        out,
        expert_count,
        token_sorted,
        weight_sorted,
        temps[0],
        temps[1],
        temps[2],
        temps[3],
        MOE_ACT_SILU,
        k,
        k,
        k,
        ptrs["gate_trellis"],
        ptrs["gate_suh"],
        ptrs["gate_svh"],
        ptrs["up_trellis"],
        ptrs["up_suh"],
        ptrs["up_svh"],
        ptrs["down_trellis"],
        ptrs["down_suh"],
        ptrs["down_svh"],
        bool(getattr(layer, "_exl3_use_mcg", True)),
        bool(getattr(layer, "_exl3_use_mul1", False)),
        bool(getattr(layer, "_exl3_use_mcg", True)),
        bool(getattr(layer, "_exl3_use_mul1", False)),
        bool(getattr(layer, "_exl3_use_mcg", True)),
        bool(getattr(layer, "_exl3_use_mul1", False)),
        float(limit),
    )
    n_args = _exl3_moe_nargs(fn)
    if n_args >= 34:
        fn(*args, -1, None, None, 1, 16, 16)
    elif n_args >= 29:
        fn(*args, -1)
    else:
        fn(*args)

    if tokens > TEMP_ROWS_FUSED:
        fat = (counts > TEMP_ROWS_FUSED).nonzero(as_tuple=False).view(-1)
        if fat.numel():
            if _fat_kernel_available():
                counts_list = counts.tolist()
                apply_exl3_batched_fat(
                    xh, token_sorted, weight_sorted, counts_list, inners, limit, TEMP_ROWS_FUSED, out, use_kernel=True
                )
            else:
                apply_exl3_python_loop(
                    x2d, ids, weights, inners, expert_map, limit, only_experts=set(int(i) for i in fat.tolist()), out=out
                )
    layer._exl3_last_apply = "fused"
    return out


def apply_exl3_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: torch.nn.Module,
    *,
    limit: float = SWIGLU_LIMIT_DEFAULT,
    fused: bool | None = None,
) -> torch.Tensor:
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)

    backend = get_moe_kernel_backend()
    global _MOE_DIAG_LOGS
    if _MOE_DIAG_LOGS < 5:
        _MOE_DIAG_LOGS += 1
        h_meta = getattr(layer, "_exl3_hidden_size", None)
        i_meta = getattr(layer, "_exl3_intermediate_local", None)
        b_meta = getattr(layer, "_exl3_bits", None)
        k_meta = getattr(layer, "_exl3_k", None)
        has_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
        import sys
        sys.stderr.write(
            f"[EXL3-MOE-DIAG #{_MOE_DIAG_LOGS}] tokens={tokens} hidden={hidden} "
            f"layer_h={h_meta} layer_inter={i_meta} bits={b_meta} k={k_meta} "
            f"has_ptrs={has_ptrs} backend={backend}\n"
        )
        sys.stderr.flush()

    global _MOE_DECODE_LOGS
    if tokens <= 8 and _MOE_DECODE_LOGS < 5:
        _MOE_DECODE_LOGS += 1
        import sys
        sys.stderr.write(
            f"[EXL3-MOE-DECODE #{_MOE_DECODE_LOGS}] tokens={tokens} backend={backend} "
            f"bits={getattr(layer, '_exl3_bits', None)} ptrs={bool(getattr(layer, '_exl3_ptrs', None))}\n"
        )
        sys.stderr.flush()

    if backend == "native" and (fused is not False):
        native_max_m = int(os.environ.get("SGLANG_EXL3_NATIVE_MOE_MAX_M", "16"))
        if 1 <= tokens <= native_max_m:
            try:
                native_out = _apply_native_fused_moe(x2d, ids, weights, layer, inners, expert_map, None if limit == float("inf") else limit)
                if native_out is not None:
                    if not getattr(layer, "_exl3_native_proven", False):
                        logger.warning("EXL3-NATIVE-ABI2 EXECUTED layer=%s tokens=%s limit=%s", getattr(layer, "layer_id", "unknown"), tokens, limit)
                        layer._exl3_native_proven = True
                    layer._exl3_last_apply = "native"
                    return native_out.to(dtype=x.dtype)
            except Exception as exc:
                logger.warning("native EXL3 MoE skipped (%s); falling back", exc)

    have_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
    if fused is True and not have_ptrs:
        raise RuntimeError("EXL3 fused apply requested but pointer tables are missing")
    use_fused = (fused_moe_enabled() if fused is None else bool(fused)) and have_ptrs
    if use_fused:
        try:
            import exllamav3_ext

            use_fused = hasattr(exllamav3_ext, "exl3_moe")
        except Exception:
            use_fused = False
    if use_fused:
        try:
            out = apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit)
            layer._exl3_last_apply = "fused"
            return out.to(dtype=x.dtype)
        except TypeError as exc:
            logger.warning("exl3_moe signature mismatch (%s); using python loop", exc)
            use_fused = False
    if not use_fused:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit, layer=layer)
        layer._exl3_last_apply = "loop"
    return out.to(dtype=x.dtype)


def _suffix_from_mapped_name(weight_name: str) -> str:
    tail = weight_name.rsplit(".", 1)[-1]
    suffix = tail.rsplit("_", 1)[-1]
    if suffix not in EXL3_SUFFIXES:
        raise ValueError(
            f"EXL3 param suffix {suffix!r} (from {tail!r}) not in {EXL3_SUFFIXES}; "
            f"full mapped name was {weight_name}"
        )
    return suffix


def _prefix_has_suffix(prefix: str, suffix: str) -> bool:
    return prefix == suffix or prefix.endswith("." + suffix)


class EXL3Config(QuantizationConfig):
    """SGLang quantization config for native sm_121 Blackwell EXL3/MCG models."""

    def __init__(
        self,
        bits: int = 4,
        codebook: str = "mcg",
        scope: str = "glm53_routed_experts_only",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.codebook = str(codebook)
        self.scope = str(scope)

        raw_layer_bits = kwargs.pop("layer_bits", None) or {}
        self.layer_bits: dict[int, int] = {
            int(k): int(v) for k, v in dict(raw_layer_bits).items()
        }
        for layer_idx, layer_k in self.layer_bits.items():
            if layer_k not in (2, 3, 4, 5, 6):
                raise ValueError(f"unsupported EXL3 bits={layer_k} for layer {layer_idx}")

        raw_nr_exl3 = kwargs.pop("non_routed_exl3", None) or {}
        self.non_routed_exl3: dict[str, Any] = dict(raw_nr_exl3) if raw_nr_exl3 else {}
        nr_bits = self.non_routed_exl3.get("bits")
        if nr_bits is not None and nr_bits not in (2, 3, 4, 5, 6):
            raise ValueError(f"unsupported non_routed_exl3 bits={nr_bits}")
        nr_codebook = self.non_routed_exl3.get("codebook", "mcg")
        if nr_codebook not in ("mcg", "mul1"):
            raise ValueError(f"unsupported non_routed_exl3 codebook={nr_codebook!r}")

        self.raw_config = dict(kwargs)
        if self.codebook not in ("mcg", "mul1"):
            raise ValueError(f"unsupported codebook={self.codebook!r}; must be 'mcg' or 'mul1'")
        if self.bits not in (2, 3, 4, 5, 6):
            raise ValueError(f"unsupported EXL3 bits={self.bits}")

    def get_name(self) -> str:
        return "exl3"

    _LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

    def bits_for_prefix(self, prefix: str) -> int:
        if not self.layer_bits:
            return self.bits
        m = self._LAYER_RE.search(prefix or "")
        if m is None:
            return self.bits
        return self.layer_bits.get(int(m.group(1)), self.bits)

    def _matches_non_routed_exl3(self, prefix: str) -> bool:
        if not self.non_routed_exl3:
            return False
        layers = self.non_routed_exl3.get("layers", {})
        if layers and prefix in layers:
            return True
        modules = self.non_routed_exl3.get("modules", [])
        return any(_prefix_has_suffix(prefix, m) for m in modules)

    def _should_exl3_all_linears(self, prefix: str) -> bool:
        """Whether this dense linear is stored as EXL3 in the current pack.

        GLM-5.3 packs only routed experts as EXL3. Qwen3.8-Flash-Next-exl3
        stores GDN (linear_attn), full-attn, and shared-expert as trellis too.
        Override with SGLANG_EXL3_ALL_LINEARS=0/1.
        """
        env = os.environ.get("SGLANG_EXL3_ALL_LINEARS", "").strip().lower()
        if env in {"0", "false", "no", "off"}:
            return False
        if env in {"1", "true", "yes", "on"}:
            return True
        if self._matches_non_routed_exl3(prefix):
            return True
        # Do not treat codebook=mul1 as "every linear": Qwen packs keep
        # in_proj_a/b and conv1d as .weight (BF16) while qkv/GDN are trellis.
        pl = (prefix or "").lower()
        needles = (
            "in_proj_qkvz",
            "in_proj_qkv",
            "in_proj_z.",
            ".in_proj_z",
            "out_proj",
            "qkv_proj",
            "o_proj",
            "shared_expert",
            "gate_up_proj",
            "lm_head",
        )
        return any(n in pl for n in needles)

    def _bits_for_non_routed(self, prefix: str) -> int:
        if not self.non_routed_exl3:
            return self.bits
        layers = self.non_routed_exl3.get("layers", {})
        if layers and prefix in layers:
            layer_cfg = layers[prefix]
            if "bits" in layer_cfg:
                return int(layer_cfg["bits"])
            return int(self.non_routed_exl3.get("bits", self.bits))
        modules = self.non_routed_exl3.get("modules", [])
        for suffix in modules:
            if _prefix_has_suffix(prefix, suffix):
                layer_bits = self.non_routed_exl3.get("layer_bits", {})
                if suffix in layer_bits:
                    return int(layer_bits[suffix])
                break
        return int(self.non_routed_exl3.get("bits", self.bits))

    def _bf16_shards_for(self, prefix: str) -> list[int]:
        if not self.non_routed_exl3:
            return []
        layers = self.non_routed_exl3.get("layers", {})
        if layers and prefix in layers:
            return list(layers[prefix].get("bf16_shards", []))
        return []

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # Blackwell GB10 is sm_121 (capability 121 >= 80)
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EXL3Config":
        skip = {
            "bits",
            "codebook",
            "scope",
            "quant_method",
            "tensor_storage",
            "non_routed_exl3",
            "non_routed_quantization",
            "mtp_experts",
            "mtp_experts_start_layer",
        }
        inst = cls(
            bits=int(config.get("bits", 4)),
            codebook=str(config.get("codebook", "mcg")),
            scope=str(config.get("scope", "glm53_routed_experts_only")),
            non_routed_exl3=config.get("non_routed_exl3"),
            **{k: v for k, v in config.items() if k not in skip},
        )
        inst.non_routed_quantization = config.get("non_routed_quantization")
        inst.non_routed_dtype_policy = str(config.get("non_routed_dtype_policy", ""))
        inst.mtp_experts = str(config.get("mtp_experts", "exl3"))
        inst.mtp_experts_start_layer = config.get("mtp_experts_start_layer")
        _mb = config.get("mtp_bits", None)
        if _mb is None:
            _mb = config.get("bits", inst.bits)
        inst.mtp_bits = int(_mb) if _mb is not None else int(inst.bits)
        return inst

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        if method == "exl3":
            return "exl3"
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: Any) -> None:
        pass

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        layer_cls_name = layer.__class__.__name__
        prefix_lower = (prefix or "").lower()
        _lm = self._LAYER_RE.search(prefix or "")
        layer_num = int(_lm.group(1)) if _lm else -1
        # GLM-5.3 used layer_id>=43 as MTP. Qwen4Exp has 48 MoE layers;
        # only real MTP/NextN prefixes are draft.
        is_draft_layer = (
            "stages." in prefix_lower
            or prefix_lower.startswith("stages.")
            or "mtp" in prefix_lower
            or "nextn" in prefix_lower
        )

        is_moe_layer = (
            "FusedMoE" in layer_cls_name
            or "MoE" in layer_cls_name
            or "RoutedExperts" in layer_cls_name
            or hasattr(layer, "num_local_experts")
        )
        if is_moe_layer:
            if is_draft_layer:
                # Qwen Flash Next EXL3 ships EXL3 MTP packs (mtp_bits). Prefer native EXL3 draft.
                if os.environ.get("SGLANG_EXL3_MTP_EXL3", "1") == "1":
                    bits = getattr(self, "mtp_bits", None) or self.bits_for_prefix(prefix)
                    return EXL3MoEMethod(self, bits=int(bits))
                if os.environ.get("SGLANG_EXL3_MTP_FP8", "1") == "1":
                    d = self._draft_delegate()
                    if d is not None and hasattr(d, "get_quant_method"):
                        dm = d.get_quant_method(layer, prefix)
                        if dm is not None:
                            return dm
                    try:
                        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
                        if isinstance(d, Fp8Config):
                            return Fp8MoEMethod(d)
                    except Exception:
                        pass
                return None

            if getattr(self, "mtp_experts", "") == "source":
                _start = getattr(self, "mtp_experts_start_layer", 43)
                if _start is not None and layer_num >= int(_start):
                    d = self._non_routed_delegate()
                    if d is not None and hasattr(d, "get_quant_method"):
                        dm = d.get_quant_method(layer, prefix)
                        if dm is not None:
                            return dm
                    return None
            return EXL3MoEMethod(self, bits=self.bits_for_prefix(prefix))

        if (RadixAttention is not None and isinstance(layer, RadixAttention)) or "RadixAttention" in layer_cls_name:
            d = self._draft_delegate() if is_draft_layer else self._non_routed_delegate()
            if d is not None and hasattr(d, "get_quant_method"):
                dm = d.get_quant_method(layer, prefix)
                if dm is not None:
                    return dm
            return None

        # Token embeddings are BF16 / PLE-backed, never an EXL3 trellis.
        if "embed_tokens" in prefix_lower or layer_cls_name == "VocabParallelEmbedding":
            return None

        is_lm_head = (
            "lm_head" in prefix_lower
            or "ParallelLMHead" in layer_cls_name
        )
        is_linear = (
            is_lm_head
            or (LinearBase is not None and isinstance(layer, LinearBase))
            or "Linear" in layer_cls_name
        )
        if is_linear:
            if not is_draft_layer and self._matches_non_routed_exl3(prefix):
                bits = self._bits_for_non_routed(prefix)
                return EXL3LinearMethod(self, bits=bits)

            if is_draft_layer:
                if os.environ.get("SGLANG_EXL3_MTP_EXL3", "1") == "1":
                    if self._matches_non_routed_exl3(prefix):
                        bits = self._bits_for_non_routed(prefix)
                    else:
                        bits = getattr(self, "mtp_bits", None) or self.bits_for_prefix(prefix)
                    return EXL3LinearMethod(self, bits=int(bits))
                if os.environ.get("SGLANG_EXL3_MTP_FP8", "1") == "1":
                    d = self._draft_delegate()
                    if d is not None:
                        if hasattr(d, "get_quant_method"):
                            dm = d.get_quant_method(layer, prefix)
                            if dm is not None:
                                return dm
                        try:
                            from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
                            if isinstance(d, Fp8Config):
                                return Fp8LinearMethod(d)
                        except Exception:
                            pass
                return None

            # Qwen3.8-Flash-Next EXL3 packs GDN / attn / shared-expert as trellis,
            # not BF16. GLM-5.3 defaulted those to UnquantizedLinearMethod, which
            # leaves in_proj_qkvz.weight empty while the checkpoint has
            # in_proj_qkv.trellis — decode then emits garbage ('!').
            if not is_draft_layer and self._should_exl3_all_linears(prefix):
                bits = self.bits_for_prefix(prefix)
                if self._matches_non_routed_exl3(prefix):
                    bits = self._bits_for_non_routed(prefix)
                return EXL3LinearMethod(self, bits=int(bits))

            # Non-draft (target) linear layers
            use_fp8_delegate = os.environ.get("SGLANG_EXL3_NON_ROUTED_FP8", "0") == "1"
            if use_fp8_delegate or getattr(self, "non_routed_dtype_policy", "") not in ("bf16_as_stored", "official_source_native", ""):
                d = self._non_routed_delegate()
                if d is not None:
                    if hasattr(d, "get_quant_method"):
                        dm = d.get_quant_method(layer, prefix)
                        if dm is not None:
                            return dm
                    try:
                        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
                        if isinstance(d, Fp8Config):
                            return Fp8LinearMethod(d)
                    except Exception:
                        pass

            if getattr(self, "non_routed_dtype_policy", "") in ("bf16_as_stored", "official_source_native") and not use_fp8_delegate:
                try:
                    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
                    return UnquantizedLinearMethod()
                except Exception:
                    return None

            # Non-routed delegate or unquantized linear fallback
            d = self._non_routed_delegate()
            if d is not None:
                if hasattr(d, "get_quant_method"):
                    dm = d.get_quant_method(layer, prefix)
                    if dm is not None:
                        return dm
                try:
                    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
                    if isinstance(d, Fp8Config):
                        return Fp8LinearMethod(d)
                except Exception:
                    pass

            try:
                from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
                return UnquantizedLinearMethod()
            except Exception:
                return None

        return None

    def _draft_delegate(self):
        if not hasattr(self, "_draft_delegate_cached"):
            self._draft_delegate_cached = None
            if os.environ.get("SGLANG_EXL3_MTP_FP8", "1") == "1":
                try:
                    from sglang.srt.layers.quantization.fp8 import Fp8Config
                    self._draft_delegate_cached = Fp8Config(
                        is_checkpoint_fp8_serialized=False,
                        activation_scheme="dynamic",
                    )
                except Exception as exc:
                    logger.debug("Failed to initialize draft FP8 delegate: %s", exc)
        return self._draft_delegate_cached

    def _non_routed_delegate(self):
        if not hasattr(self, "_nr_delegate_cached"):
            self._nr_delegate_cached = None
            nrq = getattr(self, "non_routed_quantization", None)
            if not (isinstance(nrq, dict) and nrq.get("quant_method")):
                if os.environ.get("SGLANG_EXL3_NON_ROUTED_FP8", "0") == "1":
                    nrq = {
                        "quant_method": "fp8",
                        "fmt": "e4m3",
                        "weight_block_size": [128, 128],
                        "activation_scheme": "dynamic",
                    }
            if isinstance(nrq, dict) and nrq.get("quant_method"):
                qmethod = str(nrq.get("quant_method")).lower()
                try:
                    if qmethod == "fp8":
                        from sglang.srt.layers.quantization.fp8 import Fp8Config
                        is_fp8 = nrq.get("quant_method") == "fp8"
                        self._nr_delegate_cached = Fp8Config(
                            is_checkpoint_fp8_serialized=is_fp8,
                            activation_scheme=nrq.get("activation_scheme", "dynamic"),
                            ignored_layers=nrq.get("ignored_layers"),
                            weight_block_size=nrq.get("weight_block_size", [128, 128]),
                        )
                    else:
                        from sglang.srt.layers.quantization import get_quantization_config
                        cls = get_quantization_config(qmethod)
                        if hasattr(cls, "from_config"):
                            self._nr_delegate_cached = cls.from_config(dict(nrq))
                        else:
                            self._nr_delegate_cached = cls(**dict(nrq))
                except Exception as e:
                    import logging
                    logging.getLogger("sglang.exl3").warning(f"Failed to initialize non-routed delegate for {qmethod}: {e}")
        return self._nr_delegate_cached


class EXL3MoEMethod(FusedMoEMethodBase):
    """SGLang Fused MoE layer method executing native sm_121 EXL3 kernels."""

    # EXL3 consumes routing weights as supplied; scale once in TopK.
    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, quant_config: EXL3Config, bits: int | None = None) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.bits = int(bits) if bits is not None else quant_config.bits
        self.runner = None
        self._logged = False

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config: Any):
        # Native Sm_121 Blackwell kernels manage execution directly via apply()
        self.runner = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(
                "EXL3 trellis tiles are 16-wide; "
                f"hidden={hidden_size} intermediate_local={intermediate_size_per_partition}"
            )
        k_words = self.bits * 16
        in_tiles = hidden_size // 16
        out_tiles = intermediate_size_per_partition // 16

        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}

        w13_trellis = Parameter(
            torch.empty(num_experts, 2, in_tiles, out_tiles, k_words, dtype=torch.int16),
            requires_grad=False,
        )
        w13_suh = Parameter(
            torch.empty(num_experts, 2, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w13_svh = Parameter(
            torch.empty(num_experts, 2, intermediate_size_per_partition, dtype=torch.float16),
            requires_grad=False,
        )
        w13_mcg = Parameter(
            torch.zeros((num_experts, 2, 1), dtype=torch.int32),
            requires_grad=False,
        )
        w13_mul1 = Parameter(
            torch.zeros((num_experts, 2, 1), dtype=torch.int32),
            requires_grad=False,
        )

        w2_trellis = Parameter(
            torch.empty(num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16),
            requires_grad=False,
        )
        w2_suh = Parameter(
            torch.empty(num_experts, intermediate_size_per_partition, dtype=torch.float16),
            requires_grad=False,
        )
        w2_svh = Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w2_mcg = Parameter(
            torch.zeros((num_experts, 1), dtype=torch.int32),
            requires_grad=False,
        )
        w2_mul1 = Parameter(
            torch.zeros((num_experts, 1), dtype=torch.int32),
            requires_grad=False,
        )

        packed = {
            "w13_trellis": w13_trellis,
            "w13_suh": w13_suh,
            "w13_svh": w13_svh,
            "w13_mcg": w13_mcg,
            "w13_mul1": w13_mul1,
            "w2_trellis": w2_trellis,
            "w2_suh": w2_suh,
            "w2_svh": w2_svh,
            "w2_mcg": w2_mcg,
            "w2_mul1": w2_mul1,
        }
        for name, param in packed.items():
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra)
            param.weight_loader = self._load_exl3
            param._exl3_owner = layer

        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_size_per_partition
        layer._exl3_k_words = k_words
        layer._exl3_bits = self.bits

    def _load_exl3(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str = "w1",
        expert_id: int = 0,
        return_success: bool = False,
    ) -> bool | None:
        owner = getattr(param, "_exl3_owner", None)
        if owner is not None and hasattr(owner, "_map_global_expert_id_to_local_expert_id"):
            local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            expert_id = local_id

        tp_rank = 0
        tp_size = 1
        try:
            from sglang.srt.distributed import get_tp_group
            tp_group = get_tp_group()
            tp_rank = getattr(tp_group, "rank_in_group", 0)
            tp_size = getattr(tp_group, "world_size", 1)
        except Exception:
            pass

        suffix = _suffix_from_mapped_name(weight_name)
        loaded = loaded_weight.detach().contiguous()

        if shard_id in ("w1", "w3"):
            shard_idx = 0 if shard_id == "w1" else 1
            sharded = shard_exl3_col(loaded, suffix, tp_rank, tp_size)
            dest = param.data[expert_id, shard_idx]
        elif shard_id == "w2":
            sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size)
            dest = param.data[expert_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")

        # mcg/mul1 markers are 0-d scalars in checkpoints; packed dest is (1,).
        if suffix in ("mcg", "mul1") and sharded.numel() == 1 and dest.numel() == 1:
            dest.view(-1)[0] = sharded.reshape(-1)[0].to(dest.dtype)
            return True if return_success else None
        if tuple(dest.shape) != tuple(sharded.shape):
            if sharded.numel() == dest.numel():
                dest.copy_(sharded.reshape(dest.shape))
                return True if return_success else None
            raise RuntimeError(
                f"EXL3 load shape mismatch {weight_name} shard={shard_id} "
                f"expert={expert_id}: dest {tuple(dest.shape)} != loaded {tuple(sharded.shape)}"
            )
        dest.copy_(sharded)
        return True if return_success else None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "w13_trellis"):
            return
        for name in (
            "w13_trellis",
            "w13_suh",
            "w13_svh",
            "w13_mcg",
            "w13_mul1",
            "w2_trellis",
            "w2_suh",
            "w2_svh",
            "w2_mcg",
            "w2_mul1",
        ):
            if hasattr(layer, name):
                getattr(layer, name)._exl3_owner = layer

        # Detect codebook from loaded markers. Checkpoints are exclusively mcg OR mul1.
        mul1_13 = layer.w13_mul1.reshape(-1) if hasattr(layer, "w13_mul1") else None
        mul1_2 = layer.w2_mul1.reshape(-1) if hasattr(layer, "w2_mul1") else None
        mcg13 = layer.w13_mcg.reshape(-1)
        mcg2 = layer.w2_mcg.reshape(-1)
        n_mul1 = int(((mul1_13 != 0).sum() if mul1_13 is not None else 0) + ((mul1_2 != 0).sum() if mul1_2 is not None else 0))
        n_mcg = int((mcg13 != 0).sum() + (mcg2 != 0).sum())
        cfg_cb = str(getattr(self.quant_config, "codebook", "mcg") or "mcg")
        if n_mul1 > 0 and n_mcg == 0:
            use_mul1, use_mcg = True, False
        elif n_mcg > 0 and n_mul1 == 0:
            use_mul1, use_mcg = False, True
        elif n_mul1 > 0 and n_mcg > 0:
            # Prefer whatever quant_config says; default mul1 if both somehow set
            use_mul1 = cfg_cb == "mul1"
            use_mcg = not use_mul1
            logger.warning(
                "EXL3 MoE saw both mcg (%s) and mul1 (%s) markers; using codebook=%s",
                n_mcg, n_mul1, "mul1" if use_mul1 else "mcg",
            )
        else:
            # Nothing loaded — fall back to quant_config (default was historically mcg)
            use_mul1 = cfg_cb == "mul1"
            use_mcg = not use_mul1
            logger.warning(
                "EXL3 MoE markers missing (mcg=0 mul1=0); falling back to quant_config.codebook=%s",
                cfg_cb,
            )

        if use_mcg:
            if not torch.all(mcg13 == MCG_MARKER_SIGNED_INT32) or not torch.all(mcg2 == MCG_MARKER_SIGNED_INT32):
                layer.w13_mcg.fill_(MCG_MARKER_SIGNED_INT32)
                layer.w2_mcg.fill_(MCG_MARKER_SIGNED_INT32)
        if use_mul1:
            if mul1_13 is None or not torch.all(mul1_13 == MUL1_MARKER_SIGNED_INT32) or not torch.all(mul1_2 == MUL1_MARKER_SIGNED_INT32):
                layer.w13_mul1.fill_(MUL1_MARKER_SIGNED_INT32)
                layer.w2_mul1.fill_(MUL1_MARKER_SIGNED_INT32)

        layer._exl3_use_mul1 = bool(use_mul1)
        layer._exl3_use_mcg = bool(use_mcg)

        n_exp = int(layer.w13_trellis.shape[0])
        inners: list[dict[str, Any]] = []
        for e in range(n_exp):
            if use_mul1:
                gate = make_linear_exl3(
                    layer.w13_trellis[e, 0],
                    layer.w13_suh[e, 0],
                    layer.w13_svh[e, 0],
                    None,
                    layer.w13_mul1[e, 0],
                )
                up = make_linear_exl3(
                    layer.w13_trellis[e, 1],
                    layer.w13_suh[e, 1],
                    layer.w13_svh[e, 1],
                    None,
                    layer.w13_mul1[e, 1],
                )
                down = make_linear_exl3(
                    layer.w2_trellis[e],
                    layer.w2_suh[e],
                    layer.w2_svh[e],
                    None,
                    layer.w2_mul1[e],
                )
            else:
                gate = make_linear_exl3(
                    layer.w13_trellis[e, 0],
                    layer.w13_suh[e, 0],
                    layer.w13_svh[e, 0],
                    layer.w13_mcg[e, 0],
                )
                up = make_linear_exl3(
                    layer.w13_trellis[e, 1],
                    layer.w13_suh[e, 1],
                    layer.w13_svh[e, 1],
                    layer.w13_mcg[e, 1],
                )
                down = make_linear_exl3(
                    layer.w2_trellis[e],
                    layer.w2_suh[e],
                    layer.w2_svh[e],
                    layer.w2_mcg[e],
                )
            inners.append({"gate": gate, "up": up, "down": down})
        layer._exl3_inners = inners

        backend = get_moe_kernel_backend()
        try:
            build_exl3_fused_state(layer, inners)
        except Exception as exc:
            layer._exl3_ptrs = None
            logger.warning("Failed building EXL3 fused state: %s", exc)

        if not self._logged:
            logger.info(
                "EXL3 trellis engaged for SGLang routed experts: bits=%s "
                "experts_local=%s hidden=%s intermediate_local=%s backend=%s codebook=%s",
                self.bits,
                n_exp,
                layer._exl3_hidden_size,
                layer._exl3_intermediate_local,
                backend,
                "mul1" if getattr(layer, "_exl3_use_mul1", False) else "mcg",
            )
            self._logged = True

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> Any:
        dispatch_output = kwargs.get("dispatch_output")
        is_sglang_dispatcher = False

        if dispatch_output is not None:
            is_sglang_dispatcher = True
            x = dispatch_output.hidden_states
            topk_output = dispatch_output.topk_output
            if hasattr(topk_output, "topk_weights"):
                topk_weights = topk_output.topk_weights
                topk_ids = topk_output.topk_ids
            elif hasattr(topk_output, "to_standard"):
                materialized = topk_output.to_standard()
                topk_weights = materialized.topk_weights
                topk_ids = materialized.topk_ids
            else:
                raise ValueError(f"Unsupported SGLang top-k output format: {type(topk_output)!r}")
        elif len(args) >= 1 and hasattr(args[0], "hidden_states") and hasattr(args[0], "topk_output"):
            dispatch_output = args[0]
            is_sglang_dispatcher = True
            x = dispatch_output.hidden_states
            topk_output = dispatch_output.topk_output
            if hasattr(topk_output, "topk_weights"):
                topk_weights = topk_output.topk_weights
                topk_ids = topk_output.topk_ids
            elif hasattr(topk_output, "to_standard"):
                materialized = topk_output.to_standard()
                topk_weights = materialized.topk_weights
                topk_ids = materialized.topk_ids
            else:
                raise ValueError(f"Unsupported SGLang top-k output format: {type(topk_output)!r}")
        elif len(args) >= 3:
            x = args[0]
            topk_weights = args[1]
            topk_ids = args[2]
        elif "x" in kwargs and "topk_ids" in kwargs and "topk_weights" in kwargs:
            x = kwargs["x"]
            topk_weights = kwargs["topk_weights"]
            topk_ids = kwargs["topk_ids"]
        else:
            raise ValueError(f"Unrecognized EXL3MoEMethod.apply arguments: args={args}, kwargs={kwargs}")

        limit = getattr(getattr(layer, "moe_runner_config", None), "swiglu_limit", None) or SWIGLU_LIMIT_DEFAULT
        out = apply_exl3_experts(x, topk_ids, topk_weights, layer, limit=float(limit))

        if is_sglang_dispatcher:
            sci = StandardCombineInput
            if sci is None:
                try:
                    from sglang.srt.layers.moe.token_dispatcher.standard import (
                        StandardCombineInput as sci,
                    )
                except Exception:
                    sci = None
            if sci is not None:
                try:
                    return sci(hidden_states=out)
                except TypeError:
                    pass
            # combine() does: (hidden_states,) = combine_input
            return (out,)
        return out


class EXL3LinearMethod(LinearMethodBase):
    """SGLang linear method for dense non-routed projections quantized with EXL3."""

    def __init__(self, quant_config: EXL3Config, bits: int | None = None) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.bits = int(bits) if bits is not None else quant_config.bits
        self._logged = False


    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        n_shards = len(output_partition_sizes)
        k_words = self.bits * 16
        for i, out_size in enumerate(output_partition_sizes):
            if input_size_per_partition % 16 or out_size % 16:
                raise ValueError(
                    f"EXL3 trellis tiles are 16-wide; "
                    f"shard {i}: in={input_size_per_partition} out={out_size}"
                )
        in_tiles = input_size_per_partition // 16
        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        packs = []
        for i, out_size in enumerate(output_partition_sizes):
            pack = {
                "trellis": torch.empty(in_tiles, out_size // 16, k_words, dtype=torch.int16),
                "suh": torch.empty(input_size_per_partition, dtype=torch.float16),
                "svh": torch.empty(out_size, dtype=torch.float16),
                "mcg": torch.zeros(1, dtype=torch.int32),
                "mul1": torch.zeros(1, dtype=torch.int32),
                "filled": False,
                "skip": False,
            }
            packs.append(pack)
        layer._exl3_lin_packs = packs
        layer._exl3_lin_merged = {}
        dummy = {
            "trellis": Parameter(torch.empty(in_tiles, sum(s // 16 for s in output_partition_sizes), k_words, dtype=torch.int16), requires_grad=False),
            "suh": Parameter(torch.empty(n_shards, input_size_per_partition, dtype=torch.float16), requires_grad=False),
            "svh": Parameter(torch.empty(sum(output_partition_sizes), dtype=torch.float16), requires_grad=False),
            "mcg": Parameter(torch.zeros(n_shards, 1, dtype=torch.int32), requires_grad=False),
            "mul1": Parameter(torch.zeros(n_shards, 1, dtype=torch.int32), requires_grad=False),
        }
        for sname, p in dummy.items():
            layer.register_parameter(sname, p)
            set_weight_attrs(p, extra)
            p.weight_loader = self._load_linear
            p._exl3_owner = layer
            p._exl3_suffix = sname
        layer._exl3_linear_input_size_per_partition = input_size_per_partition
        layer._exl3_linear_output_partition_sizes = list(output_partition_sizes)
        layer._exl3_linear_k_words = k_words
        layer._exl3_linear_bits = self.bits

    def _shard_index(self, shard_id):
        if shard_id is None:
            return None
        if isinstance(shard_id, str) and shard_id in ("q", "k", "v"):
            return {"q": 0, "k": 1, "v": 2}[shard_id]
        return shard_id

    def _load_linear(self, param, loaded_weight, shard_id=None):
        layer = getattr(param, "_exl3_owner", None)
        suffix = getattr(param, "_exl3_suffix", None)
        src = loaded_weight.detach().contiguous()
        if layer is None or suffix is None:
            if param.numel() == src.numel():
                param.data.copy_(src.reshape(param.shape))
            return
        packs = layer._exl3_lin_packs
        sizes = layer._exl3_linear_output_partition_sizes
        idx = self._shard_index(shard_id)
        if idx is None:
            idx = 0 if len(packs) == 1 else None
        if isinstance(idx, tuple):
            start, end = min(idx), max(idx) + 1
            key = (start, end)
            merged = layer._exl3_lin_merged.setdefault(key, {})
            out = sum(sizes[start:end])
            in_size = layer._exl3_linear_input_size_per_partition
            k_words = layer._exl3_linear_k_words
            in_tiles = in_size // 16
            if suffix == "trellis" and "trellis" not in merged:
                merged["trellis"] = torch.empty(in_tiles, out // 16, k_words, dtype=torch.int16, device=src.device)
            if suffix == "suh" and "suh" not in merged:
                merged["suh"] = torch.empty(in_size, dtype=torch.float16, device=src.device)
            if suffix == "svh" and "svh" not in merged:
                merged["svh"] = torch.empty(out, dtype=torch.float16, device=src.device)
            dest = merged.get(suffix)
            if dest is None:
                if src.numel() == 1:
                    merged[suffix] = src.reshape(1).to(torch.int32).cpu()
                    for j in range(start, end):
                        packs[j]["skip"] = True
                    return
                return
            if dest.numel() == src.numel():
                dest.copy_(src.reshape(dest.shape))
            elif suffix == "trellis":
                # Mixed-K packs (e.g. 3.05bpw_h5): GDN qkv trellis is K=5
                # (k_words=80) while create_weights used config bits=3 (48).
                # Allocate a fresh aligned buffer; mmap views crash coop_autotune.
                aligned = torch.empty(src.shape, dtype=torch.int16, device=dest.device)
                aligned.copy_(src.to(dtype=torch.int16, device=dest.device).reshape(src.shape))
                merged[suffix] = aligned
            elif suffix == "suh" and src.ndim == 1 and dest.numel() == src.numel():
                dest.copy_(src.to(dest.dtype))
            elif src.numel() == 1:
                dest.fill_(src.reshape(()).item())
            else:
                raise RuntimeError(f"EXL3 merged {suffix} {tuple(dest.shape)} vs {tuple(src.shape)} shard={shard_id}")
            for j in range(start, end):
                packs[j]["skip"] = True
            merged["filled"] = True
            return
        i = int(idx)
        pack = packs[i]
        dest = pack[suffix]
        if src.numel() == 1:
            pack[suffix] = src.reshape(1).to(dest.dtype) if suffix in ("mcg", "mul1") else dest
            if suffix in ("mcg", "mul1"):
                pack[suffix].fill_(src.reshape(()).item()) if dest.numel() == 1 else None
                pack[suffix] = torch.tensor([src.reshape(()).item()], dtype=torch.int32)
            pack["filled"] = True
            return
        if dest.numel() != src.numel():
            if suffix == "trellis":
                aligned = torch.empty(src.shape, dtype=torch.int16, device=dest.device)
                aligned.copy_(src.to(dtype=torch.int16, device=dest.device).reshape(src.shape))
                pack[suffix] = aligned
                pack["filled"] = True
                return
            raise RuntimeError(f"EXL3 linear {suffix} shard={shard_id} dest {tuple(dest.shape)} vs {tuple(src.shape)}")
        dest.copy_(src.reshape(dest.shape))
        pack["filled"] = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "_exl3_lin_packs"):
            return
        linears = []
        used = set()
        for (start, end), merged in sorted(layer._exl3_lin_merged.items()):
            if "trellis" not in merged:
                continue
            mcg = merged.get("mcg")
            mul1 = merged.get("mul1")
            mcg_t = mcg if mcg is not None and mcg.numel() and mcg.reshape(-1)[0].item() != 0 else None
            mul1_t = mul1 if mul1 is not None and mul1.numel() and mul1.reshape(-1)[0].item() != 0 else None
            linears.append(
                make_linear_exl3(
                    merged["trellis"].contiguous(),
                    merged["suh"].contiguous(),
                    merged["svh"].contiguous(),
                    mcg_t,
                    mul1_t,
                    out_dtype=torch.float16,
                )
            )
            used.update(range(start, end))
        for i, pack in enumerate(layer._exl3_lin_packs):
            if i in used or pack.get("skip") or not pack.get("filled"):
                if i not in used:
                    linears.append(None)
                continue
            mcg = pack["mcg"]
            mul1 = pack["mul1"]
            mcg_t = mcg if mcg.numel() and int(mcg.reshape(-1)[0]) != 0 else None
            mul1_t = mul1 if mul1.numel() and int(mul1.reshape(-1)[0]) != 0 else None
            linears.append(
                make_linear_exl3(
                    pack["trellis"].contiguous(),
                    pack["suh"].contiguous(),
                    pack["svh"].contiguous(),
                    mcg_t,
                    mul1_t,
                    out_dtype=torch.float16,
                )
            )
        layer._exl3_linears = [x for x in linears if x is not None]
        for param_name in ("trellis", "suh", "svh", "mcg", "mul1"):
            if hasattr(layer, param_name):
                try:
                    delattr(layer, param_name)
                except Exception:
                    pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        linears = getattr(layer, "_exl3_linears", None)
        if not linears:
            raise RuntimeError("EXL3 linear layers were not built after weight load")
        orig_shape = x.shape
        if len(orig_shape) > 2:
            rows = math.prod(orig_shape[:-1])
            x_2d = x.reshape(rows, orig_shape[-1])
        else:
            x_2d = x
        x_fp16 = x_2d.to(torch.float16).contiguous()
        outputs = [lin.forward(x_fp16, {}, out_dtype=torch.float32) for lin in linears]
        y = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        y = y.to(dtype=x.dtype)
        if bias is not None:
            y = y + bias
        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], y.shape[-1])
        return y
