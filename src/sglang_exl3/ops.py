"""PyTorch custom op registrations and meta/fake kernels for zero-break CUDA graphs.

Enables seamless integration with SGLang's CudaGraphRunner, piecewise_cuda_graph,
and torch.compile by providing schema definitions, CUDA dispatches, and meta/fake
shape inference for all sm_121 Blackwell EXL3 kernels.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger("sglang.exl3.ops")

_OPS_REGISTERED = False


def _get_c_module():
    for mod_name in ("sglang_exl3_c", "vllm_exl3_c"):
        try:
            return __import__(mod_name)
        except ImportError:
            pass
    return None


def register_custom_ops() -> bool:
    """Register EXL3 kernels into the torch.ops.sglang_exl3 namespace with meta kernels."""
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return True

    c_mod = _get_c_module()

    # 1. dequant_trellis
    @torch.library.custom_op("sglang_exl3::dequant_trellis", mutates_args=())
    def dequant_trellis(
        trellis: torch.Tensor,
        suh: torch.Tensor,
        svh: torch.Tensor,
        bits: int,
        mcg: bool,
    ) -> torch.Tensor:
        if c_mod is not None and hasattr(c_mod, "dequant_trellis"):
            return c_mod.dequant_trellis(trellis, suh, svh, bits, mcg)
        raise RuntimeError("sglang_exl3_c extension not available for dequant_trellis")

    @dequant_trellis.register_fake
    def _(trellis, suh, svh, bits, mcg):
        rows = trellis.size(0) * 16
        cols = trellis.size(1) * 16
        return torch.empty((rows, cols), dtype=torch.float16, device=trellis.device)

    # 2. exl3_gemv
    @torch.library.custom_op("sglang_exl3::exl3_gemv", mutates_args=())
    def exl3_gemv(
        x: torch.Tensor,
        trellis: torch.Tensor,
        suh: torch.Tensor,
        svh: torch.Tensor,
        K: int,
        mcg: bool,
        mmode: int = 1,
    ) -> torch.Tensor:
        if c_mod is not None and hasattr(c_mod, "exl3_gemv"):
            return c_mod.exl3_gemv(x, trellis, suh, svh, K, mcg, mmode)
        raise RuntimeError("sglang_exl3_c extension not available for exl3_gemv")

    @exl3_gemv.register_fake
    def _(x, trellis, suh, svh, K, mcg, mmode=1):
        out_dim = svh.size(0) if svh.ndim == 1 else trellis.size(1) * 16
        return torch.empty((x.shape[0], out_dim), dtype=x.dtype, device=x.device)

    # 3. exl3_gemm
    @torch.library.custom_op("sglang_exl3::exl3_gemm", mutates_args=())
    def exl3_gemm(
        x: torch.Tensor,
        trellis: torch.Tensor,
        suh: torch.Tensor,
        svh: torch.Tensor,
        K: int,
        mcg: bool,
    ) -> torch.Tensor:
        if c_mod is not None and hasattr(c_mod, "exl3_gemm"):
            return c_mod.exl3_gemm(x, trellis, suh, svh, K, mcg)
        raise RuntimeError("sglang_exl3_c extension not available for exl3_gemm")

    @exl3_gemm.register_fake
    def _(x, trellis, suh, svh, K, mcg):
        out_dim = svh.size(0) if svh.ndim == 1 else trellis.size(1) * 16
        return torch.empty((x.shape[0], out_dim), dtype=x.dtype, device=x.device)

    # 4. p2b_gemv_batched
    @torch.library.custom_op("sglang_exl3::p2b_gemv_batched", mutates_args=())
    def p2b_gemv_batched(
        x: torch.Tensor,
        trellis_ptrs: torch.Tensor,
        suh_ptrs: torch.Tensor,
        svh_ptrs: torch.Tensor,
        expert_indices: torch.Tensor,
        K: int,
        mcg: bool,
        mmode: int = 1,
    ) -> torch.Tensor:
        if c_mod is not None and hasattr(c_mod, "p2b_gemv_batched"):
            return c_mod.p2b_gemv_batched(
                x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, K, mcg, mmode
            )
        raise RuntimeError("sglang_exl3_c extension not available for p2b_gemv_batched")

    @p2b_gemv_batched.register_fake
    def _(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, K, mcg, mmode=1):
        e = expert_indices.numel()
        m = x.size(1) if x.dim() == 3 else x.size(0)
        n = mmode if mmode > 0 else 2048
        return torch.empty((e, m, n), dtype=x.dtype, device=x.device)

    # 5. p2b_fused_moe
    @torch.library.custom_op("sglang_exl3::p2b_fused_moe", mutates_args=("out",))
    def p2b_fused_moe(
        x: torch.Tensor,
        out: torch.Tensor,
        gate_trellis_ptrs: torch.Tensor,
        gate_suh_ptrs: torch.Tensor,
        gate_svh_ptrs: torch.Tensor,
        up_trellis_ptrs: torch.Tensor,
        up_suh_ptrs: torch.Tensor,
        up_svh_ptrs: torch.Tensor,
        down_trellis_ptrs: torch.Tensor,
        down_suh_ptrs: torch.Tensor,
        down_svh_ptrs: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        K_gate: int,
        K_up: int,
        K_down: int,
        mcg: bool,
    ) -> None:
        if c_mod is not None and hasattr(c_mod, "p2b_fused_moe"):
            c_mod.p2b_fused_moe(
                x,
                out,
                gate_trellis_ptrs,
                gate_suh_ptrs,
                gate_svh_ptrs,
                up_trellis_ptrs,
                up_suh_ptrs,
                up_svh_ptrs,
                down_trellis_ptrs,
                down_suh_ptrs,
                down_svh_ptrs,
                expert_indices,
                routing_weights,
                K_gate,
                K_up,
                K_down,
                mcg,
            )
            return None
        raise RuntimeError("sglang_exl3_c extension not available for p2b_fused_moe")

    @p2b_fused_moe.register_fake
    def _(
        x,
        out,
        gate_trellis_ptrs,
        gate_suh_ptrs,
        gate_svh_ptrs,
        up_trellis_ptrs,
        up_suh_ptrs,
        up_svh_ptrs,
        down_trellis_ptrs,
        down_suh_ptrs,
        down_svh_ptrs,
        expert_indices,
        routing_weights,
        K_gate,
        K_up,
        K_down,
        mcg,
    ):
        return None

    # 6. exl3_fat_gemm
    @torch.library.custom_op("sglang_exl3::exl3_fat_gemm", mutates_args=("out",))
    def exl3_fat_gemm(
        a: torch.Tensor,
        packed: torch.Tensor,
        out: torch.Tensor,
        svh: torch.Tensor,
        K: int,
        mcg: bool,
        mul1: bool,
    ) -> None:
        if c_mod is not None and hasattr(c_mod, "exl3_fat_gemm"):
            c_mod.exl3_fat_gemm(a, packed, out, svh, K, mcg, mul1)
            return None
        raise RuntimeError("sglang_exl3_c extension not available for exl3_fat_gemm")

    @exl3_fat_gemm.register_fake
    def _(a, packed, out, svh, K, mcg, mul1):
        return None

    # 7. exl3_fat_gemm_scatter
    @torch.library.custom_op("sglang_exl3::exl3_fat_gemm_scatter", mutates_args=("out",))
    def exl3_fat_gemm_scatter(
        a: torch.Tensor,
        packed: torch.Tensor,
        out: torch.Tensor,
        svh: torch.Tensor,
        token_idx: torch.Tensor,
        route_weight: torch.Tensor,
        K: int,
        mcg: bool,
        mul1: bool,
    ) -> None:
        if c_mod is not None and hasattr(c_mod, "exl3_fat_gemm_scatter"):
            c_mod.exl3_fat_gemm_scatter(a, packed, out, svh, token_idx, route_weight, K, mcg, mul1)
            return None
        raise RuntimeError("sglang_exl3_c extension not available for exl3_fat_gemm_scatter")

    @exl3_fat_gemm_scatter.register_fake
    def _(a, packed, out, svh, token_idx, route_weight, K, mcg, mul1):
        return None

    _OPS_REGISTERED = True
    logger.info("Successfully registered sglang_exl3 custom ops and meta kernels into PyTorch dispatcher.")
    return True
