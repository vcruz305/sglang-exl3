"""Dense EXL3 linear method for SGLang.

Runs ExLlamaV3's ``exl3_gemm`` kernel (any bitwidth, including half-integer
rates) with persistent, CUDA-graph-safe workspace buffers. The method is
bitwidth-agnostic: every per-tensor ``trellis`` shape is read from the loaded
checkpoint tensors, so 2, 2.5, 3, 4, 5 and 6-bit groups load identically.
Each projection keeps its OWN ``suh``/``svh`` sign vectors (they are not merged
across projections); a merged module simply holds one instance per shard and
concatenates their outputs.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import torch
from torch.nn.parameter import Parameter

from .exl3 import LinearMethodBase, set_weight_attrs

__all__ = ["EXL3DenseLinear", "EXL3LinearMethod"]


def _load_exllamav3_ext():
    import exllamav3_ext

    return exllamav3_ext


# ---------------------------------------------------------------------------
# Persistent, CUDA-graph-safe workspace + fused multi-GEMM helpers.
# ---------------------------------------------------------------------------
# Workspace is keyed by (device, stream) so capture never reallocates mid-graph.
# Buffers are 1D flat (always contiguous for any (rows, cols) .view) so we never
# hand the kernel a strided view and never resize during capture.
_WORKSPACE: dict[Any, dict[str, Any]] = {}

# Bound for the fused mgemm path: it shines for small row counts (decode /
# target-verify blocks), and falls back to per-linear exl3_gemm otherwise.
_MGEMM_MAX_ROWS = 32


def _get_workspace(device: torch.device, need_had: int, need_out: int) -> dict[str, Any]:
    key = (device.index or 0, id(torch.cuda.current_stream(device)))
    ws = _WORKSPACE.get(key)
    if ws is not None and ws["had_flat"].numel() >= need_had and ws["out_flat"].numel() >= need_out:
        return ws
    if ws is not None:
        need_had = max(need_had, ws["had_flat"].numel())
        need_out = max(need_out, ws["out_flat"].numel())
    ws = {
        "had_flat": torch.empty((need_had,), dtype=torch.float16, device=device),
        "out_flat": torch.empty((need_out,), dtype=torch.float16, device=device),
    }
    _WORKSPACE[key] = ws
    return ws


def _mgemm_pair_compatible(lin0, lin1) -> bool:
    return (
        lin0 is not None
        and lin1 is not None
        and lin0.K == lin1.K
        and lin0.in_features == lin1.in_features
        and lin0.out_features == lin1.out_features
        and lin0.mcg == lin1.mcg
        and lin0.mul1 == lin1.mul1
    )


def _get_mgemm_ptrs(layer, lin0, lin1, dev) -> dict[str, Any]:
    """Lazily build + cache device pointer tensors for a fused 2-GEMM pair."""
    ptrs = getattr(layer, "_exl3_mgemm_ptrs", None)
    if ptrs is None:
        ptrs = {}
        layer._exl3_mgemm_ptrs = ptrs
    key = (id(lin0), id(lin1))
    entry = ptrs.get(key)
    if entry is None:
        entry = {
            "trellis": torch.tensor([lin0.trellis.data_ptr(), lin1.trellis.data_ptr()], dtype=torch.long, device=dev),
            "suh": torch.tensor([lin0.suh.data_ptr(), lin1.suh.data_ptr()], dtype=torch.long, device=dev),
            "svh": torch.tensor([lin0.svh.data_ptr(), lin1.svh.data_ptr()], dtype=torch.long, device=dev),
        }
        ptrs[key] = entry
    return entry


def _mgemm_two(layer, x2, lin0, lin1):
    """Fused 2-matrix EXL3 GEMM over a shared input.

    x2: (rows, in_f) fp16 contiguous. Returns (out0, out1) each (rows, out_f).
    """
    ext = _load_exllamav3_ext()
    dev = x2.device
    rows, in_f = x2.shape
    out_f = lin0.out_features
    ptrs = _get_mgemm_ptrs(layer, lin0, lin1, dev)
    ws = _get_workspace(dev, 2 * rows * in_f, 2 * rows * out_f)
    xv = x2.view(1, rows, in_f)
    ahad = ws["had_flat"][: 2 * rows * in_f].view(2, rows, in_f)
    out = ws["out_flat"][: 2 * rows * out_f].view(2, rows, out_f)
    ext.exl3_mgemm(
        xv, ptrs["trellis"], out, ptrs["suh"], ahad, ptrs["svh"],
        None, None, lin0.K, -1, lin0.mcg, lin0.mul1, -1, -1, 0, 1,
        None, None, None, None, 0,
    )
    return out[0], out[1]


class EXL3DenseLinear:
    """Hold one projected EXL3 group + the metadata ExLlamaV3 kernels need."""

    __slots__ = (
        "trellis",
        "suh",
        "svh",
        "K",
        "mcg",
        "mul1",
        "in_features",
        "out_features",
    )

    def __init__(
        self,
        trellis: torch.Tensor,
        suh: torch.Tensor,
        svh: torch.Tensor,
        mcg: Optional[torch.Tensor] = None,
        mul1: Optional[torch.Tensor] = None,
    ):
        self.trellis = trellis.contiguous()
        self.suh = suh.contiguous()
        self.svh = svh.contiguous()
        self.in_features = int(suh.numel())
        self.out_features = int(svh.numel())
        K = self.trellis.shape[-1] / 16
        self.K = int(K) if float(K).is_integer() else K
        mcg_on = bool(mcg is not None and int(mcg.reshape(-1)[0]) != 0)
        mul1_on = bool(mul1 is not None and int(mul1.reshape(-1)[0]) != 0)
        if mcg_on and mul1_on:
            raise ValueError("EXL3 linear has both mcg and mul1 markers set")
        self.mcg = mcg_on or (not mul1_on)
        self.mul1 = mul1_on


class EXL3LinearMethod(LinearMethodBase):
    """SGLang linear method executing dense EXL3 trellis GEMMs."""

    def __init__(self, quant_config, bits: Optional[int] = None) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.bits = int(bits) if bits is not None else int(quant_config.bits)
        self._logged = False

    # ------------------------------------------------------------------
    # weight creation / loading
    # ------------------------------------------------------------------

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        n_shards = len(output_partition_sizes)
        for out_size in output_partition_sizes:
            if input_size_per_partition % 16 or out_size % 16:
                raise ValueError(
                    "EXL3 trellis tiles are 16-wide; "
                    f"in={input_size_per_partition} out={out_size}"
                )
        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        layer._exl3_out_dtype = params_dtype or torch.bfloat16
        # DFlash2's candidate selector requires lm_head.weight to exist;
        # keep a non-dense placeholder so should_apply_lm_head_quant_method
        # routes to our quant_method.apply instead of the dense matmul path.
        # Only for ParallelLMHead, never for the actual embedding table.
        if layer.__class__.__name__ == "ParallelLMHead" and not hasattr(layer, "weight"):
            layer.register_parameter(
                "weight", Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False)
            )
        packs = [
            {
                "trellis": None,
                "suh": None,
                "svh": None,
                "mcg": None,
                "mul1": None,
            }
            for _ in range(n_shards)
        ]
        layer._exl3_lin_packs = packs
        layer._exl3_lin_n_shards = n_shards
        layer._exl3_lin_input_size_per_partition = int(input_size_per_partition)
        layer._exl3_lin_output_partition_sizes = [int(s) for s in output_partition_sizes]
        layer._exl3_linears = None
        layer._exl3_ws = None
        for i in range(n_shards):
            for suffix in ("trellis", "suh", "svh", "mcg", "mul1"):
                p = Parameter(torch.empty(0), requires_grad=False)
                layer.register_parameter(f"exl3_{i}_{suffix}", p)
                set_weight_attrs(p, extra)
                p.weight_loader = self._load_linear
                p._exl3_owner = layer
                p._exl3_shard = i
                p._exl3_suffix = suffix

    @staticmethod
    def _shard_index(shard_id):
        if shard_id is None:
            return 0
        if isinstance(shard_id, str):
            if shard_id in ("q", "k", "v"):
                return {"q": 0, "k": 1, "v": 2}[shard_id]
            try:
                return int(shard_id)
            except ValueError as exc:
                raise ValueError(f"unknown EXL3 linear shard {shard_id!r}") from exc
        return int(shard_id)

    def _load_linear(self, param, loaded_weight, shard_id=None):
        layer = getattr(param, "_exl3_owner", None)
        suffix = getattr(param, "_exl3_suffix", None)
        if layer is None or suffix is None:
            return
        idx = self._shard_index(shard_id)
        pack = layer._exl3_lin_packs[idx]
        src = loaded_weight.detach()
        dev = torch.device("cuda", torch.cuda.current_device())
        if src.numel() == 1 and suffix in ("mcg", "mul1"):
            pack[suffix] = torch.tensor(
                [int(src.reshape(()).item())], dtype=torch.int32, device=dev
            )
        else:
            if suffix in ("suh", "svh"):
                src = src.to(torch.float16)
            pack[suffix] = src.to(device=dev).contiguous()

    def process_weights_after_loading(self, layer) -> None:
        packs = getattr(layer, "_exl3_lin_packs", None)
        if not packs:
            return
        linears = []
        for i, pack in enumerate(packs):
            trellis = pack.get("trellis")
            if trellis is None:
                raise RuntimeError(
                    f"EXL3 linear shard {i} received no trellis tensor; "
                    "checkpoint/quantization mismatch?"
                )
            suh = pack.get("suh")
            svh = pack.get("svh")
            if suh is None or svh is None:
                raise RuntimeError(
                    f"EXL3 linear shard {i} missing suh/svh sign vectors"
                )
            from .exl3 import make_linear_exl3

            mcg = pack.get("mcg")
            mul1 = pack.get("mul1")
            mcg_t = mcg if mcg is not None and int(mcg.reshape(-1)[0]) != 0 else None
            mul1_t = mul1 if mul1 is not None and int(mul1.reshape(-1)[0]) != 0 else None
            linears.append(
                make_linear_exl3(
                    trellis,
                    suh,
                    svh,
                    mcg_t,
                    mul1_t,
                    out_dtype=torch.float16,
                )
            )
        layer._exl3_linears = linears
        layer._exl3_ws = None
        if not self._logged:
            bits = [getattr(lin, "K", None) for lin in linears]
            self._logged = True
            import logging

            logging.getLogger("sglang.exl3").info(
                "EXL3 dense linears engaged: shards=%d K=%s",
                len(linears),
                bits,
            )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        linears = layer._exl3_linears
        if not linears:
            raise RuntimeError("EXL3 linear layers were not built after weight load")
        orig_shape = x.shape
        rows = math.prod(orig_shape[:-1]) if len(orig_shape) > 2 else orig_shape[0]
        # Selective FP32 dynamic scaling: only for linears flagged by the caller
        # (the DFlash2 draft's fc over raw concatenated target hidden states and
        # its down_proj, which has no activation clamp). Those run before a norm
        # and would otherwise overflow FP16 into NaN. The 80-layer target model
        # (RMSNorm + activation_clamp on every sublayer) pays ZERO scaling cost.
        scale = None
        if getattr(layer, "_exl3_scale_input", False):
            x_flat = x.reshape(rows, orig_shape[-1]).float()
            scale = (x_flat.abs().amax(dim=-1, keepdim=True) / 16.0).clamp_min_(1e-6)
            x2 = (x_flat / scale).to(torch.float16).contiguous()
        else:
            x2 = x.reshape(rows, orig_shape[-1])
            if x2.dtype != torch.float16 or not x2.is_contiguous():
                x2 = x2.to(torch.float16).contiguous()

        n = len(linears)
        # The fused exl3_mgemm pair path is opt-in (SGLANG_EXL3_MGEMM=1) pending
        # correctness sign-off; per-linear exl3_gemm is the default.
        use_fused = (
            rows <= _MGEMM_MAX_ROWS
            and os.environ.get("SGLANG_EXL3_MGEMM", "0") == "1"
        )
        if n == 2 and use_fused and _mgemm_pair_compatible(linears[0], linears[1]):
            # Merged gate_up_proj: one fused 2-matrix GEMM, then concat.
            o0, o1 = _mgemm_two(layer, x2, linears[0], linears[1])
            y = torch.cat((o0, o1), dim=1)
        elif (
            n == 3
            and use_fused
            and _mgemm_pair_compatible(linears[1], linears[2])
        ):
            # qkv_proj: q on its own GEMM, k+v fused into one mgemm.
            q = linears[0].forward(x2, {}, out_dtype=torch.float16)
            k, v = _mgemm_two(layer, x2, linears[1], linears[2])
            y = torch.cat((q, k, v), dim=1)
        else:
            outputs = [lin.forward(x2, {}, out_dtype=torch.float16) for lin in linears]
            y = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]

        out_dtype = getattr(layer, "_exl3_out_dtype", x.dtype)
        if scale is not None:
            y = (y.float() * scale).to(dtype=out_dtype)
        else:
            y = y.to(dtype=out_dtype)
        if bias is not None:
            y = y + bias.to(dtype=out_dtype)
        if len(orig_shape) > 2:
            y = y.view(*orig_shape[:-1], y.shape[-1])
        return y

    def apply_kv_only(self, layer, x: torch.Tensor):
        """Run ONLY the K and V shards of a qkv_proj (skip the Q GEMM).

        Used by DFlash's ``kv_proj_only`` during context KV materialization,
        where Q (10/12 of the qkv width) would otherwise be computed and thrown
        away. Returns (k, v) as separate (rows, kv_width) tensors.
        """
        linears = layer._exl3_linears
        if not linears or len(linears) != 3:
            raise RuntimeError("apply_kv_only requires a 3-shard qkv_proj")
        orig_shape = x.shape
        rows = math.prod(orig_shape[:-1]) if len(orig_shape) > 2 else orig_shape[0]
        x2 = x.reshape(rows, orig_shape[-1])
        if x2.dtype != torch.float16 or not x2.is_contiguous():
            x2 = x2.to(torch.float16).contiguous()
        k_lin, v_lin = linears[1], linears[2]
        fused = (
            rows <= _MGEMM_MAX_ROWS
            and os.environ.get("SGLANG_EXL3_MGEMM", "0") == "1"
            and _mgemm_pair_compatible(k_lin, v_lin)
        )
        if fused:
            k, v = _mgemm_two(layer, x2, k_lin, v_lin)
        else:
            k = k_lin.forward(x2, {}, out_dtype=torch.float16)
            v = v_lin.forward(x2, {}, out_dtype=torch.float16)
        out_dtype = getattr(layer, "_exl3_out_dtype", x.dtype)
        k, v = k.to(out_dtype), v.to(out_dtype)
        # fp16->fp16 is a no-op that would return views into the shared mgemm
        # workspace (overwritten by the next layer); clone in that case. bf16
        # conversion already allocates a fresh tensor.
        if out_dtype == torch.float16 and fused:
            k, v = k.clone(), v.clone()
        if len(orig_shape) > 2:
            k = k.view(*orig_shape[:-1], k.shape[-1])
            v = v.view(*orig_shape[:-1], v.shape[-1])
        return k, v
