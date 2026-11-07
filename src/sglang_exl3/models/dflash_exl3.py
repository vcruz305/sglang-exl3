"""Patches so DFlash2DraftModel loads EXL3-quantized draft checkpoints.

SGLang's stock DFlashDraftModel builds its context-feature projector (``fc``)
as an unquantized ``nn.Linear`` and its draft loader resolves parameters by
exact name. A exL3 draft keeps ``fc`` + ``layers.*.{qkv,o,gate_up,down}_proj``
as trellis groups (per-projection suh/svh), so:

* ``fc`` is rebuilt as a ``ReplicatedLinear`` carrying the EXL3 quantization
  config (the group tensors then load into per-shard EXL3 parameters);
* the load loop is replaced with a version that resolves EXL3 sub-parameters
  (``<proj>.exl3_<i>_<suffix>``) from checkpoint group names, passing the
  stacked-projection shard id (0/1 or q/k/v) through to the EXL3 loader.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional, Tuple

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.model_loader.weight_utils import default_weight_loader

from ..exl3 import EXL3Config

logger = logging.getLogger("sglang.exl3.dflash")

_EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
_EXL3_SUBPARAM_RE = re.compile(r"^exl3_(\d+)_(trellis|suh|svh|mcg|mul1)$")

_PATCHED = False


def _is_exl3_subparam(p) -> bool:
    return hasattr(p, "_exl3_shard") and hasattr(p, "weight_loader")


class _EXL3FCLinear(ReplicatedLinear):
    """ReplicatedLinear with the nn.Linear interface DFlash expects."""

    @property
    def in_features(self):
        return self.input_size

    @property
    def out_features(self):
        return self.output_size

    def forward(self, x):
        out, _ = super().forward(x)
        return out


def patch_dflash_draft_exl3() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from sglang.srt.models.dflash import DFlashDraftModel
    except Exception:
        return False

    # --- 1. fc -> ReplicatedLinear(quant_config) for EXL3 drafts ----------
    orig_init = DFlashDraftModel.__init__

    def __init__(self, config, quant_config=None, prefix: str = ""):
        orig_init(self, config, quant_config=quant_config, prefix=prefix)
        if isinstance(quant_config, EXL3Config) and not getattr(self, "_fc_exl3_replaced", False):
            old_fc = self.fc
            fc_prefix = "fc"
            new_fc = _EXL3FCLinear(
                int(old_fc.in_features),
                int(old_fc.out_features),
                bias=False,
                quant_config=quant_config,
                prefix=fc_prefix,
            )
            self.fc = new_fc
            self._fc_exl3_replaced = True
            logger.info("DFlash draft fc rebuilt as EXL3 ReplicatedLinear (%d -> %d)",
                        old_fc.in_features, old_fc.out_features)

            # Flag the two draft linears that run on raw (un-normalized)
            # activations for FP32 dynamic scaling in EXL3LinearMethod.apply:
            #   * fc runs BEFORE hidden_norm on concatenated target residual
            #     states (in = num_context_features * hidden = 25600);
            #   * each down_proj runs after Silu(gate)*up with NO activation
            #     clamp, so its input is quadratic in magnitude.
            # Without scaling these overflow FP16 -> NaN -> garbage candidate
            # ids (accept len collapses to 1.0). Target-model linears keep the
            # zero-overhead path.
            self.fc._exl3_scale_input = True
            for layer in self.layers:
                mlp = getattr(layer, "mlp", None)
                down = getattr(mlp, "down_proj", None) if mlp is not None else None
                if down is not None:
                    down._exl3_scale_input = True

    DFlashDraftModel.__init__ = __init__

    # --- 2. robust load_weights ------------------------------------------
    orig_load = DFlashDraftModel.load_weights

    _VENDOR_ENCODER_ALIASES = {
        "encoder.fc.weight": "fc.weight",
        "encoder.output_norm_enc.weight": "hidden_norm.weight",
    }

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped = name[len("model."):]
                if stripped in params_dict:
                    return stripped
            else:
                prefixed = f"model.{name}"
                if prefixed in params_dict:
                    return prefixed
            return _VENDOR_ENCODER_ALIASES.get(name)

        seen = []
        def watched(stream):
            for name, w in stream:
                if len(seen) < 8:
                    seen.append(name)
                    logger.info("DFlash incoming weight: %s", name)
                yield name, w

        for name, loaded_weight in watched(weights):
            stem, _, suffix = name.rpartition(".")
            if suffix in _EXL3_SUFFIXES:
                # EXL3 group tensor -> "<stem'>.exl3_<shard_idx>_<suffix>"
                shard_id = None
                shard_idx = 0
                mapped = stem
                for proj, idx, sid in (
                    (".q_proj", 0, "q"), (".k_proj", 1, "k"), (".v_proj", 2, "v"),
                ):
                    if proj in stem:
                        mapped = stem.replace(proj, ".qkv_proj")
                        shard_id, shard_idx = sid, idx
                        break
                else:
                    for proj, idx in ((".gate_proj", 0), (".up_proj", 1)):
                        if proj in stem:
                            mapped = stem.replace(proj, ".gate_up_proj")
                            shard_id = idx
                            shard_idx = idx
                            break
                pname = resolve_param_name(f"{mapped}.exl3_{shard_idx}_{suffix}")
                if pname is not None:
                    p = params_dict[pname]
                    if shard_id is None:
                        p.weight_loader(p, loaded_weight)
                    else:
                        p.weight_loader(p, loaded_weight, shard_id)
                    continue
                logger.warning("DFlash EXL3 group not consumed: %s", name)
                continue

            # Non-EXL3 (e.g. BF16 draft) tensors: map stacked projections
            # (q/k/v -> qkv_proj, gate/up -> gate_up_proj) exactly like the
            # upstream DFlashDraftModel.load_weights, otherwise those weights
            # are silently skipped and the draft stays at zero.
            if name.endswith(".weight"):
                handled = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name in name:
                        mapped_name = name.replace(weight_name, param_name)
                        resolved_name = resolve_param_name(mapped_name)
                        if resolved_name is not None:
                            param = params_dict[resolved_name]
                            param.weight_loader(param, loaded_weight, shard_id)
                            handled = True
                        break
                if handled:
                    continue

            resolved_name = resolve_param_name(name)
            if resolved_name is None:
                continue
            param = params_dict[resolved_name]
            wl = getattr(param, "weight_loader", default_weight_loader)
            wl(param, loaded_weight)

    DFlashDraftModel.load_weights = load_weights

    # --- 3. clamp candidate ids for the EXL3 target lm_head ---------------
    # SGLang's stock DFlash selector only understands dense lm_heads or the
    # quantization methods it knows about. For an EXL3 (trellis) target head,
    # ``is_dense_head_weight`` and ``should_apply_lm_head_quant_method`` both
    # return False, so ``_SelectorDraftSampler`` raises before baking the
    # static tensors. The eager fallback then re-attaches the target head
    # lazily and computes candidate ids WITHOUT setting the draft graph
    # capture tensors, leaving candidate_out as uninitialized garbage, which
    # becomes out-of-vocab draft tokens and crashes the target embedding with
    # ``srcIndex < srcSelectDimSize``.
    # Clamping candidate ids keeps them in vocab and lets the eager selector
    # path produce valid drafts; the true fix (EXL3-aware quantized head path)
    # is future work.
    from sglang.srt.models.dflash import DFlash2DraftModel

    orig_compute_candidates = DFlash2DraftModel.compute_candidates

    def compute_candidates(self, hidden):
        ids, logits = orig_compute_candidates(self, hidden)
        # Always clamp on-device (no .item()/host sync: this runs inside CUDA
        # graph capture, where host reads are illegal).
        return ids.clamp_(0, int(self.config.vocab_size) - 1), logits

    DFlash2DraftModel.compute_candidates = compute_candidates

    # --- 4. graph-safe EXL3 lm_head screening -----------------------------
    # SGLang's selector screens the shared target lm_head before folding the
    # selector into the draft graph: it only accepts a "dense" head weight or
    # a quant_method it recognizes. An EXL3 (trellis) head has a non-dense
    # placeholder weight, so the stock screen rejects the graph path even
    # though our EXL3 quant_method.apply IS graph-safe. Force the graph path
    # by advertising the EXL3 method as an accepted lm_head method, which
    # bakes candidate_out into the captured graph (the eager fallback left
    # those capture tensors uninitialized -> garbage ids -> CUDA assert).
    from sglang.srt.layers import logits_processor as _lp

    _UNQUANT = _lp._UNQUANTIZED_LM_HEAD_METHODS

    if "EXL3LinearMethod" in _UNQUANT:
        _UNQUANT.discard("EXL3LinearMethod")

    _orig_should = _lp.should_apply_lm_head_quant_method

    def _should_apply(lm_head, quant_method):
        if type(quant_method).__name__ == "EXL3LinearMethod":
            return True
        return _orig_should(lm_head, quant_method)

    _lp.should_apply_lm_head_quant_method = _should_apply
    # dflash.py imported it by value at module load; patch that reference too.
    import sglang.srt.models.dflash as _dflash_mod

    _dflash_mod.should_apply_lm_head_quant_method = _should_apply

    # --- 5. skip-Q KV-only projection for quantized qkv_proj ---------------
    # SGLang's DFlashAttention.kv_proj_only computes the full qkv GEMM and
    # slices out Q (10/12 of the width) when the qkv_proj is quantized and it
    # can't slice a dense weight. With an EXL3 qkv_proj we can run ONLY the K
    # and V shards, saving ~83% of the context-KV materialization GEMM cost.
    from sglang.srt.models.dflash import DFlashAttention

    orig_kv_proj_only = DFlashAttention.kv_proj_only

    def kv_proj_only(self, hidden_states):
        qm = getattr(self.qkv_proj, "quant_method", None)
        if qm is not None and hasattr(qm, "apply_kv_only"):
            return qm.apply_kv_only(self.qkv_proj, hidden_states)
        return orig_kv_proj_only(self, hidden_states)

    DFlashAttention.kv_proj_only = kv_proj_only

    _PATCHED = True
    logger.info("Patched DFlashDraftModel for EXL3 drafts (fc + sharded loading + clamped candidate ids + EXL3 lm_head screen)")
    return True
