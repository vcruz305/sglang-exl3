"""RicursiveForCausalLM for SGLang + sglang-exl3 (Ripple-65B-Preview).

Adapted from Ricursive's quantization-aware SGLang integration
(ripple_sglang/local/model.py). Differences for EXL3/SAGE-REXL3 packs:

* Every linear projector is a SGLang ``LinearBase`` that receives the active
  quantization config (EXL3Config) so EXL3 trellis groups load and run on
  device. Embeddings and norms stay BF16.
* The MLP uses separate ``gate_proj`` / ``up_proj`` ColumnParallelLinears
  (no fused gate_up_proj) so the EXL3 GEMMs never concatenate/``-split two
  independent sign-vector groups.
* Attention and MLP run in float16 internally: the hidden state is converted
  once per sublayer, ``q/k/v/g`` projections reuse that single conversion, and
  the final output is converted back to BF16 once per sublayer.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers
from torch import nn
from transformers import PretrainedConfig

from ..dense import _mgemm_pair_compatible, _mgemm_two

# Fused 2-matrix EXL3 GEMM (q+g, k+v, gate+up) is opt-in until sign-off.
_FUSE_MGEMM = os.environ.get("SGLANG_EXL3_MGEMM", "0") == "1"
_MGEMM_MAX_ROWS = 32


def _exl3_lines(linear):
    return getattr(linear, "_exl3_linears", None) or [None]


def rotary_cache_length(config) -> int:
    """Include DFlash lookahead beyond the admitted context boundary."""
    from sglang.srt.server_args import get_global_server_args

    args = get_global_server_args()
    native = int(args.context_length or config.max_position_embeddings)
    return native + (
        int(args.speculative_num_draft_tokens) if args.speculative_algorithm == "DFLASH" else 0
    )


class RicursiveMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_clamp: Optional[float] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.activation_clamp = activation_clamp
        # Separate gate/up projections: each EXL3 group keeps its own suh/svh
        # sign vectors, so they are never fused into one GEMM.
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXL3 linears output fp16; convert once per MLP call.
        x16 = x.to(torch.float16).contiguous() if x.dtype != torch.float16 else x.contiguous()
        gl = _exl3_lines(self.gate_proj)[0]
        ul = _exl3_lines(self.up_proj)[0]
        if (
            _FUSE_MGEMM
            and x16.shape[0] <= _MGEMM_MAX_ROWS
            and _mgemm_pair_compatible(gl, ul)
        ):
            gate, up = _mgemm_two(self.gate_proj, x16, gl, ul)
        else:
            gate, _ = self.gate_proj(x16)
            up, _ = self.up_proj(x16)
        if self.activation_clamp is not None:
            gate = gate.clamp(max=self.activation_clamp)
            up = up.clamp(min=-self.activation_clamp, max=self.activation_clamp)
        act = F.silu(gate) * up
        out, _ = self.down_proj(act)
        return out.to(torch.bfloat16) if out.dtype != torch.bfloat16 else out


class RicursiveAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError(
                "Ricursive separate Q/K/V projections require heads divisible by attention TP"
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        if getattr(config, "use_gated_attention", True):
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("g_proj", prefix),
            )
        else:
            self.g_proj = None

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.use_valres = bool(
            getattr(config, "use_valres", getattr(config, "use_value_residuals", False))
        )
        if self.use_valres:
            self.value_res_lambda = nn.Parameter(torch.zeros(1))

        layer_type = config.layer_types[layer_id]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = (config.sliding_window - 1) if self.is_sliding else -1

        self.rotary_emb = None
        if self.is_sliding:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=rotary_cache_length(config),
                base=config.rope_theta,
                is_neox_style=True,
            )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The q/k norm weights are BF16; casting once per sublayer keeps the
        # triton attention backend happy (it requires q/k/v to share dtype with
        # the bf16 KV cache) and lets the fused mgemm outputs stay fp16.
        if q.dtype != torch.bfloat16:
            q = q.to(torch.bfloat16)
        if k.dtype != torch.bfloat16:
            k = k.to(torch.bfloat16)
        q = self.q_norm.forward_native(q.reshape(-1, self.head_dim)).reshape(-1, self.q_size)
        k = self.k_norm.forward_native(k.reshape(-1, self.head_dim)).reshape(-1, self.kv_size)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        value_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value_input = hidden_states
        if value_residual is not None and self.use_valres:
            blend = torch.sigmoid(self.value_res_lambda[0]).to(dtype=hidden_states.dtype)
            value_input = torch.lerp(
                value_residual.to(dtype=hidden_states.dtype), hidden_states, blend
            )

        # Single fp16 conversion for the whole attention block.
        h16 = hidden_states.to(torch.float16).contiguous() if hidden_states.dtype != torch.float16 else hidden_states.contiguous()
        v16 = value_input.to(torch.float16).contiguous() if value_input.dtype != torch.float16 else value_input.contiguous()

        ql = _exl3_lines(self.q_proj)[0]
        gl = _exl3_lines(self.g_proj)[0] if self.g_proj is not None else None
        kl = _exl3_lines(self.k_proj)[0]
        vl = _exl3_lines(self.v_proj)[0]
        rows = h16.shape[0]
        fuse = _FUSE_MGEMM and rows <= _MGEMM_MAX_ROWS

        # (q, g) share h16 and have identical shapes in 80/80 layers -> 1 GEMM.
        gate = None
        if fuse and gl is not None and _mgemm_pair_compatible(ql, gl):
            q, gate = _mgemm_two(self.q_proj, h16, ql, gl)
        else:
            q, _ = self.q_proj(h16)
            if self.g_proj is not None:
                gate, _ = self.g_proj(h16)

        # (k, v) share h16 when no value residual -> 1 fused GEMM.
        if fuse and v16 is h16 and _mgemm_pair_compatible(kl, vl):
            k, v = _mgemm_two(self.k_proj, h16, kl, vl)
            # mgemm returns views into the shared workspace; copy v to bf16 so
            # the KV-cache store (and attention dtype check) sees bf16.
            v = v.to(torch.bfloat16)
        else:
            k, _ = self.k_proj(h16)
            v, _ = self.v_proj(v16)
            if v.dtype != torch.bfloat16:
                v = v.to(torch.bfloat16)

        q, k = self._apply_qk_norm(q, k)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)

        if self.g_proj is not None:
            gate = torch.sigmoid(gate.float()).to(attn_output.dtype)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output.to(torch.bfloat16) if output.dtype != torch.bfloat16 else output


class RicursiveDecoderLayer(nn.Module):
    """PeriNorm decoder layer: 4 RMSNorms, post-norm inside the residual on both sublayers."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = RicursiveAttention(
            config, layer_id, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = RicursiveMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation_clamp=getattr(
                config, "ffn_activation_clamp", getattr(config, "ffn_clamp_threshold", None)
            ),
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        value_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions, hidden_states, forward_batch, value_residual=value_residual
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        return hidden_states


class RicursiveModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: RicursiveDecoderLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_scale = getattr(config, "embed_scale", None)
        self.use_valres = bool(getattr(config, "use_valres", False))
        self.dflash_capture_layers: tuple[int, ...] = ()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if input_embeds is None:
            # Defensive: clamp to the physical vocab range. Under DFLASH with
            # a quantized (EXL3) lm_head the candidate selector can produce
            # out-of-range ids; clamping keeps the embedding lookup valid
            # instead of triggering a device-side assert that poisons the
            # whole CUDA context. Cost is negligible.
            hidden_states = self.embed_tokens(
                input_ids.clamp(0, int(self.config.vocab_size) - 1)
            )
        else:
            hidden_states = input_embeds
        if self.embed_scale is not None:
            hidden_states = hidden_states * self.embed_scale
        value_residual = hidden_states if self.use_valres else None

        captured = {}
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                positions, hidden_states, forward_batch, value_residual=value_residual
            )
            if index in self.dflash_capture_layers:
                captured[index] = hidden_states

        hidden_states = self.norm(hidden_states)
        if self.dflash_capture_layers:
            aux_hidden_states = [captured[index] for index in self.dflash_capture_layers]
            return hidden_states, aux_hidden_states
        return hidden_states


class RicursiveForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.model = RicursiveModel(config, quant_config, prefix=add_prefix("model", prefix))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        """Expose the target embedding table shared by the DFlash worker."""
        return self.model.embed_tokens

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture post-block, pre-final-norm states in the drafter's configured order."""
        if (
            not layer_ids
            or any(type(index) is not int for index in layer_ids)
            or len(set(layer_ids)) != len(layer_ids)
            or any(index < 0 or index >= len(self.model.layers) for index in layer_ids)
        ):
            raise ValueError("DFlash requires unique, valid target layer indices")
        self.model.dflash_capture_layers = tuple(layer_ids)

    def get_attention_sliding_window_size(self) -> Optional[int]:
        layer_types = getattr(self.config, "layer_types", ()) or ()
        if "sliding_attention" not in layer_types:
            return None
        sliding_window = getattr(self.config, "sliding_window", None)
        if sliding_window is None:
            return None
        return sliding_window - 1

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if self.model.dflash_capture_layers:
            hidden_states, aux_hidden_states = hidden_states
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def _find_exl3_param(self, params: dict, base: str, suffix: str):
        for pname, p in params.items():
            if pname.startswith(base + ".") and pname.endswith("_" + suffix):
                if hasattr(p, "_exl3_shard") and hasattr(p, "weight_loader"):
                    return p
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load checkpoint tensors, mapping EXL3 group files onto linear params."""
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded = set()
        unused = []
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            p = params.get(name)
            if p is not None:
                getattr(p, "weight_loader", default_weight_loader)(p, weight)
                loaded.add(name)
                continue
            # EXL3 group tensor: "<proj>.trellis" -> "<proj>.exl3_<i>_trellis"
            handled = False
            for suffix in ("trellis", "suh", "svh", "mcg", "mul1"):
                end = "." + suffix
                if name.endswith(end):
                    base = name[: -len(end)]
                    sub = self._find_exl3_param(params, base, suffix)
                    if sub is not None:
                        sub.weight_loader(sub, weight)
                        handled = True
                    break
            if not handled and not name.endswith("rotary_emb.inv_freq"):
                unused.append(name)
        tied = {"lm_head.weight"} if self.config.tie_word_embeddings else set()
        unloaded = {
            n
            for n, p in params.items()
            if not hasattr(p, "_exl3_shard")
            and n not in loaded
            and n not in tied
            and not n.endswith(("k_scale", "v_scale"))
        }
        if unused:
            import logging

            logging.getLogger("sglang.exl3.ricursive").warning(
                "Ricursive: %d unconsumed checkpoint tensors, first: %s", len(unused), unused[:8]
            )
        if unloaded:
            import logging

            logging.getLogger("sglang.exl3.ricursive").warning(
                "Ricursive: %d non-EXL3 params never loaded, first: %s", len(unloaded), sorted(unloaded)[:8]
            )


def _register_ricursive():
    try:
        from sglang.srt.models.registry import ModelRegistry

        ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
    except Exception:
        pass


_register_ricursive()

