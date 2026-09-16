"""Monkey-patch and registration utilities for SGLang EXL3 integration."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("sglang.exl3")


def patch_quantization_methods() -> bool:
    """Register EXL3 into SGLang's QUANTIZATION_METHODS table."""
    try:
        from sglang.srt.layers import quantization as sq
        from .exl3 import EXL3Config

        if hasattr(sq, "QUANTIZATION_METHODS"):
            sq.QUANTIZATION_METHODS["exl3"] = EXL3Config
        if hasattr(sq, "BASE_QUANTIZATION_METHODS"):
            sq.BASE_QUANTIZATION_METHODS["exl3"] = EXL3Config

        # Also patch model_config if it has already imported QUANTIZATION_METHODS
        try:
            from sglang.srt.configs import model_config as smc
            if hasattr(smc, "QUANTIZATION_METHODS"):
                smc.QUANTIZATION_METHODS["exl3"] = EXL3Config
        except (ImportError, AttributeError):
            pass

        return True
    except (ImportError, AttributeError) as exc:
        logger.debug("Failed to patch SGLang QUANTIZATION_METHODS: %s", exc)
        return False


def _append_choice(seq, value: str) -> None:
    if seq is not None and value not in seq:
        seq.append(value)


def patch_server_args() -> bool:
    """Add 'exl3' / 'exl3_mmap' to SGLang CLI choice lists (current and legacy homes)."""
    try:
        from sglang.srt import server_args as sa

        if hasattr(sa, "QUANTIZATION_CHOICES"):
            if "exl3" not in sa.QUANTIZATION_CHOICES:
                if hasattr(sa, "add_quantization_method_choices"):
                    sa.add_quantization_method_choices(["exl3"])
                else:
                    sa.QUANTIZATION_CHOICES.append("exl3")
        if hasattr(sa, "LOAD_FORMAT_CHOICES"):
            _append_choice(sa.LOAD_FORMAT_CHOICES, "exl3_mmap")
        try:
            from sglang.srt.arg_groups import choices as agc

            _append_choice(getattr(agc, "QUANTIZATION_CHOICES", None), "exl3")
            _append_choice(getattr(agc, "LOAD_FORMAT_CHOICES", None), "exl3_mmap")
        except (ImportError, AttributeError):
            pass
        return True
    except (ImportError, AttributeError) as exc:
        logger.debug("Failed to patch SGLang QUANTIZATION_CHOICES: %s", exc)
        return False


def patch_deepseek_v4_weights_loading() -> bool:
    """Filter extraneous multimodal weights (e.g. vision aligner) for DeepseekV4ForCausalLM."""
    try:
        from sglang.srt.models import deepseek_v4
        if hasattr(deepseek_v4, "DeepseekV4ForCausalLM"):
            cls = deepseek_v4.DeepseekV4ForCausalLM
            if not getattr(cls, "_exl3_weights_patched", False):
                orig_load_weights = cls.load_weights

                def filtered_load_weights(self, weights, is_nextn=False):
                    params_dict = dict(self.named_parameters())

                    def gen():
                        for name, weight in weights:
                            # Checkpoint is hybrid vision; DeepseekV4ForCausalLM is text-only backbone
                            if (
                                name.startswith("vision.")
                                or name.startswith("aligner.")
                                or name.startswith("vision_model.")
                                or name.startswith("visual.")
                                or name.startswith("image_")
                            ):
                                continue
                            # Some DSV4 checkpoints store non-routed FP8 source
                            # tensors as BF16.  The model allocates these
                            # parameters as float8; convert only when the
                            # destination parameter explicitly requires it.
                            remapped_name = self.remap_weight_name_to_dpsk_hf_format(
                                name,
                                is_nextn=is_nextn,
                                num_hidden_layers=getattr(self.config, "num_hidden_layers", None),
                            )
                            param = params_dict.get(remapped_name)
                            if param is not None and weight.dtype != param.dtype:
                                float8_dtypes = {
                                    torch.float8_e4m3fn,
                                    torch.float8_e5m2,
                                    getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
                                    getattr(torch, "float8_e5m2fnuz", torch.float8_e5m2),
                                }
                                if param.dtype in float8_dtypes and weight.dtype in {
                                    torch.float16,
                                    torch.bfloat16,
                                }:
                                    weight = weight.to(dtype=param.dtype)
                            yield name, weight
                    return orig_load_weights(self, gen(), is_nextn=is_nextn)

                cls.load_weights = filtered_load_weights
                cls._exl3_weights_patched = True
                logger.info("Patched DeepseekV4ForCausalLM.load_weights for hybrid vision checkpoints")
        return True
    except (ImportError, AttributeError) as exc:
        logger.debug("Failed to patch DeepseekV4ForCausalLM.load_weights: %s", exc)
        return False


def patch_dspark_runtime_gamma() -> bool:
    """Allow DSpark's bundled confidence head to honor a shorter verify window.

    The DSV4 draft checkpoint was trained with ``block_size=5``.  SGLang can
    safely schedule a smaller runtime gamma (for example the vLLM-aligned
    three-token window, gamma=2), but the model's confidence helper still
    reads the checkpoint gamma during CUDA-graph capture.  Temporarily bind
    the model helper to the actual sampled-token width so its reshape and
    Markov context match the runtime schedule.
    """
    try:
        from sglang.srt.models import deepseek_v4_dspark

        cls = deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark
        if getattr(cls, "_exl3_runtime_gamma_patched", False):
            return True
        original = cls.compute_confidence

        def runtime_gamma_compute_confidence(
            self, *, anchor_tokens, sampled_tokens, x_post_hc
        ):
            runtime_gamma = int(sampled_tokens.shape[1])
            checkpoint_gamma = getattr(self, "gamma", runtime_gamma)
            if runtime_gamma == int(checkpoint_gamma):
                return original(
                    self,
                    anchor_tokens=anchor_tokens,
                    sampled_tokens=sampled_tokens,
                    x_post_hc=x_post_hc,
                )
            self.gamma = runtime_gamma
            try:
                return original(
                    self,
                    anchor_tokens=anchor_tokens,
                    sampled_tokens=sampled_tokens,
                    x_post_hc=x_post_hc,
                )
            finally:
                self.gamma = checkpoint_gamma

        cls.compute_confidence = runtime_gamma_compute_confidence
        cls._exl3_runtime_gamma_patched = True
        logger.info("Patched DSpark confidence helper for runtime gamma overrides")
        return True
    except (ImportError, AttributeError) as exc:
        logger.debug("Failed to patch DSpark runtime gamma: %s", exc)
        return False


def patch_model_loader() -> bool:
    """Register 'exl3_mmap' format and EXL3MMapModelLoader into SGLang model loader."""
    try:
        from sglang.srt.model_loader import loader as sml
        from .loader import EXL3MMapModelLoader, LoadFormatEXL3

        # SGLang's SM120 DeepSeek-V4 post-processing enables the TileLang DSA
        # indexer unconditionally.  The bundled CUDA 13 nvcc/headers can be
        # incompatible with the host toolkit, so keep that optional JIT path
        # disabled for EXL3 unless explicitly opted in.
        try:
            from sglang.srt.environ import envs

            tile_flag = envs.SGLANG_OPT_USE_TILELANG_INDEXER
            if not getattr(tile_flag, "_exl3_guarded", False):
                original_set = tile_flag.set

                def guarded_set(value):
                    if value and os.environ.get("SGLANG_EXL3_ALLOW_TILELANG_INDEXER") != "1":
                        return original_set(False)
                    return original_set(value)

                tile_flag.set = guarded_set
                tile_flag._exl3_guarded = True
                original_set(False)
        except (ImportError, AttributeError):
            pass

        # SGLang validates string load formats in LoadConfig.__post_init__ before
        # get_model_loader() is called.  Accept our extension there while
        # preserving the native enum validation for every built-in format.
        try:
            from sglang.srt.configs.load_config import LoadConfig

            if not getattr(LoadConfig, "_exl3_mmap_patched", False):
                orig_verify_load_format = LoadConfig._verify_load_format

                def patched_verify_load_format(self):
                    if isinstance(self.load_format, str) and self.load_format.lower() == "exl3_mmap":
                        self.load_format = LoadFormatEXL3.EXL3_MMAP
                        return
                    return orig_verify_load_format(self)

                LoadConfig._verify_load_format = patched_verify_load_format
                LoadConfig._exl3_mmap_patched = True
        except (ImportError, AttributeError):
            pass

        if hasattr(sml, "get_model_loader"):
            orig_get_model_loader = sml.get_model_loader

            def patched_get_model_loader(load_config, model_config=None):
                lf = getattr(load_config, "load_format", None)
                if lf in ("exl3_mmap", LoadFormatEXL3.EXL3_MMAP):
                    return EXL3MMapModelLoader(load_config)
                return orig_get_model_loader(load_config, model_config)

            sml.get_model_loader = patched_get_model_loader
            logger.info("Patched SGLang get_model_loader with EXL3MMapModelLoader")

        try:
            from sglang.srt import server_args as sa
            if hasattr(sa, "LOAD_FORMAT_CHOICES") and "exl3_mmap" not in sa.LOAD_FORMAT_CHOICES:
                sa.LOAD_FORMAT_CHOICES.append("exl3_mmap")
        except (ImportError, AttributeError):
            pass

        return True
    except (ImportError, AttributeError) as exc:
        logger.debug("Failed to patch SGLang model loader: %s", exc)
        return False


def apply_all_patches() -> bool:
    """Apply all SGLang integration patches."""
    ok_methods = patch_quantization_methods()
    ok_args = patch_server_args()
    ok_dsv4 = patch_deepseek_v4_weights_loading()
    ok_dspark_gamma = patch_dspark_runtime_gamma()
    ok_loader = patch_model_loader()
    return ok_methods or ok_args or ok_dsv4 or ok_dspark_gamma or ok_loader
