"""Register the routed-expert and linear EXL3 implementation with SGLang."""

from __future__ import annotations

from . import exl3
from . import ops
from . import patch
from . import speculative
from . import hadamard_cache
from . import loader

__version__ = "0.2.1"



__all__ = [
    "register",
    "EXL3Config",
    "EXL3LinearMethod",
    "EXL3MoEMethod",
    "get_speculative_draft_tokens",
    "parse_speculative_schedule",
    "is_adaptive_verification_enabled",
    "filter_speculative_candidates",
    "compute_mla_kv_cache_bytes",
    "validate_context_scaling",
    "EXL3MMapModelLoader",
    "exl3_mmap_weights_iterator",
    "LoadFormatEXL3",
]


def register() -> None:
    """Register EXL3 quantization methods, custom ops, and CLI options into SGLang."""
    from . import patch
    from . import ops

    patch.apply_all_patches()
    try:
        ops.register_custom_ops()
    except Exception:
        pass
    try:
        from . import speculative
        speculative.EXL3SpeculativeHook.apply_hooks()
    except Exception:
        pass



# Automatically hook into SGLang when sglang_exl3 is imported
try:
    register()
except Exception:
    # Do not raise on import if SGLang is not installed yet (e.g. during build)
    pass


def __getattr__(name: str):
    if name in {
        "EXL3Config",
        "EXL3LinearMethod",
        "EXL3MoEMethod",
        "get_speculative_draft_tokens",
        "parse_speculative_schedule",
        "is_adaptive_verification_enabled",
        "filter_speculative_candidates",
        "compute_mla_kv_cache_bytes",
        "validate_context_scaling",
    }:
        from . import exl3

        return getattr(exl3, name)
    try:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    except ImportError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
